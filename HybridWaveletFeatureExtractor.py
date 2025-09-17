import numpy as np
import cv2
import dlib
import pywt
import torch
import torch.nn as nn
from skimage.feature import local_binary_pattern, hessian_matrix, hessian_matrix_eigvals
from scipy.fftpack import dct, fft2
from scipy.signal import convolve2d
from typing import Dict, Tuple, List

# phasepack may be absent or return a different signature; provide a compatibility wrapper
try:
    from phasepack import phasecong as _phasecong_raw
except Exception:
    _phasecong_raw = None

import warnings
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2.")
warnings.filterwarnings("ignore", message="Module 'pyfftw' (FFTW Python bindings) could not be imported")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy: _ARRAY_API not found")
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern` to floating-point images may give unexpected results")
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------- Utility functions: robust array / 3rd-party signature handling -----------------
def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Convert input to a 2D float array (without changing information if possible)."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1, 1).astype(np.float32)
    if a.ndim == 1:
        return a.reshape(1, -1).astype(np.float32)
    if a.ndim > 2:
        # Multi-channel -> average to single channel
        return np.mean(a, axis=2).astype(np.float32)
    return a.astype(np.float32)


def safe_phasecong(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compatibility wrapper for different phasecong return signatures; return all-zero matrix if unavailable."""
    img = ensure_2d(image)
    if _phasecong_raw is None:
        return np.zeros_like(img)
    try:
        res = _phasecong_raw(img, **kwargs)
        # May return (pc, EO, T, ph) or only pc or a numpy array
        if isinstance(res, tuple) or isinstance(res, list):
            return np.asarray(res[0])
        return np.asarray(res)
    except Exception:
        # Try the simplest call signature
        try:
            res = _phasecong_raw(img)
            if isinstance(res, (tuple, list)):
                return np.asarray(res[0])
            return np.asarray(res)
        except Exception:
            return np.zeros_like(img)


def safe_hessian_eigvals(detail: np.ndarray):
    """Compatibility wrapper for different return signatures of skimage.feature Hessian functions."""
    det = ensure_2d(detail)
    try:
        H_elems = hessian_matrix(det, sigma=1.0)
        try:
            # Common use: hessian_matrix_eigvals(H_elems) or hessian_matrix_eigvals(*H_elems)
            eig = hessian_matrix_eigvals(H_elems)
        except Exception:
            eig = hessian_matrix_eigvals(*H_elems)
        # eig might be tuple (k1,k2) or ndarray shape (2,h,w)
        if isinstance(eig, tuple) or isinstance(eig, list):
            k1, k2 = eig[0], eig[1]
        else:
            eig = np.asarray(eig)
            if eig.ndim >= 3 and eig.shape[0] == 2:
                k1, k2 = eig[0], eig[1]
            elif eig.ndim == 2:
                # Less common signature: two channels concatenated in a 2D matrix
                k1 = eig
                k2 = np.zeros_like(k1)
            else:
                k1 = np.zeros_like(det)
                k2 = np.zeros_like(det)
        return k1, k2
    except Exception:
        # On failure return zeros
        return np.zeros_like(det), np.zeros_like(det)


# ----------------- 1. 68-point regions definition and geometric feature calculation -----------------
class FaceGeometryAnalyzer:
    REGION_MAPPING = {
        "jaw": list(range(0, 17)),
        "left_eyebrow": list(range(17, 22)),
        "right_eyebrow": list(range(22, 27)),
        "nose_bridge": list(range(27, 31)),
        "nose_tip": list(range(31, 36)),
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "lips": list(range(48, 60)),
        "inner_lips": list(range(60, 68))
    }

    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except Exception as e:
            # If predictor file is unavailable, allow construction but further calls will raise
            print(f"[Warning] Failed to load shape predictor: {e}")
            self.predictor = None
        self.region_masks = None

    def get_landmarks(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            raise ValueError("shape_predictor_68_face_landmarks.dat failed to load")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        if not faces:
            raise ValueError("No face detected")
        shape = self.predictor(gray, faces[0])
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

    def create_region_masks(self, img_shape: Tuple[int, int], landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        masks = {}
        h, w = img_shape[:2]
        for name, indices in self.REGION_MAPPING.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = landmarks[indices].astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 1, 2)
            masks[name] = mask
        self.region_masks = masks
        return masks

    def calculate_region_curvature(self, landmarks: np.ndarray) -> Dict[str, float]:
        region_curvature = {}
        for name, indices in self.REGION_MAPPING.items():
            pts = landmarks[indices]
            if len(pts) < 3:
                region_curvature[name] = 0.0
                continue

            dx = np.gradient(pts[:, 0])
            dy = np.gradient(pts[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
            region_curvature[name] = float(np.mean(curvature))

        return region_curvature

    def calculate_region_smoothness(self, gray_img: np.ndarray) -> Dict[str, float]:
        if self.region_masks is None:
            raise ValueError("Region masks not created. Call create_region_masks first.")

        region_smoothness = {}
        h, w = gray_img.shape
        # Spectrum
        try:
            f = np.abs(fft2(gray_img.astype(np.float32)))
        except Exception:
            # Fallback: simple Sobel energy estimate
            gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1)
            f = np.hypot(gx, gy)

        mask_center = np.zeros((h, w), np.uint8)
        crow, ccol = h // 2, w // 2
        cv2.circle(mask_center, (ccol, crow), min(h, w) // 8, 1, -1)

        for name, mask in self.region_masks.items():
            if np.sum(mask) == 0:
                region_smoothness[name] = 0.0
                continue

            # Convert mask to the same shape/type as the spectrum
            maskf = mask.astype(np.float32)
            low = np.sum(f * maskf * mask_center)
            high = np.sum(f * maskf * (1 - mask_center))
            region_smoothness[name] = float(low / (low + high + 1e-8))

        return region_smoothness


# ----------------- 2. Dynamic weight learner -----------------
class DynamicWeightLearner(nn.Module):
    def __init__(self, num_regions: int = 9):
        super().__init__()
        try:
            # Conservative default tensor type setup (avoid string form causing issues in some environments)
            if not torch.cuda.is_available():
                try:
                    torch.set_default_tensor_type(torch.FloatTensor)
                except Exception:
                    pass

            self.wS = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))
            self.wC = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))
        except Exception as e:
            print(f"[Warning] Parameter initialization failed: {e}, using fallback initialization")
            self.wS = nn.Parameter(torch.tensor([1.0 for _ in range(num_regions)], dtype=torch.float32))
            self.wC = nn.Parameter(torch.tensor([1.0 for _ in range(num_regions)], dtype=torch.float32))

    def forward(self, S: Dict[str, float], C: Dict[str, float]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        region_weights = {}
        regions = list(S.keys())

        for i, region in enumerate(regions):
            s_val = torch.tensor(float(S.get(region, 0.0)), dtype=torch.float32)
            c_val = torch.tensor(float(C.get(region, 0.0)), dtype=torch.float32)

            alpha = torch.sigmoid(self.wS[i] * s_val - self.wC[i] * c_val)
            beta = 1.0 - alpha

            region_weights[region] = (alpha.clamp(0.1, 0.9), beta.clamp(0.1, 0.9))

        return region_weights


# ----------------- 3. Hybrid wavelet tensor decomposition -----------------
class HybridWaveletTransformer:
    def __init__(self, levels=3):
        self.levels = levels
        self.coif_wavelet = 'coif1'
        self.haar_wavelet = 'haar'

    def _safe_wavedec2(self, img: np.ndarray, wavelet: str):
        """Try pywt.wavedec2; on failure fall back to a simple pyramid approximation to keep runnable."""
        img2 = ensure_2d(img)
        try:
            coeffs = pywt.wavedec2(img2, wavelet, level=self.levels)
            return coeffs
        except Exception as e:
            # Fallback: construct a list approximating wavedec2 structure
            print(f"[Fallback] pywt.wavedec2({wavelet}) failed: {e}; falling back to pyrDown implementation")
            coeffs = []
            ll = img2.copy()
            for _ in range(self.levels):
                # Simulate low-frequency approximation
                ll = cv2.pyrDown(ll)
            coeffs.append(ll)
            for _ in range(self.levels):
                h, w = ll.shape[:2]
                coeffs.append((np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))))
            return coeffs

    def decompose_separate(self, img: np.ndarray) -> Tuple[List, List, List, List]:
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got {img.ndim}D instead")

        coeffs_x_coif = self._safe_wavedec2(img, self.coif_wavelet)
        coeffs_x_haar = self._safe_wavedec2(img, self.haar_wavelet)
        coeffs_y_coif = self._safe_wavedec2(img, self.coif_wavelet)
        coeffs_y_haar = self._safe_wavedec2(img, self.haar_wavelet)

        return (coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar)

    def tensor_product_merge(self,
                             coeffs_x_coif: List, coeffs_x_haar: List,
                             coeffs_y_coif: List, coeffs_y_haar: List,
                             region_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                             region_masks: Dict[str, np.ndarray]) -> List:
        ll = coeffs_x_coif[0].copy() if isinstance(coeffs_x_coif[0], np.ndarray) else np.array(coeffs_x_coif[0])
        mixed_coeffs = [ll]

        for level in range(1, self.levels + 1):
            try:
                c_x = coeffs_x_coif[level]
                h_x = coeffs_x_haar[level]
                c_y = coeffs_y_coif[level]
                h_y = coeffs_y_haar[level]

                term1 = self._compute_tensor_product(c_x, h_y)
                term2 = self._compute_tensor_product(h_x, c_y)

                term1 = self._ensure_compatible_shapes(term1, term2)

                # Initialize weight_mask to the same shape as term1[0]
                base_shape = term1[0].shape
                weight_mask = np.zeros(base_shape, dtype=np.float32)

                for region, (alpha, beta) in region_weights.items():
                    mask = region_masks.get(region)
                    if mask is None:
                        continue
                    mask_scaled = cv2.resize(mask.astype(np.float32), (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
                    # broadcast if needed
                    mask_scaled = ensure_2d(mask_scaled)
                    # term1/term2 each channel may be 2D or 3D; merge mainly on the 0th channel
                    contribution_0 = mask_scaled * (float(alpha.item()) * term1[0] + float(beta.item()) * term2[0])
                    # If contribution_0 is not the same shape as weight_mask (safety)
                    if contribution_0.shape != weight_mask.shape:
                        contribution_0 = cv2.resize(contribution_0, (weight_mask.shape[1], weight_mask.shape[0]), interpolation=cv2.INTER_AREA)
                    weight_mask += contribution_0

                # Second and third components weighted by channels if exist
                ch1 = float(alpha.item()) * term1[1] + float(beta.item()) * term2[1]
                ch2 = float(alpha.item()) * term1[2] + float(beta.item()) * term2[2]

                mixed_level = (
                    weight_mask,
                    ch1,
                    ch2
                )
                mixed_coeffs.append(mixed_level)
            except Exception as e:
                print(f"[Warning] Decomposition failed at level {level}: {e}, skipping this level")
                continue

        return mixed_coeffs

    @staticmethod
    def _compute_tensor_product(x, y):
        # x,y expected to be triplets (LH, HL, HH) or similar
        try:
            a0 = ensure_2d(x[0])
            b0 = ensure_2d(y[0])
            # outer product -> try to produce usable 2D/3D result
            t0 = np.tensordot(a0, b0, axes=0)
            # squeeze may collapse shapes; attempt reasonable reshape if possible
            if t0.ndim >= 2:
                t0 = t0.reshape(a0.shape[0], b0.shape[1] if b0.ndim > 1 else b0.shape[0])
        except Exception:
            t0 = np.zeros_like(ensure_2d(x[0]))

        try:
            a1 = ensure_2d(x[1])
            b1 = ensure_2d(y[1])
            t1 = np.tensordot(a1, b1, axes=0)
            if t1.ndim >= 2:
                t1 = t1.reshape(a1.shape[0], b1.shape[1] if b1.ndim > 1 else b1.shape[0])
        except Exception:
            t1 = np.zeros_like(ensure_2d(x[1]))

        try:
            a2 = ensure_2d(x[2])
            b2 = ensure_2d(y[2])
            t2 = np.tensordot(a2, b2, axes=0)
            if t2.ndim >= 2:
                t2 = t2.reshape(a2.shape[0], b2.shape[1] if b2.ndim > 1 else b2.shape[0])
        except Exception:
            t2 = np.zeros_like(ensure_2d(x[2]))

        return (t0, t1, t2)

    @staticmethod
    def _ensure_compatible_shapes(term1, term2):
        adjusted_term1 = list(term1)
        for i in range(len(term1)):
            if term1[i].shape != term2[i].shape:
                adjusted_term1[i] = HybridWaveletTransformer._resize_tensor(term1[i], term2[i].shape)
        return tuple(adjusted_term1)

    @staticmethod
    def _resize_tensor(tensor, target_shape):
        tensor = ensure_2d(tensor)
        if tensor.shape == target_shape:
            return tensor
        try:
            return cv2.resize(tensor, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            # Safest: fill zeros
            out = np.zeros(target_shape, dtype=tensor.dtype)
            h = min(tensor.shape[0], target_shape[0])
            w = min(tensor.shape[1], target_shape[1])
            out[:h, :w] = tensor[:h, :w]
            return out


# ----------------- 4. Three-level feature extractor -----------------
class ThreeLevelFeatureExtractor:
    def __init__(self):
        self.lbp_n_points = 8
        self.lbp_radius = 1
        self.lbp_method = 'uniform'
        self.srm_filters = [
            np.atleast_2d(np.array([[-1, 2, -1]])),
            np.atleast_2d(np.array([[-1], [2], [-1]])),
            np.atleast_2d(np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]])),
            np.atleast_2d(np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]))
        ]

    def extract_level1(self, coeffs: List) -> np.ndarray:
        try:
            if len(coeffs) < 2:
                raise ValueError("Not enough coefficient levels to extract Level 1 features")

            level1 = coeffs[1]
            if isinstance(level1, tuple) and len(level1) == 3:
                hh1 = level1[2]
            elif isinstance(level1, list) and len(level1) >= 3:
                hh1 = level1[2]
            else:
                hh1 = level1[-1] if isinstance(level1, (list, tuple)) else level1

            hh1 = ensure_2d(hh1)
            hh1_int = ((hh1 - np.min(hh1)) / (np.max(hh1) - np.min(hh1) + 1e-8) * 255).astype(np.uint8)
            weight_map = np.abs(hh1) / (np.max(np.abs(hh1)) + 1e-8)

            features = []
            for kernel in self.srm_filters:
                kernel2 = np.atleast_2d(kernel.astype(np.float32))
                filtered = convolve2d(hh1, kernel2, mode='same', boundary='symm')
                filtered = filtered * (1 + weight_map)
                # Avoid std=0 for constant images
                if np.max(filtered) - np.min(filtered) < 1e-8:
                    filtered_int = np.zeros_like(hh1_int)
                else:
                    filtered_int = ((filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8) * 255).astype(np.uint8)

                lbp = local_binary_pattern(
                    filtered_int,
                    self.lbp_n_points,
                    self.lbp_radius,
                    method=self.lbp_method
                )
                # 'uniform' has 59 bins; try to be compatible if method differs
                bins = 59
                hist, _ = np.histogram(lbp, bins=bins, range=(0, bins - 1))
                features.append(hist.astype(np.float32))

            return np.concatenate(features) if features else np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"[Warning] Level 1 feature extraction failed: {e}, returning empty feature")
            return np.zeros(4 * 59, dtype=np.float32)

    def extract_level2(self, coeffs: List) -> np.ndarray:
        try:
            if len(coeffs) >= 3:
                level_data = coeffs[2]
            else:
                level_data = coeffs[1]

            # Flexible parsing
            if isinstance(level_data, (list, tuple)):
                if len(level_data) == 3:
                    lh2, hl2, hh2 = level_data[:3]
                elif len(level_data) > 2 and isinstance(level_data[2], (list, tuple)):
                    lh2, hl2, hh2 = level_data[2][:3]
                else:
                    # take first three or duplicate
                    parts = list(level_data) + [level_data[-1]] * 3
                    lh2, hl2, hh2 = parts[:3]
            else:
                h, w = level_data.shape[:2]
                lh2 = hl2 = hh2 = np.zeros((h, w))

            lh2 = ensure_2d(lh2)
            hl2 = ensure_2d(hl2)
            hh2 = ensure_2d(hh2)

            detail = lh2 + hl2 + hh2

            kappa1, kappa2 = safe_hessian_eigvals(detail)

            # Phase congruency (compatible signatures)
            pc = safe_phasecong(detail, nscale=4, norient=6)
            weighted_curvature = ((kappa1 + kappa2) / 2.0) * pc

            # Extract statistics
            flat = weighted_curvature.flatten()
            flat = flat[~np.isnan(flat)]
            if flat.size == 0:
                return np.zeros(6, dtype=np.float32)
            return np.array([
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.percentile(flat, 25)),
                float(np.percentile(flat, 75)),
                float(np.max(flat)),
                float(np.min(flat))
            ], dtype=np.float32)
        except Exception as e:
            print(f"[Warning] Level 2 feature extraction failed: {e}, returning default features")
            return np.zeros(6, dtype=np.float32)

    def extract_level3(self, coeffs: List) -> np.ndarray:
        try:
            ll3 = coeffs[0]
            ll3 = ensure_2d(ll3)

            # FFT energy statistics
            try:
                fft_mag = np.abs(fft2(ll3))
            except Exception:
                fft_mag = np.abs(np.fft.fft2(ll3))

            fft_feat = [
                float(np.mean(fft_mag)),
                float(np.std(fft_mag)),
                float(np.percentile(fft_mag, 25)),
                float(np.percentile(fft_mag, 75)),
                float(np.sum(fft_mag[:fft_mag.shape[0] // 4, :fft_mag.shape[1] // 4]))
            ]

            # DCT coefficient statistics
            try:
                dct_coef = dct(dct(ll3.T, norm='ortho').T, norm='ortho')
            except Exception:
                dct_coef = np.zeros_like(ll3)
            dct_feat = [
                float(np.mean(dct_coef)),
                float(np.std(dct_coef)),
                float(np.percentile(dct_coef, 25)),
                float(np.percentile(dct_coef, 75)),
                float(np.sum(dct_coef[:dct_coef.shape[0] // 4, :dct_coef.shape[1] // 4]))
            ]

            return np.array(fft_feat + dct_feat, dtype=np.float32)
        except Exception as e:
            print(f"[Warning] Level 3 feature extraction failed: {e}, returning default features")
            return np.zeros(10, dtype=np.float32)

    def extract_cross_scale(self, coeffs: List) -> np.ndarray:
        try:
            ll3 = coeffs[0]
            if len(coeffs) >= 2:
                level1 = coeffs[1]
                if isinstance(level1, (list, tuple)):
                    hh1 = level1[-1] if len(level1) >= 3 else level1[0]
                else:
                    hh1 = level1
            else:
                hh1 = np.zeros_like(ll3)

            ll3 = ensure_2d(ll3)
            hh1 = ensure_2d(hh1)

            dct_ll3 = dct(dct(ll3.T, norm='ortho').T, norm='ortho')
            dct_hist, _ = np.histogram(dct_ll3.flatten(), bins=32, range=(-10, 10))

            hh1_int = ((hh1 - np.min(hh1)) / (np.max(hh1) - np.min(hh1) + 1e-8) * 255).astype(np.uint8)
            lbp_hh1 = local_binary_pattern(
                hh1_int,
                self.lbp_n_points,
                self.lbp_radius,
                method=self.lbp_method
            )
            lbp_hist, _ = np.histogram(lbp_hh1.flatten(), bins=32, range=(0, 58))

            # Cross-scale correlation (careful with short lengths or NaNs)
            try:
                a = dct_ll3.flatten()
                b = lbp_hh1.flatten()
                # Truncate to same length and non-NaN
                n = min(a.size, b.size, 1000)
                if n < 2:
                    cross_corr = 0.0
                else:
                    a = a[:n]
                    b = b[:n]
                    if np.isnan(a).any() or np.isnan(b).any():
                        cross_corr = 0.0
                    else:
                        cross_corr = float(np.corrcoef(a, b)[0, 1])
                        if not np.isfinite(cross_corr):
                            cross_corr = 0.0
            except Exception:
                cross_corr = 0.0

            return np.concatenate([dct_hist.astype(np.float32), lbp_hist.astype(np.float32), np.array([cross_corr], dtype=np.float32)])
        except Exception as e:
            print(f"[Warning] Cross-scale feature extraction failed: {e}, returning default features")
            return np.zeros(65, dtype=np.float32)


# ----------------- 5. Feature extractor integration -----------------
class HybridWaveletFeatureExtractor:
    def __init__(self, feature_dim=512, wavelet_levels=2):
        super().__init__()
        self.geometry_analyzer = FaceGeometryAnalyzer()
        self.weight_learner = DynamicWeightLearner(num_regions=len(self.geometry_analyzer.REGION_MAPPING))
        self.wavelet_transformer = HybridWaveletTransformer(levels=wavelet_levels)
        self.feature_extractor = ThreeLevelFeatureExtractor()

        # Projection layer: input 317 -> projection
        self.projection = nn.Sequential(
            nn.Linear(317, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def extract_features(self, img_bgr: np.ndarray) -> torch.Tensor:
        # 1. Face geometry analysis
        try:
            landmarks = self.geometry_analyzer.get_landmarks(img_bgr)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = (gray - gray.mean()) / (gray.std() + 1e-8)

            if gray.ndim != 2:
                gray = np.mean(gray, axis=2)

            region_masks = self.geometry_analyzer.create_region_masks(gray.shape, landmarks)
            region_smoothness = self.geometry_analyzer.calculate_region_smoothness(gray)
            region_curvature = self.geometry_analyzer.calculate_region_curvature(landmarks)
            region_weights = self.weight_learner(region_smoothness, region_curvature)

        except Exception as e:
            print(f"[Warning] Geometry analysis failed: {e}, using default weights")
            h, w = (gray.shape if 'gray' in locals() else (img_bgr.shape[0], img_bgr.shape[1]))
            region_weights = {
                name: (torch.tensor(0.5), torch.tensor(0.5))
                for name in self.geometry_analyzer.REGION_MAPPING.keys()
            }
            region_masks = {
                name: (np.ones((h, w), dtype=np.uint8) / len(self.geometry_analyzer.REGION_MAPPING)).astype(np.float32)
                for name in self.geometry_analyzer.REGION_MAPPING.keys()
            }

        # 2. Hybrid wavelet decomposition
        try:
            coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar = self.wavelet_transformer.decompose_separate(
                gray)
            mixed_coeffs = self.wavelet_transformer.tensor_product_merge(
                coeffs_x_coif, coeffs_x_haar,
                coeffs_y_coif, coeffs_y_haar,
                region_weights, region_masks
            )
        except Exception as e:
            print(f"[Warning] Wavelet decomposition failed: {e}, using simplified features")
            h, w = gray.shape
            mixed_coeffs = [np.zeros((h // 4 if h >= 4 else 1, w // 4 if w >= 4 else 1), dtype=np.float32)]
            mixed_coeffs.append((np.zeros_like(mixed_coeffs[0]), np.zeros_like(mixed_coeffs[0]), np.zeros_like(mixed_coeffs[0])))

        # 3. Multi-level feature extraction
        features = [
            self.feature_extractor.extract_level1(mixed_coeffs),
            self.feature_extractor.extract_level2(mixed_coeffs),
            self.feature_extractor.extract_level3(mixed_coeffs),
            self.feature_extractor.extract_cross_scale(mixed_coeffs)
        ]

        # Ensure feature dimension is correct (317)
        try:
            features = np.concatenate([f.flatten() for f in features]).astype(np.float32)
            if len(features) != 317:
                print(f"[Info] Feature dimension mismatch: {len(features)}, adjusting to 317")
                if len(features) < 317:
                    features = np.pad(features, (0, 317 - len(features)), mode='constant')
                else:
                    features = features[:317]
        except Exception as e:
            print(f"[Warning] Feature concatenation failed: {e}, using default feature vector")
            features = np.zeros(317, dtype=np.float32)

        # 4. Feature projection (robustly handle numpy/tensor)
        try:
            if isinstance(features, np.ndarray):
                feat_tensor = torch.from_numpy(features).float().unsqueeze(0)
            else:
                feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            projected = self.projection(feat_tensor)
            return projected.squeeze()
        except Exception as e:
            print(f"[Warning] Feature projection failed: {e}, returning raw feature tensor")
            # Final fallback: return a torch tensor (length 317)
            try:
                return torch.tensor(features, dtype=torch.float32)
            except Exception:
                return torch.zeros(317, dtype=torch.float32)


# ----------------- 6. Usage example -----------------
if __name__ == "__main__":
    try:
        extractor = HybridWaveletFeatureExtractor(wavelet_levels=1)

        img = cv2.imread("test_face.jpg")
        if img is None:
            raise FileNotFoundError("Test image 'test_face.jpg' not found")

        with torch.no_grad():
            features = extractor.extract_features(img)
            # features may be a 1D tensor
            feat = features
            # If it's a torch tensor
            if isinstance(feat, torch.Tensor):
                f_np = feat.detach().cpu().numpy()
            else:
                f_np = np.asarray(feat)

            print("dtype, shape:", f_np.dtype, f_np.shape)
            print("finite count:", np.isfinite(f_np).sum(), "/", f_np.size)
            print("nan count:", np.isnan(f_np).sum())
            print("inf count:", np.isinf(f_np).sum())
            print("min, 1pct, median, 99pct, max:",
                  np.nanmin(f_np), np.nanpercentile(f_np, 1),
                  np.nanmedian(f_np), np.nanpercentile(f_np, 99), np.nanmax(f_np))
            print("L2 norm:", np.linalg.norm(np.nan_to_num(f_np)))
            # Simple text histogram
            hist, bins = np.histogram(np.nan_to_num(f_np), bins=10)
            print("hist bins:", bins)
            print("hist counts:", hist)
            print(f"Extracted feature vector shape: {tuple(features.shape) if hasattr(features, 'shape') else 'unknown'}")
            try:
                print(f"Feature vector norm: {torch.norm(features):.4f}")
            except Exception:
                print("Failed to compute norm (type issue); returned feature tensor.")
    except Exception as e:
        print(f"Main program error: {e}")
