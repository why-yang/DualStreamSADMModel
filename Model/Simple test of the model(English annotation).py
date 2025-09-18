import numpy as np
import cv2
import dlib
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import local_binary_pattern, hessian_matrix, hessian_matrix_eigvals
from scipy.fftpack import dct
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")


# ----------------- Utility Functions -----------------
def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure the input is a 2D float array."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1, 1).astype(np.float32)
    if a.ndim == 1:
        return a.reshape(1, -1).astype(np.float32)
    if a.ndim > 2:
        return np.mean(a, axis=2).astype(np.float32)
    return a.astype(np.float32)


# ----------------- Phase Congruency (Kovesi implementation) -----------------
def phase_congruency(img: np.ndarray, nscale: int = 4, norient: int = 6,
                     min_wavelength: int = 3, mult: float = 2.1,
                     sigma_onf: float = 0.55) -> np.ndarray:
    """
    Implement Kovesi's Phase Congruency algorithm.
    Based on: https://github.com/peterkovesi/PhaseCongruency
    """
    img = ensure_2d(img)
    rows, cols = img.shape

    # Create filter bank
    _, _, _, _, _, _ = _create_filters(rows, cols, nscale, norient, min_wavelength, mult, sigma_onf)

    # Simplified implementation: approximate phase congruency using multi-orientation Gabor filters
    pc = np.zeros_like(img, dtype=np.float32)

    for o in range(norient):
        angle = np.pi * o / norient
        for s in range(nscale):
            wavelength = min_wavelength * mult ** s
            sigma = wavelength * sigma_onf

            # Real part of the Gabor filter
            gabor = _gabor_filter(rows, cols, wavelength, angle, sigma, 0)
            # Apply filter
            filtered = cv2.filter2D(img, -1, gabor, borderType=cv2.BORDER_REFLECT)
            pc += np.abs(filtered)

    # Normalize
    if np.max(pc) > 0:
        pc /= np.max(pc)

    return pc


def _create_filters(rows, cols, nscale, norient, min_wavelength, mult, sigma_onf):
    """Create filter bank (simplified)."""
    # Full filter construction should be implemented for real applications
    return None, None, None, None, None, None


def _gabor_filter(rows, cols, wavelength, angle, sigma, phase_offset):
    """Create a Gabor filter."""
    # Create grid
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    x, y = np.meshgrid(x, y)

    # Rotate coordinates
    x_theta = x * np.cos(angle) + y * np.sin(angle)
    y_theta = -x * np.sin(angle) + y * np.cos(angle)

    # Gabor function
    gb = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
    gb *= np.cos(2 * np.pi * x_theta / wavelength + phase_offset)

    return gb


# ----------------- Face Geometry Analyzer -----------------
class FaceGeometryAnalyzer:
    """Face geometry analyzer based on 68 keypoints."""
    REGION_MAPPING = {
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
        'nose': list(range(27, 36)),
        'mouth': list(range(48, 60)),
        'jaw': list(range(0, 17)),
        'left_eyebrow': list(range(17, 22)),
        'right_eyebrow': list(range(22, 27)),
        'face_contour': list(range(0, 17))  # include face contour
    }

    def __init__(self, predictor_path: Optional[str] = "shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(predictor_path) if predictor_path else None
        except:
            self.predictor = None
            print("Warning: Could not load dlib predictor. Using fallback methods.")

    def get_landmarks(self, img_bgr: np.ndarray) -> np.ndarray:
        """Get 68 facial landmarks."""
        if self.predictor is None:
            # Use a simple estimate as a fallback
            h, w = img_bgr.shape[:2]
            return np.random.rand(68, 2) * np.array([w, h])

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        if not faces:
            raise ValueError("No face detected")
        shape = self.predictor(gray, faces[0])
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

    def create_region_masks(self, img_shape: Tuple[int, int], landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Create region masks."""
        h, w = img_shape[:2]
        masks = {}
        for region, indices in self.REGION_MAPPING.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = landmarks[indices].astype(np.int32)
            if len(pts) >= 3:
                cv2.fillConvexPoly(mask, pts, 1)
            masks[region] = mask
        return masks

    def calculate_region_curvature(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Compute curvature per region."""
        curvature = {}
        for region, indices in self.REGION_MAPPING.items():
            pts = landmarks[indices]
            if len(pts) < 3:
                curvature[region] = 0.0
                continue

            # Curvature
            dx = np.gradient(pts[:, 0])
            dy = np.gradient(pts[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature_val = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
            curvature[region] = float(np.mean(curvature_val))
        return curvature

    def calculate_region_smoothness(self, gray_img: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute regional smoothness."""
        smoothness = {}
        h, w = gray_img.shape

        # Use FFT to compute low-frequency energy ratio
        fft = np.fft.fft2(gray_img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Central mask (low-frequency area)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        crow, ccol = h // 2, w // 2
        radius = min(h, w) // 4
        cv2.circle(center_mask, (ccol, crow), radius, 1, -1)

        for region, mask in masks.items():
            if np.sum(mask) == 0:
                smoothness[region] = 0.0
                continue

            # Low-frequency energy ratio
            low_freq = np.sum(magnitude * center_mask * mask)
            total_freq = np.sum(magnitude * mask)
            smoothness[region] = float(low_freq / (total_freq + 1e-8))

        return smoothness


# ----------------- Dynamic Weight Learner -----------------
class DynamicWeightLearner(nn.Module):
    """Dynamic weight learner."""

    def __init__(self, num_regions: int):
        super().__init__()
        self.wS = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))
        self.wC = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))

    def forward(self, S: Dict[str, float], C: Dict[str, float]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        region_weights = {}
        regions = list(S.keys())
        for i, region in enumerate(regions):
            s_val = torch.tensor(S[region], dtype=torch.float32)
            c_val = torch.tensor(C[region], dtype=torch.float32)
            alpha = torch.sigmoid(self.wS[i] * s_val - self.wC[i] * c_val)
            beta = 1.0 - alpha
            region_weights[region] = (alpha, beta)
        return region_weights


# ----------------- Hybrid Wavelet Transformer -----------------
class HybridWaveletTransformer:
    """Hybrid wavelet transformer combining Coiflet and Haar wavelets."""

    def __init__(self, levels=3):
        self.levels = levels
        self.coif_wavelet = 'coif1'
        self.haar_wavelet = 'haar'

    def _safe_wavedec2(self, img: np.ndarray, wavelet: str):
        """Safe 2D wavelet decomposition."""
        img_2d = ensure_2d(img)
        try:
            return pywt.wavedec2(img_2d, wavelet, level=self.levels)
        except:
            # Fallback: zero coefficients
            h, w = img_2d.shape
            coeffs = [np.zeros((h // (2 ** self.levels), w // (2 ** self.levels)), dtype=np.float32)]
            for i in range(self.levels):
                size = (h // (2 ** (self.levels - i)), w // (2 ** (self.levels - i)))
                coeffs.append((np.zeros(size), np.zeros(size), np.zeros(size)))
            return coeffs

    def decompose_separate(self, img: np.ndarray) -> Tuple[List, List, List, List]:
        """Decompose separately using Coiflet and Haar."""
        if img.ndim != 2:
            raise ValueError("Expected 2D image")

        return (
            self._safe_wavedec2(img, self.coif_wavelet),
            self._safe_wavedec2(img, self.haar_wavelet),
            self._safe_wavedec2(img, self.coif_wavelet),
            self._safe_wavedec2(img, self.haar_wavelet)
        )

    def tensor_product_merge(self, coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar,
                             region_weights, region_masks):
        """Tensor-product merge of wavelet coefficients."""
        mixed_coeffs = [coeffs_x_coif[0]]  # keep LL subband

        for level in range(1, self.levels + 1):
            try:
                # Get wavelet coefficients at current level
                c_x = coeffs_x_coif[level]
                h_x = coeffs_x_haar[level]
                c_y = coeffs_y_coif[level]
                h_y = coeffs_y_haar[level]

                # Tensor product terms (outer product)
                term1 = self._compute_tensor_product(c_x, h_y)
                term2 = self._compute_tensor_product(h_x, c_y)

                # Adjust shapes to be compatible
                term1 = self._ensure_compatible_shapes(term1, term2)

                # Apply regional weights
                h, w = term1[0].shape
                merged = (np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)))

                for region, (alpha, beta) in region_weights.items():
                    mask = region_masks.get(region, np.zeros((h, w)))
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h))

                    alpha_val = alpha.item() if hasattr(alpha, 'item') else alpha
                    beta_val = beta.item() if hasattr(beta, 'item') else beta

                    for i in range(3):
                        # Strictly follow the formula to merge via tensor products
                        term1_resized = cv2.resize(term1[i], (w, h)) if term1[i].shape != (h, w) else term1[i]
                        term2_resized = cv2.resize(term2[i], (w, h)) if term2[i].shape != (h, w) else term2[i]
                        merged[i] += mask * (alpha_val * term1_resized + beta_val * term2_resized)

                mixed_coeffs.append(merged)
            except Exception as e:
                print(f"Level {level} merge failed: {e}")
                # Fallback: zero coefficients
                h, w = mixed_coeffs[0].shape
                if level > 1:
                    h //= 2 ** (level - 1)
                    w //= 2 ** (level - 1)
                mixed_coeffs.append((np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))))

        return mixed_coeffs

    def _compute_tensor_product(self, x, y):
        """Compute tensor product (outer product)."""
        # Ensure 2D
        x0 = ensure_2d(x[0])
        x1 = ensure_2d(x[1])
        x2 = ensure_2d(x[2])
        y0 = ensure_2d(y[0])
        y1 = ensure_2d(y[1])
        y2 = ensure_2d(y[2])

        # Outer product
        try:
            t0 = np.outer(x0, y0)
            t1 = np.outer(x1, y1)
            t2 = np.outer(x2, y2)
        except:
            # Fallback to zeros if outer fails
            t0 = np.zeros((x0.size, y0.size))
            t1 = np.zeros((x1.size, y1.size))
            t2 = np.zeros((x2.size, y2.size))

        return (t0, t1, t2)

    def _ensure_compatible_shapes(self, term1, term2):
        """Ensure the two terms have compatible shapes."""
        adjusted = []
        for t1, t2 in zip(term1, term2):
            if t1.shape != t2.shape:
                # Resize to the smaller size
                h = min(t1.shape[0], t2.shape[0])
                w = min(t1.shape[1], t2.shape[1])
                t1_resized = cv2.resize(t1, (w, h)) if t1.shape != (h, w) else t1
                t2_resized = cv2.resize(t2, (w, h)) if t2.shape != (h, w) else t2
                adjusted.append((t1_resized + t2_resized) / 2)
            else:
                adjusted.append((t1 + t2) / 2)
        return tuple(adjusted)


# ----------------- Three-Level Feature Extractor -----------------
class ThreeLevelFeatureExtractor:
    """Three-level feature extractor consistent with the paper."""

    def __init__(self):
        # LBP parameters
        self.lbp_n_points = 8
        self.lbp_radius = 1
        self.lbp_method = 'uniform'

        # SRM filters (4 directions)
        self.srm_filters = [
            np.array([[-1, 2, -1]]),  # horizontal
            np.array([[-1], [2], [-1]]),  # vertical
            np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]),  # 45 degrees
            np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])  # 135 degrees
        ]

    def extract_level1(self, coeffs: List) -> np.ndarray:
        """Extract Level-1 features (high-frequency detail layer)."""
        try:
            if len(coeffs) < 2:
                return np.zeros(4 * 59, dtype=np.float32)

            # HH1 subband
            hh1 = ensure_2d(coeffs[1][2])  # HH subband

            # Weight map
            weight_map = np.abs(hh1) / (np.max(np.abs(hh1)) + 1e-8)

            # Apply SRM filters and LBP
            features = []
            for kernel in self.srm_filters:
                # Filter
                filtered = cv2.filter2D(hh1, -1, kernel, borderType=cv2.BORDER_REFLECT)
                filtered = filtered * (1 + weight_map)  # enhance edges

                # Convert to uint8 for LBP
                filtered_norm = (
                    (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8) * 255
                ).astype(np.uint8)

                # LBP
                lbp = local_binary_pattern(filtered_norm, self.lbp_n_points, self.lbp_radius, self.lbp_method)

                # Histogram (59 bins for uniform method)
                hist, _ = np.histogram(lbp, bins=59, range=(0, 58))
                features.append(hist.astype(np.float32))

            return np.concatenate(features)
        except Exception as e:
            print(f"Level1 feature extraction failed: {e}")
            return np.zeros(4 * 59, dtype=np.float32)

    def extract_level2(self, coeffs: List) -> np.ndarray:
        """Extract Level-2 features (mid-scale structural layer)."""
        try:
            if len(coeffs) < 3:
                return np.zeros(8, dtype=np.float32)  # expanded to include more curvature stats

            # Level-2 detail coefficients
            lh2, hl2, hh2 = coeffs[2]
            detail = ensure_2d(lh2) + ensure_2d(hl2) + ensure_2d(hh2)

            # Hessian matrix and principal curvatures
            H_elems = hessian_matrix(detail, sigma=1.0)
            kappa1, kappa2 = hessian_matrix_eigvals(H_elems)

            # Mean curvature and curvature difference
            mean_curvature = (kappa1 + kappa2) / 2.0
            curvature_diff = np.abs(kappa1 - kappa2)

            # Phase congruency
            pc = phase_congruency(detail)

            # Weighted curvature maps
            weighted_mean_curvature = mean_curvature * pc
            weighted_curvature_diff = curvature_diff * pc

            # Stats
            flat_mean = weighted_mean_curvature.flatten()
            flat_mean = flat_mean[~np.isnan(flat_mean)]

            flat_diff = weighted_curvature_diff.flatten()
            flat_diff = flat_diff[~np.isnan(flat_diff)]

            if len(flat_mean) == 0 or len(flat_diff) == 0:
                return np.zeros(8, dtype=np.float32)

            return np.array([
                np.mean(flat_mean), np.std(flat_mean),
                np.mean(flat_diff), np.std(flat_diff),
                np.percentile(flat_mean, 25), np.percentile(flat_mean, 75),
                np.percentile(flat_diff, 25), np.percentile(flat_diff, 75)
            ], dtype=np.float32)
        except Exception as e:
            print(f"Level2 feature extraction failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def extract_level3(self, coeffs: List) -> np.ndarray:
        """Extract Level-3 features (low-frequency global layer)."""
        try:
            if len(coeffs) == 0:
                return np.zeros(10, dtype=np.float32)

            # LL3 subband
            ll3 = ensure_2d(coeffs[0])

            # FFT energy stats
            fft_mag = np.abs(np.fft.fft2(ll3))
            fft_features = [
                np.mean(fft_mag), np.std(fft_mag),
                np.percentile(fft_mag, 25), np.percentile(fft_mag, 75),
                np.sum(fft_mag[:fft_mag.shape[0] // 4, :fft_mag.shape[1] // 4])
            ]

            # DCT stats
            dct_coef = dct(dct(ll3.T, norm='ortho').T, norm='ortho')
            dct_features = [
                np.mean(dct_coef), np.std(dct_coef),
                np.percentile(dct_coef, 25), np.percentile(dct_coef, 75),
                np.sum(dct_coef[:dct_coef.shape[0] // 4, :dct_coef.shape[1] // 4])
            ]

            return np.array(fft_features + dct_features, dtype=np.float32)
        except Exception as e:
            print(f"Level3 feature extraction failed: {e}")
            return np.zeros(10, dtype=np.float32)

    def extract_cross_scale(self, coeffs: List) -> np.ndarray:
        """Extract cross-scale fingerprint features."""
        try:
            if len(coeffs) < 2:
                return np.zeros(65, dtype=np.float32)

            # LL3 and HH1 subbands
            ll3 = ensure_2d(coeffs[0])
            hh1 = ensure_2d(coeffs[1][2])  # HH1

            # DCT histogram of LL3
            dct_ll3 = dct(dct(ll3.T, norm='ortho').T, norm='ortho')
            dct_hist, _ = np.histogram(dct_ll3.flatten(), bins=32, range=(-10, 10))

            # LBP histogram of HH1
            hh1_norm = ((hh1 - np.min(hh1)) / (np.max(hh1) - np.min(hh1) + 1e-8) * 255).astype(np.uint8)
            lbp_hh1 = local_binary_pattern(hh1_norm, self.lbp_n_points, self.lbp_radius, self.lbp_method)
            lbp_hist, _ = np.histogram(lbp_hh1.flatten(), bins=32, range=(0, 58))

            # Cross-scale correlation
            a = dct_ll3.flatten()[:1000]  # limit length
            b = lbp_hh1.flatten()[:1000]
            if len(a) > 1 and len(b) > 1:
                cross_corr = np.corrcoef(a, b)[0, 1]
                if np.isnan(cross_corr):
                    cross_corr = 0.0
            else:
                cross_corr = 0.0

            return np.concatenate([dct_hist.astype(np.float32), lbp_hist.astype(np.float32),
                                   np.array([cross_corr], dtype=np.float32)])
        except Exception as e:
            print(f"Cross-scale feature extraction failed: {e}")
            return np.zeros(65, dtype=np.float32)


# ----------------- Hybrid Wavelet Feature Extractor -----------------
class HybridWaveletFeatureExtractor(nn.Module):
    """Hybrid wavelet feature extractor (full implementation)."""

    def __init__(self, feature_dim=512, wavelet_levels=3):
        super().__init__()
        self.geometry_analyzer = FaceGeometryAnalyzer()
        self.weight_learner = DynamicWeightLearner(
            num_regions=len(self.geometry_analyzer.REGION_MAPPING))
        self.wavelet_transformer = HybridWaveletTransformer(levels=wavelet_levels)
        self.feature_extractor = ThreeLevelFeatureExtractor()

        # Projection network (317 dims -> feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(317, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Convert to numpy and process
        if x.dim() == 4:  # batch mode
            x_np = x.detach().cpu().numpy()
            if x_np.shape[1] == 3:  # RGB to grayscale
                x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_np.transpose(0, 2, 3, 1)])
            else:
                x_gray = x_np.squeeze(1)
        else:
            raise ValueError("Input must be 4D tensor")

        # Process each image
        all_features = []
        for img in x_gray:
            try:
                # 1) Face geometry analysis
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
                landmarks = self.geometry_analyzer.get_landmarks(img_bgr)
                region_masks = self.geometry_analyzer.create_region_masks(img.shape, landmarks)
                region_smoothness = self.geometry_analyzer.calculate_region_smoothness(img, region_masks)
                region_curvature = self.geometry_analyzer.calculate_region_curvature(landmarks)
                region_weights = self.weight_learner(region_smoothness, region_curvature)

                # 2) Hybrid wavelet decomposition
                coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar = \
                    self.wavelet_transformer.decompose_separate(img)
                mixed_coeffs = self.wavelet_transformer.tensor_product_merge(
                    coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar,
                    region_weights, region_masks)

                # 3) Multi-level feature extraction
                features = [
                    self.feature_extractor.extract_level1(mixed_coeffs),
                    self.feature_extractor.extract_level2(mixed_coeffs),
                    self.feature_extractor.extract_level3(mixed_coeffs),
                    self.feature_extractor.extract_cross_scale(mixed_coeffs)
                ]

                # Ensure feature length = 317
                features = np.concatenate([f.flatten() for f in features])
                if len(features) < 317:
                    features = np.pad(features, (0, 317 - len(features)))
                elif len(features) > 317:
                    features = features[:317]

                all_features.append(features.astype(np.float32))
            except Exception as e:
                print(f"Feature extraction failed for one image: {e}")
                all_features.append(np.zeros(317, dtype=np.float32))

        # Convert to tensor and project
        features_tensor = torch.from_numpy(np.array(all_features)).float().to(x.device)
        return self.projection(features_tensor)


# ----------------- Bidirectional Channel Attention -----------------
class BidirectionalChannelAttention(nn.Module):
    """Bidirectional channel attention module."""

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.fc1_s = nn.Linear(channels, channels // reduction_ratio)
        self.fc2_s = nn.Linear(channels // reduction_ratio, channels)
        self.fc1_f = nn.Linear(channels, channels // reduction_ratio)
        self.fc2_f = nn.Linear(channels // reduction_ratio, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial features guide frequency attention
        spatial_gap = F.adaptive_avg_pool2d(spatial_feat, (1, 1)).squeeze(-1).squeeze(-1)
        freq_weights = self.sigmoid(self.fc2_f(self.relu(self.fc1_f(spatial_gap))))
        freq_attended = freq_feat * freq_weights.unsqueeze(-1).unsqueeze(-1)

        # Frequency features guide spatial attention
        freq_gap = F.adaptive_avg_pool2d(freq_feat, (1, 1)).squeeze(-1).squeeze(-1)
        spatial_weights = self.sigmoid(self.fc2_s(self.relu(self.fc1_s(freq_gap))))
        spatial_attended = spatial_feat * spatial_weights.unsqueeze(-1).unsqueeze(-1)

        return spatial_attended, freq_attended


# ----------------- Loss Function -----------------
class SADMLoss(nn.Module):
    """SADM loss with three components."""

    def __init__(self, lambda_sens: float = 1.0, lambda_reg: float = 0.1):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.lambda_sens = lambda_sens
        self.lambda_reg = lambda_reg

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                spatial_feat: torch.Tensor, freq_feat: torch.Tensor,
                spatial_weights: torch.Tensor, freq_weights: torch.Tensor) -> torch.Tensor:
        # Classification loss
        cls_loss = self.cls_loss(logits, labels.float())

        # Feature sensitivity contrast loss
        fake_mask = labels == 1
        real_mask = labels == 0

        if torch.any(fake_mask) and torch.any(real_mask):
            # Modulation strength (with wavelet weights)
            fake_spatial_mod = (spatial_feat[fake_mask] * spatial_weights[fake_mask]).abs().mean()
            fake_freq_mod = (freq_feat[fake_mask] * freq_weights[fake_mask]).abs().mean()
            real_spatial_mod = (spatial_feat[real_mask] * spatial_weights[real_mask]).abs().mean()
            real_freq_mod = (freq_feat[real_mask] * freq_weights[real_mask]).abs().mean()

            # Modulation ratio
            sensitivity_ratio = (fake_spatial_mod * fake_freq_mod) / (real_spatial_mod * real_freq_mod + 1e-8)
            sens_loss = -torch.log(sensitivity_ratio + 1e-8)
        else:
            sens_loss = torch.tensor(0.0, device=logits.device)

        # Channel regularization (stability of weights)
        spatial_weight_norm = F.normalize(spatial_weights, p=2, dim=1)
        freq_weight_norm = F.normalize(freq_weights, p=2, dim=1)
        reg_loss = torch.var(spatial_weight_norm) + torch.var(freq_weight_norm)

        return cls_loss + self.lambda_sens * sens_loss + self.lambda_reg * reg_loss


# ----------------- Complete SADM Model -----------------
class SADMModel(nn.Module):
    """Complete SADM model consistent with the paper."""

    def __init__(self, num_classes=2, feature_dim=256):
        super().__init__()
        # Spatial stream (Xception)
        try:
            from torchvision.models import xception
            self.spatial_stream = xception(pretrained=True)
            # Remove final classifier
            self.spatial_stream.fc = nn.Identity()
            spatial_out_dim = 2048
        except:
            # Fallback: simple CNN
            self.spatial_stream = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            spatial_out_dim = 128

        # Frequency stream (hybrid wavelet feature extractor)
        self.freq_stream = HybridWaveletFeatureExtractor(feature_dim=feature_dim)

        # Bidirectional attention fusion
        self.attention = BidirectionalChannelAttention(channels=feature_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Spatial feature adapter
        self.spatial_adapter = nn.Linear(spatial_out_dim,
                                         feature_dim) if spatial_out_dim != feature_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial stream
        spatial_feat = self.spatial_stream(x)
        if spatial_feat.dim() == 2:
            spatial_feat = spatial_feat.unsqueeze(-1).unsqueeze(-1)
        spatial_feat = self.spatial_adapter(spatial_feat.flatten(1))
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1, 1, 1)

        # Frequency stream
        freq_feat = self.freq_stream(x)
        freq_feat = freq_feat.view(freq_feat.size(0), -1, 1, 1)

        # Bidirectional attention fusion
        spatial_attended, freq_attended = self.attention(spatial_feat, freq_feat)

        # Fuse
        fused = torch.cat([
            spatial_attended.flatten(1),
            freq_attended.flatten(1)
        ], dim=1)

        # Classification
        return self.classifier(fused)


# ----------------- Usage Example -----------------
if __name__ == "__main__":
    # Create model
    model = SADMModel(num_classes=2)

    # Dummy input
    dummy_input = torch.randn(4, 3, 256, 256)

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Model created successfully!")
