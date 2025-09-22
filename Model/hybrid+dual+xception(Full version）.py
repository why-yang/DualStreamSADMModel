import numpy as np
import cv2
import dlib
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import local_binary_pattern, hessian_matrix, hessian_matrix_eigvals
from scipy.fftpack import dct, fft2
from scipy.signal import convolve2d
from typing import Dict, Tuple, List, Optional
import os
import time
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ----------------- Device configuration -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Data augmentation & preprocessing -----------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ----------------- Dataset class -----------------
class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 递归遍历所有子文件夹
        self._load_images_recursive(root_dir)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def _load_images_recursive(self, current_dir):
        """Recursively load images from all subdirectories."""
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)

            if os.path.isdir(item_path):
                # 如果是文件夹，递归遍历
                self._load_images_recursive(item_path)
            elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 如果是图像文件，根据父文件夹名称确定标签
                parent_dir = os.path.basename(os.path.dirname(item_path))
                if parent_dir.lower() in ['real', '0']:
                    label = 0
                elif parent_dir.lower() in ['fake', '1']:
                    label = 1
                else:
                    # 如果父文件夹不是real/fake，根据路径判断
                    if 'real' in current_dir.lower() or '0' in current_dir.lower():
                        label = 0
                    elif 'fake' in current_dir.lower() or '1' in current_dir.lower():
                        label = 1
                    else:
                        # 如果无法确定，跳过
                        continue

                self.image_paths.append(item_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # 返回黑色图像作为fallback
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                # 返回默认transform
                image = transforms.ToTensor()(image)

        return image, label


# ----------------- Utility functions -----------------
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


def safe_phasecong(image: np.ndarray, **kwargs) -> np.ndarray:
    """Phase congruency (fallback/placeholder implementation)."""
    img = ensure_2d(image)
    return np.zeros_like(img)


def safe_hessian_eigvals(detail: np.ndarray):
    """Compatibility wrapper for skimage.feature Hessian eigenvalues across versions."""
    det = ensure_2d(detail)
    try:
        H_elems = hessian_matrix(det, sigma=1.0)
        try:
            eig = hessian_matrix_eigvals(H_elems)
        except Exception:
            eig = hessian_matrix_eigvals(*H_elems)

        if isinstance(eig, tuple) or isinstance(eig, list):
            k1, k2 = eig[0], eig[1]
        else:
            eig = np.asarray(eig)
            if eig.ndim >= 3 and eig.shape[0] == 2:
                k1, k2 = eig[0], eig[1]
            elif eig.ndim == 2:
                k1 = eig
                k2 = np.zeros_like(k1)
            else:
                k1 = np.zeros_like(det)
                k2 = np.zeros_like(det)
        return k1, k2
    except Exception:
        return np.zeros_like(det), np.zeros_like(det)


# ----------------- Face geometry analyzer -----------------
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

    def __init__(self, predictor_path: Optional[str] = "shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        try:
            if predictor_path and os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                self.predictor = None
                print("Warning: dlib predictor file not found")
        except Exception as e:
            self.predictor = None
            print(f"Warning: Could not load dlib predictor. {e}")

    def get_landmarks(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            h, w = img_bgr.shape[:2]
            return np.random.rand(68, 2) * np.array([w, h])

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        if not faces:
            h, w = img_bgr.shape[:2]
            return np.random.rand(68, 2) * np.array([w, h])

        shape = self.predictor(gray, faces[0])
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

    def create_region_masks(self, img_shape: Tuple[int, int], landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = img_shape[:2]
        masks = {}
        for name, indices in self.REGION_MAPPING.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = landmarks[indices].astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 1, 2)
            masks[name] = mask
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

    def calculate_region_smoothness(self, gray_img: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        region_smoothness = {}
        h, w = gray_img.shape

        try:
            f = np.abs(fft2(gray_img.astype(np.float32)))
        except Exception:
            gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1)
            f = np.hypot(gx, gy)

        mask_center = np.zeros((h, w), np.uint8)
        crow, ccol = h // 2, w // 2
        cv2.circle(mask_center, (ccol, crow), min(h, w) // 8, 1, -1)

        for name, mask in masks.items():
            if np.sum(mask) == 0:
                region_smoothness[name] = 0.0
                continue

            maskf = mask.astype(np.float32)
            low = np.sum(f * maskf * mask_center)
            high = np.sum(f * maskf * (1 - mask_center))
            region_smoothness[name] = float(low / (low + high + 1e-8))

        return region_smoothness


# ----------------- Dynamic weight learner -----------------
class DynamicWeightLearner(nn.Module):
    def __init__(self, num_regions: int = 9):
        super().__init__()
        self.wS = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))
        self.wC = nn.Parameter(torch.ones(num_regions, dtype=torch.float32))

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


# ----------------- Hybrid wavelet transformer -----------------
class HybridWaveletTransformer:
    def __init__(self, levels=3):
        self.levels = levels
        self.coif_wavelet = 'coif1'
        self.haar_wavelet = 'haar'

    def _safe_wavedec2(self, img: np.ndarray, wavelet: str):
        img2 = ensure_2d(img)
        try:
            return pywt.wavedec2(img2, wavelet, level=self.levels)
        except Exception:
            h, w = img2.shape
            coeffs = [np.zeros((h // (2 ** self.levels), w // (2 ** self.levels)), dtype=np.float32)]
            for i in range(self.levels):
                size = (h // (2 ** (self.levels - i)), w // (2 ** (self.levels - i)))
                coeffs.append((np.zeros(size), np.zeros(size), np.zeros(size)))
            return coeffs

    def decompose_separate(self, img: np.ndarray) -> Tuple[List, List, List, List]:
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got {img.ndim}D instead")

        return (
            self._safe_wavedec2(img, self.coif_wavelet),
            self._safe_wavedec2(img, self.haar_wavelet),
            self._safe_wavedec2(img, self.coif_wavelet),
            self._safe_wavedec2(img, self.haar_wavelet)
        )

    def tensor_product_merge(self, coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar,
                             region_weights, region_masks):
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

                base_shape = term1[0].shape
                weight_mask = np.zeros(base_shape, dtype=np.float32)

                for region, (alpha, beta) in region_weights.items():
                    mask = region_masks.get(region)
                    if mask is None:
                        continue
                    mask_scaled = cv2.resize(mask.astype(np.float32), (base_shape[1], base_shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    mask_scaled = ensure_2d(mask_scaled)

                    contribution_0 = mask_scaled * (float(alpha.item()) * term1[0] + float(beta.item()) * term2[0])
                    if contribution_0.shape != weight_mask.shape:
                        contribution_0 = cv2.resize(contribution_0, (weight_mask.shape[1], weight_mask.shape[0]),
                                                    interpolation=cv2.INTER_AREA)
                    weight_mask += contribution_0

                ch1 = float(alpha.item()) * term1[1] + float(beta.item()) * term2[1]
                ch2 = float(alpha.item()) * term1[2] + float(beta.item()) * term2[2]

                mixed_level = (weight_mask, ch1, ch2)
                mixed_coeffs.append(mixed_level)
            except Exception as e:
                h, w = mixed_coeffs[0].shape
                if level > 1:
                    h //= 2 ** (level - 1)
                    w //= 2 ** (level - 1)
                mixed_coeffs.append((np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))))

        return mixed_coeffs

    @staticmethod
    def _compute_tensor_product(x, y):
        try:
            a0 = ensure_2d(x[0])
            b0 = ensure_2d(y[0])
            t0 = np.tensordot(a0, b0, axes=0)
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
            out = np.zeros(target_shape, dtype=tensor.dtype)
            h = min(tensor.shape[0], target_shape[0])
            w = min(tensor.shape[1], target_shape[1])
            out[:h, :w] = tensor[:h, :w]
            return out


# ----------------- Three-level feature extractor -----------------
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
                return np.zeros(4 * 59, dtype=np.float32)

            level1 = coeffs[1]
            if isinstance(level1, tuple) and len(level1) == 3:
                hh1 = level1[2]
            elif isinstance(level1, list) and len(level1) >= 3:
                hh1 = level1[2]
            else:
                hh1 = level1[-1] if isinstance(level1, (list, tuple)) else level1

            hh1 = ensure_2d(hh1)
            weight_map = np.abs(hh1) / (np.max(np.abs(hh1)) + 1e-8)

            features = []
            for kernel in self.srm_filters:
                kernel2 = np.atleast_2d(kernel.astype(np.float32))
                filtered = convolve2d(hh1, kernel2, mode='same', boundary='symm')
                filtered = filtered * (1 + weight_map)

                if np.max(filtered) - np.min(filtered) < 1e-8:
                    filtered_int = np.zeros_like(hh1, dtype=np.uint8)
                else:
                    filtered_int = ((filtered - np.min(filtered)) / (
                                np.max(filtered) - np.min(filtered) + 1e-8) * 255).astype(np.uint8)

                lbp = local_binary_pattern(filtered_int, self.lbp_n_points, self.lbp_radius, method=self.lbp_method)
                hist, _ = np.histogram(lbp, bins=59, range=(0, 58))
                features.append(hist.astype(np.float32))

            return np.concatenate(features)
        except Exception as e:
            return np.zeros(4 * 59, dtype=np.float32)

    def extract_level2(self, coeffs: List) -> np.ndarray:
        try:
            if len(coeffs) >= 3:
                level_data = coeffs[2]
            else:
                level_data = coeffs[1]

            if isinstance(level_data, (list, tuple)):
                if len(level_data) == 3:
                    lh2, hl2, hh2 = level_data[:3]
                elif len(level_data) > 2 and isinstance(level_data[2], (list, tuple)):
                    lh2, hl2, hh2 = level_data[2][:3]
                else:
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
            pc = safe_phasecong(detail, nscale=4, norient=6)
            weighted_curvature = ((kappa1 + kappa2) / 2.0) * pc

            flat = weighted_curvature.flatten()
            flat = flat[~np.isnan(flat)]
            if flat.size == 0:
                return np.zeros(6, dtype=np.float32)

            return np.array([
                float(np.mean(flat)), float(np.std(flat)),
                float(np.percentile(flat, 25)), float(np.percentile(flat, 75)),
                float(np.max(flat)), float(np.min(flat))
            ], dtype=np.float32)
        except Exception as e:
            return np.zeros(6, dtype=np.float32)

    def extract_level3(self, coeffs: List) -> np.ndarray:
        try:
            ll3 = coeffs[0]
            ll3 = ensure_2d(ll3)

            try:
                fft_mag = np.abs(fft2(ll3))
            except Exception:
                fft_mag = np.abs(np.fft.fft2(ll3))

            fft_features = [
                float(np.mean(fft_mag)), float(np.std(fft_mag)),
                float(np.percentile(fft_mag, 25)), float(np.percentile(fft_mag, 75)),
                float(np.sum(fft_mag[:fft_mag.shape[0] // 4, :fft_mag.shape[1] // 4]))
            ]

            try:
                dct_coef = dct(dct(ll3.T, norm='ortho').T, norm='ortho')
            except Exception:
                dct_coef = np.zeros_like(ll3)

            dct_features = [
                float(np.mean(dct_coef)), float(np.std(dct_coef)),
                float(np.percentile(dct_coef, 25)), float(np.percentile(dct_coef, 75)),
                float(np.sum(dct_coef[:dct_coef.shape[0] // 4, :dct_coef.shape[1] // 4]))
            ]

            return np.array(fft_features + dct_features, dtype=np.float32)
        except Exception as e:
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
            lbp_hh1 = local_binary_pattern(hh1_int, self.lbp_n_points, self.lbp_radius, method=self.lbp_method)
            lbp_hist, _ = np.histogram(lbp_hh1.flatten(), bins=32, range=(0, 58))

            try:
                a = dct_ll3.flatten()
                b = lbp_hh1.flatten()
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

            return np.concatenate([dct_hist.astype(np.float32), lbp_hist.astype(np.float32),
                                   np.array([cross_corr], dtype=np.float32)])
        except Exception as e:
            return np.zeros(65, dtype=np.float32)


# ----------------- Hybrid wavelet feature extractor -----------------
class HybridWaveletFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, wavelet_levels=3):
        super().__init__()
        self.geometry_analyzer = FaceGeometryAnalyzer()
        self.weight_learner = DynamicWeightLearner(num_regions=len(self.geometry_analyzer.REGION_MAPPING))
        self.wavelet_transformer = HybridWaveletTransformer(levels=wavelet_levels)
        self.feature_extractor = ThreeLevelFeatureExtractor()

        # Feature dimension calculation
        self.feature_dim = 4 * 59 + 6 + 10 + 65  # 317 dims

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        try:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            landmarks = self.geometry_analyzer.get_landmarks(img_bgr)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = (gray - gray.mean()) / (gray.std() + 1e-8)

            region_masks = self.geometry_analyzer.create_region_masks(gray.shape, landmarks)
            region_smoothness = self.geometry_analyzer.calculate_region_smoothness(gray, region_masks)
            region_curvature = self.geometry_analyzer.calculate_region_curvature(landmarks)
            region_weights = self.weight_learner(region_smoothness, region_curvature)

            coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar = \
                self.wavelet_transformer.decompose_separate(gray)
            mixed_coeffs = self.wavelet_transformer.tensor_product_merge(
                coeffs_x_coif, coeffs_x_haar, coeffs_y_coif, coeffs_y_haar,
                region_weights, region_masks)

            features = [
                self.feature_extractor.extract_level1(mixed_coeffs),
                self.feature_extractor.extract_level2(mixed_coeffs),
                self.feature_extractor.extract_level3(mixed_coeffs),
                self.feature_extractor.extract_cross_scale(mixed_coeffs)
            ]

            return np.concatenate([f.flatten() for f in features]).astype(np.float32)

        except Exception as e:
            return np.zeros(self.feature_dim, dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()

        if x_np.shape[1] == 3:
            x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_np.transpose(0, 2, 3, 1)])
        else:
            x_gray = x_np.squeeze(1)

        all_features = []
        for img in x_gray:
            features = self.extract_features(img)
            all_features.append(features)

        features_tensor = torch.from_numpy(np.array(all_features)).float().to(x.device)
        return self.projection(features_tensor)


# ----------------- Bidirectional channel attention -----------------
class BidirectionalChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.fc1_s = nn.Linear(channels, channels // reduction_ratio)
        self.fc2_s = nn.Linear(channels // reduction_ratio, channels)
        self.fc1_f = nn.Linear(channels, channels // reduction_ratio)
        self.fc2_f = nn.Linear(channels // reduction_ratio, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_gap = F.adaptive_avg_pool2d(spatial_feat, (1, 1)).squeeze(-1).squeeze(-1)
        freq_weights = self.sigmoid(self.fc2_f(self.relu(self.fc1_f(spatial_gap))))
        freq_attended = freq_feat * freq_weights.unsqueeze(-1).unsqueeze(-1)

        freq_gap = F.adaptive_avg_pool2d(freq_feat, (1, 1)).squeeze(-1).squeeze(-1)
        spatial_weights = self.sigmoid(self.fc2_s(self.relu(self.fc1_s(freq_gap))))
        spatial_attended = spatial_feat * spatial_weights.unsqueeze(-1).unsqueeze(-1)

        return spatial_attended, freq_attended


# ----------------- Full SADM model -----------------
class SADMModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=256):
        super().__init__()
        # 空间流 (Xception)
        try:
            from torchvision.models import xception
            self.spatial_stream = xception(pretrained=True)
            self.spatial_stream.fc = nn.Identity()
            spatial_out_dim = 2048
        except:
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

        # 频率流
        self.freq_stream = HybridWaveletFeatureExtractor(feature_dim=feature_dim)

        # 双向注意力融合
        self.attention = BidirectionalChannelAttention(channels=feature_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.spatial_adapter = nn.Linear(spatial_out_dim,
                                         feature_dim) if spatial_out_dim != feature_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间流
        spatial_feat = self.spatial_stream(x)
        if spatial_feat.dim() == 2:
            spatial_feat = spatial_feat.unsqueeze(-1).unsqueeze(-1)
        spatial_feat = self.spatial_adapter(spatial_feat.flatten(1))
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1, 1, 1)

        # 频率流
        freq_feat = self.freq_stream(x)
        freq_feat = freq_feat.view(freq_feat.size(0), -1, 1, 1)

        # 双向注意力融合
        spatial_attended, freq_attended = self.attention(spatial_feat, freq_feat)

        # 融合特征
        fused = torch.cat([
            spatial_attended.flatten(1),
            freq_attended.flatten(1)
        ], dim=1)

        return self.classifier(fused)


# ----------------- Training loop -----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': accuracy
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Best model saved with accuracy: {accuracy:.4f}')

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': accuracy
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    return train_losses, val_losses, val_accuracies


# ----------------- Test / evaluation -----------------
def test_model(model, test_loader, model_path=None):
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['real', 'fake'])

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return all_preds, all_labels, all_probs


# ----------------- Main entry -----------------
def main():
    # 数据路径
    train_dir = r"H:\xception\train"
    test_dir = r"H:\xception\test"
    val_dir = r"H:\xception\val"

    # 检查路径是否存在
    for path in [train_dir, test_dir, val_dir]:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist!")

    # 创建数据集
    train_dataset = DeepFakeDataset(train_dir, transform=train_transform)
    val_dataset = DeepFakeDataset(val_dir, transform=val_transform)
    test_dataset = DeepFakeDataset(test_dir, transform=val_transform)

    if len(train_dataset) == 0:
        print("No training images found! Please check your data directory structure.")
        print("Expected structure:")
        print("H:\\xception\\train\\")
        print("  ├── real\\")
        print("  │   ├── image1.jpg")
        print("  │   ├── image2.jpg")
        print("  │   └── ...")
        print("  └── fake\\")
        print("      ├── image1.jpg")
        print("      ├── image2.jpg")
        print("      └── ...")
        return

    # 创建数据加载器
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 显示一些样本信息
    print("\nSample image paths:")
    for i in range(min(5, len(train_dataset))):
        print(f"  {train_dataset.image_paths[i]} -> label: {train_dataset.labels[i]}")

    # 创建模型
    model = SADMModel(num_classes=2).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练模型
    print("Starting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=1, save_dir='./checkpoints'
    )

    # 测试模型
    print("Testing model...")
    test_preds, test_labels, test_probs = test_model(model, test_loader, './checkpoints/best_model.pth')

if __name__ == "__main__":
    main()

