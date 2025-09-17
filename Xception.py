import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import pandas as pd

# Basic configuration
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 1. Hyperparameter configuration (designed for class balance) --------------------------
class Params:
    # Data-related (keep original paths)
    IMAGE_SIZE = 299
    CROP_SIZE = 299
    DATA_ROOT = r"C:\Users\18223\PycharmProjects\PythonProject3\FF++_split"
    # Sampling parameters (to reduce memory usage)
    TRAIN_SAMPLE_RATIO = 0.3  # use 30% of training data
    VAL_SAMPLE_RATIO = 0.5  # use 50% of validation data
    # Training configuration (batch sizing)
    TRAIN_BATCH_SIZE = 8  # fits ~6GB GPU memory
    EVAL_BATCH_SIZE = 16
    INIT_LR = 1e-4
    EPOCHS = 30
    PATIENCE = 8
    # Core parameters for dataset balancing (ensure at least 5x oversampling)
    REAL_AUGMENT_TIMES = 5  # oversample real samples 5x
    START_FROM_BEST = True  # continue training from best checkpoint
    # Memory optimization
    NUM_WORKERS = min(os.cpu_count(), 4)
    PIN_MEMORY = False
    # Loss-related hyperparameters (for weighted loss)
    WEIGHT_DECAY = 1e-4
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTH = 0.1
    DROPOUT_RATE = 0.4
    # Save paths (keep original structure)
    WEIGHT_SAVE_DIR = "./Full_Xception_Weights"
    PRETRAIN_WEIGHT = r"C:\Users\18223\.cache\torch\hub\checkpoints\xception-43020ad28.pth"
    LOG_PLOT_PATH = os.path.join(WEIGHT_SAVE_DIR, "metrics.png")
    EXCEL_OUTPUT_PATH = os.path.join(WEIGHT_SAVE_DIR, "prediction_probabilities.xlsx")
    BEST_WEIGHT_PATH = os.path.join(WEIGHT_SAVE_DIR, "best_model.pth")


# -------------------------- 2. Lightweight attention module (unchanged) --------------------------
class LightChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = self.avg_pool(x).view(b, c)
        attn = self.fc(attn).view(b, c, 1, 1)
        return x * attn


# -------------------------- 3. Full Xception network architecture (100% preserved structure) --------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, grow_first=True, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.relu = nn.ReLU(inplace=True)
        self.residual = None

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(reps):
            if grow_first and i == 0:
                self.layers.append(SeparableConv2d(
                    current_channels, out_channels,
                    stride=stride if i == reps - 1 else 1
                ))
                current_channels = out_channels
            else:
                self.layers.append(SeparableConv2d(
                    current_channels, current_channels,
                    stride=stride if i == reps - 1 else 1
                ))

        if not grow_first:
            self.layers.append(SeparableConv2d(current_channels, out_channels, stride=1))

        self.attn = LightChannelAttention(out_channels) if use_attn else None
        self.dropout = nn.Dropout2d(p=Params.DROPOUT_RATE / 2)

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        x = x + residual
        return self.attn(x) if self.attn else x


class FullXception(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, device=torch.device('cpu')):
        super().__init__()
        self.device = device

        # Entry Flow (kept intact)
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            XceptionBlock(64, 128, reps=2, stride=2, grow_first=True),
            XceptionBlock(128, 256, reps=2, stride=2, grow_first=True),
            XceptionBlock(256, 728, reps=2, stride=2, grow_first=True)
        )

        # Middle Flow (8 repeated blocks, kept intact)
        self.middle_flow = nn.Sequential(
            *[XceptionBlock(728, 728, reps=3, stride=1, grow_first=True) for _ in range(8)]
        )

        # Exit Flow (kept intact)
        self.exit_flow = nn.Sequential(
            XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False),

            SeparableConv2d(1024, 1536, stride=1),
            SeparableConv2d(1536, 2048, stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification head (kept intact)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.7),
            nn.Linear(2048, num_classes)
        )

        self._init_weights()
        if pretrained:
            self._load_pretrained()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self):
        try:
            pretrained_state = torch.load(Params.PRETRAIN_WEIGHT, map_location=self.device)
            model_state = self.state_dict()

            filtered_state = {k: v for k, v in pretrained_state.items()
                              if k in model_state and v.size() == model_state[k].size()}

            model_state.update(filtered_state)
            self.load_state_dict(model_state)
            print(f"✅ Pretrained weights loaded successfully (matched {len(filtered_state)}/{len(pretrained_state)} layers)")
        except Exception as e:
            print(f"⚠️ Partial failure loading pretrained weights: {str(e)}, continuing training")

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.classifier(x)
        return x


# -------------------------- 4. Dataset loading (designed for class balance) --------------------------
class OriginalFFPPDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        # Stronger real-sample augmentation (meets the requirement)
        self.real_augment = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(Params.CROP_SIZE, scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.fake_augment = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(Params.CROP_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # Load and sample data (added sampling, keep balancing strategy)
        self.frame_paths, self.labels = self._load_and_sample_data()

        # Compute sample counts (keep oversampling logic)
        self.real_len = sum(1 for l in self.labels if l == 0)
        self.fake_len = sum(1 for l in self.labels if l == 1)
        self.total = self.real_len * Params.REAL_AUGMENT_TIMES + self.fake_len

        # Print balance ratio (for visualization)
        balanced_real = self.real_len * Params.REAL_AUGMENT_TIMES
        balanced_fake = self.fake_len
        ratio = balanced_real / balanced_fake if balanced_fake > 0 else 0
        print(
            f"📊 {split} set (after oversampling): Real={balanced_real} | Fake={balanced_fake} | "
            f"Real:Fake≈{ratio:.2f}:1 | Sampling ratio={self._get_sample_ratio():.0%}"
        )

    def _get_sample_ratio(self):
        if self.split == "train":
            return Params.TRAIN_SAMPLE_RATIO
        elif self.split == "val":
            return Params.VAL_SAMPLE_RATIO
        return 1.0  # no sampling for test set

    def _load_and_sample_data(self):
        real_frames = []
        fake_frames = []

        base_path = os.path.join(Params.DATA_ROOT, self.split)

        # Load real samples
        real_path = os.path.join(base_path, "real")
        if os.path.exists(real_path):
            for root, _, files in os.walk(real_path):
                image_files = [os.path.join(root, f) for f in files
                               if f.lower().endswith((".jpg", ".png"))]
                real_frames.extend(image_files)

        # Load fake samples
        fake_path = os.path.join(base_path, "fake")
        if os.path.exists(fake_path):
            for root, _, files in os.walk(fake_path):
                image_files = [os.path.join(root, f) for f in files
                               if f.lower().endswith((".jpg", ".png"))]
                fake_frames.extend(image_files)

        if not real_frames and not fake_frames:
            raise FileNotFoundError(f"No image files found under {base_path}")

        # Sampling (only for train/val, preserve class proportions)
        sample_ratio = self._get_sample_ratio()
        if sample_ratio < 1.0:
            sampled_real = self._sample_list(real_frames, sample_ratio)
            sampled_fake = self._sample_list(fake_frames, sample_ratio)
            print(
                f"🔍 {self.split} sampling: Real={len(real_frames)}→{len(sampled_real)}, Fake={len(fake_frames)}→{len(sampled_fake)}")
            real_frames, fake_frames = sampled_real, sampled_fake

        # Combine and shuffle
        combined = list(zip(real_frames + fake_frames,
                            [0] * len(real_frames) + [1] * len(fake_frames)))
        random.shuffle(combined)

        return zip(*combined) if combined else ([], [])

    def _sample_list(self, data_list, ratio):
        if len(data_list) == 0:
            return data_list
        sample_size = max(100, int(len(data_list) * ratio))  # keep at least 100 samples
        return random.sample(data_list, min(sample_size, len(data_list)))

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        try:
            # Oversampling logic (meet 5x augmentation requirement)
            if idx < self.real_len * Params.REAL_AUGMENT_TIMES and self.real_len > 0:
                real_idx = idx % self.real_len
                img_path = self.frame_paths[real_idx]
                label = 0
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((Params.IMAGE_SIZE, Params.IMAGE_SIZE),
                                                    Image.LANCZOS)
                img = self.real_augment(img)  # augmentation for real samples
            else:
                fake_idx = idx - self.real_len * Params.REAL_AUGMENT_TIMES
                img_path = self.frame_paths[self.real_len + fake_idx]
                label = 1
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((Params.IMAGE_SIZE, Params.IMAGE_SIZE),
                                                    Image.LANCZOS)
                img = self.fake_augment(img)

            if self.transform:
                img = self.transform(img)

            # Channel validation
            if img.shape[0] != 3:
                img = img[:3, ...] if img.shape[0] > 3 else img.repeat((3 // img.shape[0] + 1), 1, 1)[:3, ...]

            return img, torch.tensor(label, dtype=torch.long), img_path
        except Exception as e:
            # Handle corrupted images
            img = Image.new("RGB", (Params.IMAGE_SIZE, Params.IMAGE_SIZE), (127, 127, 127))
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(1 if idx % 2 == 0 else 0, dtype=torch.long), f"corrupted_image_{idx}.jpg"


# -------------------------- 5. Data loaders --------------------------
def get_transforms(split):
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(Params.CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_dataloader(split, batch_size):
    dataset = OriginalFFPPDataset(
        split=split,
        transform=get_transforms(split)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=Params.NUM_WORKERS,
        pin_memory=Params.PIN_MEMORY,
        drop_last=(split == "train"),
        persistent_workers=True
    )


# -------------------------- 6. Loss function (weighted as required) --------------------------
class WeightedFocalLoss(nn.Module):
    def __init__(self, real_weight=5.0, fake_weight=1.0):
        super().__init__()
        self.real_weight = real_weight  # real sample weight is 5x fake sample
        self.fake_weight = fake_weight
        self.gamma = Params.FOCAL_GAMMA
        self.label_smooth = Params.LABEL_SMOOTH
        print(f"⚖️ Loss weights: real={real_weight}, fake={fake_weight}")  # transparency info

    def forward(self, logits, labels):
        num_classes = logits.size(-1)
        one_hot = torch.full_like(logits, self.label_smooth / (num_classes - 1))
        one_hot.scatter_(1, labels.unsqueeze(1), 1 - self.label_smooth)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -one_hot * log_probs
        ce_loss = ce_loss.sum(dim=-1)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights (real has higher weight)
        weights = torch.where(labels == 0, self.real_weight, self.fake_weight).to(logits.device)
        return (focal_loss * weights).mean()


def calculate_metrics(all_preds, all_labels, all_probs):
    acc = accuracy_score(all_labels, all_preds)
    has_real = (np.array(all_labels) == 0).any()
    has_fake = (np.array(all_labels) == 1).any()

    prec_real = precision_score(all_labels, all_preds, pos_label=0, zero_division=1.0) if has_real else 0.0
    prec_fake = precision_score(all_labels, all_preds, pos_label=1, zero_division=1.0) if has_fake else 0.0
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.5

    return acc, prec_real, prec_fake, auc


# -------------------------- 7. Evaluation and testing functions (metrics display) --------------------------
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, save_probs=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    all_paths = []

    for imgs, labels, paths in tqdm(dataloader, desc="Evaluation", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1])
        all_paths.extend(paths)
        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    acc, prec_real, prec_fake, auc = calculate_metrics(all_preds, all_labels, all_probs)

    if save_probs:
        import threading
        threading.Thread(target=save_probabilities_to_excel,
                         args=(all_paths, all_labels,
                               list(zip(1 - np.array(all_probs), all_probs))),
                         daemon=True).start()

    return avg_loss, acc, prec_real, prec_fake, auc


def save_probabilities_to_excel(image_paths, true_labels, probabilities):
    data = []
    for path, label, prob in zip(image_paths, true_labels, probabilities):
        sample_type = "Real (negative)" if label == 0 else "Fake (positive)"
        data.append({
            "image_path": path,
            "sample_type": sample_type,
            "true_label": label,
            "prob_real": prob[0],
            "prob_fake": prob[1],
            "prediction": "Real" if prob[0] > prob[1] else "Fake"
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(Params.EXCEL_OUTPUT_PATH), exist_ok=True)
    df.to_excel(Params.EXCEL_OUTPUT_PATH, index=False)
    print(f"📊 Probability statistics saved to Excel: {Params.EXCEL_OUTPUT_PATH}")


def test_best_model():
    os.makedirs(Params.WEIGHT_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test device] {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    print("Loading test set...")
    test_loader = get_dataloader("test", Params.EVAL_BATCH_SIZE)

    model = FullXception(pretrained=False, device=device).to(device)
    if not os.path.exists(Params.BEST_WEIGHT_PATH):
        print(f"Error: best model weights not found at {Params.BEST_WEIGHT_PATH}")
        return

    model.load_state_dict(torch.load(Params.BEST_WEIGHT_PATH, map_location=device))
    print(f"✅ Loaded best model weights: {Params.BEST_WEIGHT_PATH}")

    criterion = WeightedFocalLoss()
    print("Starting testing...")
    test_loss, test_acc, test_prec_real, test_prec_fake, test_auc = evaluate(
        model, test_loader, criterion, device, save_probs=True
    )

    print("\n" + "=" * 60)
    print("Best model test results:")
    print(f"Loss: {test_loss:.4f} | ACC: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"Precision (real): {test_prec_real:.4f} | Precision (fake): {test_prec_fake:.4f}")
    print("=" * 60)


# -------------------------- 8. Main training function (meets all requirements) --------------------------
def main():
    os.makedirs(Params.WEIGHT_SAVE_DIR, exist_ok=True)

    # Fix random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # GPU memory optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training device] {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Check dataset path
    if not os.path.exists(Params.DATA_ROOT):
        print(f"Error: dataset path does not exist {Params.DATA_ROOT}")
        sys.exit(1)

    # Load data
    print("\nLoading datasets...")
    train_loader = get_dataloader("train", Params.TRAIN_BATCH_SIZE)
    val_loader = get_dataloader("val", Params.EVAL_BATCH_SIZE)

    # Model and training components (support resuming from best checkpoint)
    model = FullXception(pretrained=not Params.START_FROM_BEST, device=device).to(device)

    start_epoch = 0
    if Params.START_FROM_BEST and os.path.exists(Params.BEST_WEIGHT_PATH):
        model.load_state_dict(torch.load(Params.BEST_WEIGHT_PATH, map_location=device))
        print(f"✅ Loaded best model; continuing training from checkpoint: {Params.BEST_WEIGHT_PATH}")
        if os.path.exists(os.path.join(Params.WEIGHT_SAVE_DIR, "last_epoch.txt")):
            with open(os.path.join(Params.WEIGHT_SAVE_DIR, "last_epoch.txt"), "r") as f:
                start_epoch = int(f.read().strip())
            print(f"🔄 Resuming from epoch {start_epoch + 1}")

    # Optimizer and loss
    criterion = WeightedFocalLoss(real_weight=5.0, fake_weight=1.0)  # real samples weighted 5x
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Params.INIT_LR,
        weight_decay=Params.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    scaler = GradScaler()

    # Training records
    best_auc = 0.0
    best_real_precision = 0.0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "train_auc": [], "train_prec_real": [], "train_prec_fake": [],
               "val_loss": [], "val_acc": [], "val_auc": [], "val_prec_real": [], "val_prec_fake": []}

    print(f"\n[Start training] {Params.EPOCHS} epochs | LR={Params.INIT_LR} | EarlyStop patience={Params.PATIENCE}")

    for epoch in range(start_epoch, Params.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels, all_train_probs = [], [], []

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{Params.EPOCHS}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probs.extend(probs)
            train_loss += loss.item() * imgs.size(0)

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Compute training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc, train_prec_real, train_prec_fake, train_auc = calculate_metrics(
            all_train_preds, all_train_labels, all_train_probs
        )

        # Record training metrics
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["train_prec_real"].append(train_prec_real)
        history["train_prec_fake"].append(train_prec_fake)

        # Validation phase
        val_loss, val_acc, val_prec_real, val_prec_fake, val_auc = evaluate(
            model, val_loader, criterion, device, save_probs=False
        )

        # Record validation metrics
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_prec_real"].append(val_prec_real)
        history["val_prec_fake"].append(val_prec_fake)

        # LR scheduler step
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc or (val_auc == best_auc and val_prec_real > best_real_precision):
            best_auc = val_auc
            best_real_precision = val_prec_real
            torch.save(model.state_dict(), Params.BEST_WEIGHT_PATH)
            with open(os.path.join(Params.WEIGHT_SAVE_DIR, "last_epoch.txt"), "w") as f:
                f.write(str(epoch))
            no_improve = 0
            print(f"📌 Saved best model (val AUC: {best_auc:.4f}, real precision: {best_real_precision:.4f})")
        else:
            no_improve += 1
            if no_improve >= Params.PATIENCE:
                print(f"⏹️ Early stopping: validation AUC did not improve for {no_improve} epochs")
                break

        # Per-epoch summary (detailed reporting)
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"Train - Loss: {avg_train_loss:.4f} | ACC: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"        Precision (real): {train_prec_real:.4f} | Precision (fake): {train_prec_fake:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | ACC: {val_acc:.4f} | AUC: {val_auc:.4f}")
        print(f"        Precision (real): {val_prec_real:.4f} | Precision (fake): {val_prec_fake:.4f}\n")

    # Test the best model after training
    test_best_model()

    # Save metric plots
    plot_history(history)


def plot_history(history):
    plt.figure(figsize=(15, 10))

    # Loss curve
    plt.subplot(231)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ACC curve
    plt.subplot(232)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # AUC curve
    plt.subplot(233)
    plt.plot(history["train_auc"], label="train")
    plt.plot(history["val_auc"], label="val")
    plt.title("AUC over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    # Precision for real samples
    plt.subplot(234)
    plt.plot(history["train_prec_real"], label="train")
    plt.plot(history["val_prec_real"], label="val")
    plt.title("Precision (real) over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    # Precision for fake samples
    plt.subplot(235)
    plt.plot(history["train_prec_fake"], label="train")
    plt.plot(history["val_prec_fake"], label="val")
    plt.title("Precision (fake) over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(Params.LOG_PLOT_PATH, dpi=300)
    print(f"📊 Metric curves saved to {Params.LOG_PLOT_PATH}")


if __name__ == "__main__":
    # main()
    # Uncomment to run test only
    test_best_model()
