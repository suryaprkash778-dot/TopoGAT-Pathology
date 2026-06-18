import os
import sys

# =====================================================================
# 0. AUTOMATED GIGAPIXEL ENVIRONMENT INITIALIZER
# =====================================================================
try:
    import openslide
    import torch_geometric

    print("[SYSTEM] Environment verification successful. All engines online.")
except ModuleNotFoundError:
    print("\n[SYSTEM] Detected fresh runtime. Rebuilding gigapixel environment automatically...")
    print("[SYSTEM] Installing OpenSlide C-Libraries (this takes ~15 seconds)...")
    os.system("apt-get update -qq && apt-get install -y openslide-tools > /dev/null 2>&1")
    print("[SYSTEM] Installing Python wrappers and Graph components...")
    os.system("pip install openslide-python torch-geometric -q")
    print("[SYSTEM] Environment built successfully! Proceeding to Mark 2 Execution...\n")

    # Force the active Python kernel to recognize the brand new installations
    import site
    from importlib import reload

    reload(site)



# ... (The rest of your Mark 2 script continues exactly as it was below this)
import os
import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from google.colab import drive

# =====================================================================
# 1. PERSISTENT STORAGE & CONFIGURATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INITIATING MARK 2 (100-SLIDE RUN) ON {device.type.upper()} ---")

# Mount Google Drive to protect against Colab disconnects
drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/Mark2_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mid_flight_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_mark2.pth")

EPOCHS = 10
TRAIN_CHUNKS = [1, 2, 3, 4, 5, 6, 7, 8]  # 80 Slides for Training
VAL_CHUNKS = [9]  # 10 Slides for Validation (Tuning)
TEST_CHUNKS = [10]  # 10 Slides for Final Test (The Exam)


# =====================================================================
# 2. THE MARK 2 ARCHITECTURE
# =====================================================================
def get_tissue_coordinates(slide_path, downsample_level=4):
    try:
        slide = openslide.OpenSlide(slide_path)
        thumb = slide.read_region((0, 0), downsample_level, slide.level_dimensions[downsample_level])
        thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGBA2GRAY)
        _, mask = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        y_coords, x_coords = np.nonzero(mask)

        scale_factor = int(slide.level_downsamples[downsample_level])
        coords = [[x_coords[i] * scale_factor, y_coords[i] * scale_factor] for i in range(len(x_coords))]

        if len(coords) > 800:
            coords = random.sample(coords, 800)

        return coords
    except Exception as e:
        return []


class ClinicalWSIDataset(Dataset):
    def __init__(self, slide_path, coords_list, patch_size=256):
        self.slide = openslide.OpenSlide(slide_path)
        self.coords_list = coords_list
        self.patch_size = patch_size

    def __len__(self): return len(self.coords_list)

    def __getitem__(self, idx):
        x, y = self.coords_list[idx]
        patch = self.slide.read_region((int(x), int(y)), 0, (self.patch_size, self.patch_size)).convert("L")
        return TF.to_tensor(patch), torch.tensor([x, y], dtype=torch.float32)


class MultiScaleWaveletExtractor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        haar_weights = torch.tensor([[[1., 1.], [1., 1.]], [[-1., -1.], [1., 1.]],
                                     [[-1., 1.], [-1., 1.]], [[1., -1.], [-1., 1.]]]).unsqueeze(1) / 4.0
        self.wavelet = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.wavelet.weight = nn.Parameter(haar_weights, requires_grad=True)

        self.pool_small = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_med = nn.AdaptiveAvgPool2d((8, 8))
        self.pool_large = nn.AdaptiveAvgPool2d((16, 16))

        self.compressor = nn.Sequential(
            nn.Linear(1344, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1344)
        )

    def extract_raw(self, x):
        w = self.wavelet(x)
        p_s = self.pool_small(w).view(x.size(0), -1)
        p_m = self.pool_med(w).view(x.size(0), -1)
        p_l = self.pool_large(w).view(x.size(0), -1)
        return torch.cat([p_s, p_m, p_l], dim=1)

    def forward(self, x):
        return self.compressor(self.extract_raw(x))

    def self_supervised_forward(self, x, mask_prob=0.30):
        raw = self.extract_raw(x)
        mask = (torch.rand(raw.shape).to(x.device) > mask_prob).float()
        compressed = self.compressor(raw * mask)
        return self.decoder(compressed), raw


class AdvancedAttentionGNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(2, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, edge_index, coords):
        norm_coords = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-5)
        nodes = nodes + self.pos_encoder(norm_coords)

        x1 = F.leaky_relu(self.conv1(nodes, edge_index), 0.01)
        x2 = F.leaky_relu(self.conv2(x1, edge_index), 0.01)
        x = nodes + x1 + x2

        weights = F.softmax(self.attention(x), dim=0)
        prediction = torch.sigmoid(self.classifier(torch.sum(x * weights, dim=0, keepdim=True)))
        return prediction, weights


def process_slide(slide_path, label, extractor, gnn, criterion, is_training=True):
    coords = get_tissue_coordinates(slide_path)
    if not coords: return 0, 0, 0, None, None

    loader = DataLoader(ClinicalWSIDataset(slide_path, coords), batch_size=64, shuffle=False)
    all_nodes, all_coords = [], []

    context = torch.enable_grad() if is_training else torch.no_grad()
    if is_training:
        extractor.train(); gnn.train()
    else:
        extractor.eval(); gnn.eval()

    with context:
        for patches, batch_coords in loader:
            all_nodes.append(extractor(patches.to(device)))
            all_coords.append(batch_coords.to(device))

    master_nodes = torch.cat(all_nodes)
    master_coords = torch.cat(all_coords)
    dist = torch.cdist(master_coords, master_coords)
    edge_index = ((dist <= 400.0) & (dist > 0)).nonzero(as_tuple=False).t().contiguous()

    prediction, weights = gnn(master_nodes, edge_index, master_coords)
    loss = criterion(prediction, torch.tensor([[label]]).to(device))

    if is_training: loss.backward()

    acc = 100 if (prediction.item() >= 0.5) == label else 0
    return loss.item(), acc, prediction.item(), weights, master_coords


# =====================================================================
# 3. CLOUD MANAGER
# =====================================================================
def manage_cloud_chunk(chunk_id, download=True):
    start_idx = ((chunk_id - 1) * 10) + 1
    end_idx = start_idx + 9

    if download:
        print(f"\n[CLOUD] Preparing Chunk {chunk_id} (Patient Slides {start_idx:03d} to {end_idx:03d})...")
        os.system("pip install awscli -q")
        for i in range(start_idx, end_idx + 1):
            tumor_file, normal_file = f"tumor_{i:03d}.tif", f"normal_{i:03d}.tif"
            os.system(
                f"python -m awscli s3 cp s3://camelyon-dataset/CAMELYON16/images/{tumor_file} ./ --no-sign-request > /dev/null 2>&1")
            os.system(
                f"python -m awscli s3 cp s3://camelyon-dataset/CAMELYON16/images/{normal_file} ./ --no-sign-request > /dev/null 2>&1")
        return len([f for f in os.listdir('.') if f.endswith('.tif')]) > 0
    else:
        for i in range(start_idx, end_idx + 1):
            os.system(f"rm -f tumor_{i:03d}.tif normal_{i:03d}.tif")


# =====================================================================
# 4. THE MARK 2 MASTER LOOP (WITH FAST-FORWARD RECOVERY)
# =====================================================================
extractor = MultiScaleWaveletExtractor().to(device)
gnn = AdvancedAttentionGNN().to(device)

optimizer = optim.AdamW(list(extractor.parameters()) + list(gnn.parameters()), lr=0.005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCELoss()

# Initialize Recovery Variables
start_epoch = 1
start_chunk = 0
best_val_loss = float('inf')

if os.path.exists(CHECKPOINT_PATH):
    print("\n[SYSTEM] Found checkpoint on Google Drive! Recovering brain state...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    gnn.load_state_dict(checkpoint['model_state_dict'])
    extractor.load_state_dict(checkpoint['extractor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    start_chunk = checkpoint.get('chunk', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"[SYSTEM] Successfully resumed. Picking up at Epoch {start_epoch}, Chunk {start_chunk + 1}")

# DINOv2 WARM-UP (First 3 slides to establish geometry baseline)
if start_epoch == 1 and start_chunk == 0 and not os.path.exists(CHECKPOINT_PATH):
    print("\n[HACK 1] INITIATING DINOv2 WARMUP (3 Slides)...")
    manage_cloud_chunk(1, download=True)
    slides = [f for f in os.listdir('.') if f.endswith('.tif')]
    mse_loss = nn.MSELoss()
    extractor.train()
    for slide in slides[:3]:
        coords = get_tissue_coordinates(slide)
        if coords:
            loader = DataLoader(ClinicalWSIDataset(slide, coords), batch_size=64, shuffle=True)
            for patches, _ in loader:
                optimizer.zero_grad()
                reconstructed, original = extractor.self_supervised_forward(patches.to(device))
                loss = mse_loss(reconstructed, original)
                loss.backward()
                optimizer.step()
    print("[SYSTEM] Warmup Complete. Wavelet eyes initialized.")
    manage_cloud_chunk(1, download=False)

# ----------------- PHASE 1 & 2: TRAIN AND VALIDATE -----------------
for epoch in range(start_epoch, EPOCHS + 1):
    print(
        f"\n=========================================\n         EPOCH {epoch}/{EPOCHS}         \n=========================================")
    print("\n[PHASE 1] TRAINING")

    for chunk in TRAIN_CHUNKS:

        # --- THE FAST-FORWARD GUARD ---
        if epoch == start_epoch and chunk <= start_chunk:
            print(f"[SYSTEM] Fast-forwarding past Chunk {chunk} (Already completed).")
            continue






        if not manage_cloud_chunk(chunk, download=True): continue
        optimizer.zero_grad()

        slides = [f for f in os.listdir('.') if f.endswith('.tif')]
        for slide in slides:
            label = 1.0 if "tumor" in slide else 0.0
            loss, _, pred, _, _ = process_slide(slide, label, extractor, gnn, criterion, is_training=True)
            print(f"  Train -> {slide} | Pred: {pred:.4f} | Loss: {loss:.4f}")
            torch.cuda.empty_cache()

        optimizer.step()

        # SECURE GOOGLE DRIVE SAVE (Including Chunk)
        torch.save({
            'epoch': epoch,
            'chunk': chunk,
            'best_val_loss': best_val_loss,
            'model_state_dict': gnn.state_dict(),
            'extractor_state_dict': extractor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, CHECKPOINT_PATH)
        print(f"[SYSTEM] Chunk {chunk} checkpoint saved to Google Drive.")

        manage_cloud_chunk(chunk, download=False)

    # CRITICAL RESET: Reset start_chunk so the next epoch doesn't accidentally skip!
    start_chunk = 0
    scheduler.step()

    print("\n[PHASE 2] VALIDATION")
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for chunk in VAL_CHUNKS:
            if not manage_cloud_chunk(chunk, download=True): continue
            slides = [f for f in os.listdir('.') if f.endswith('.tif')]
            for slide in slides:
                label = 1.0 if "tumor" in slide else 0.0
                loss, acc, pred, _, _ = process_slide(slide, label, extractor, gnn, criterion, is_training=False)
                val_loss += loss
                if acc == 100: val_correct += 1
                val_total += 1
            manage_cloud_chunk(chunk, download=False)

    val_accuracy = (val_correct / val_total) * 100 if val_total > 0 else 0
    print(f"  --> Validation Accuracy: {val_accuracy:.2f}% | Loss: {val_loss:.4f}")

    if val_loss < best_val_loss and val_loss > 0:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': gnn.state_dict(),
            'extractor_state_dict': extractor.state_dict()
        }, BEST_MODEL_PATH)
        print(f"[SYSTEM] Validation Loss Improved! Saved best weights to Drive.")

# ----------------- PHASE 3: THE FINAL HOLD-OUT TEST -----------------
print(
    "\n=========================================\n[PHASE 3] FINAL HOLD-OUT TEST\n=========================================")
if os.path.exists(BEST_MODEL_PATH):
    checkpoint = torch.load(BEST_MODEL_PATH)
    gnn.load_state_dict(checkpoint['model_state_dict'])
    extractor.load_state_dict(checkpoint['extractor_state_dict'])
    print("[SYSTEM] Loaded elite performing weights from Phase 2.")

gnn.eval()
extractor.eval()
true_labels = []
predictions = []

with torch.no_grad():
    for chunk in TEST_CHUNKS:
        if not manage_cloud_chunk(chunk, download=True): continue
        slides = [f for f in os.listdir('.') if f.endswith('.tif')]
        for slide in slides:
            label = 1.0 if "tumor" in slide else 0.0
            _, _, pred, _, _ = process_slide(slide, label, extractor, gnn, criterion, is_training=False)

            true_labels.append(int(label))
            predictions.append(1 if pred >= 0.5 else 0)
            print(
                f"  Test -> {slide} | Actual: {'Tumor' if label == 1.0 else 'Normal'} | Guessed: {'Tumor' if pred >= 0.5 else 'Normal'} ({pred:.4f})")
            torch.cuda.empty_cache()
        manage_cloud_chunk(chunk, download=False)

# Generate Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
plt.xlabel('AI Predicted Diagnosis')
plt.ylabel('Actual Patient Diagnosis')
plt.title('Mark 2 (100 Slides) - Final Accuracy Matrix')
plt.savefig('mark2_confusion_matrix.png')
print("\n[SYSTEM] Run complete! Download 'mark2_confusion_matrix.png' from the Colab file explorer.")




