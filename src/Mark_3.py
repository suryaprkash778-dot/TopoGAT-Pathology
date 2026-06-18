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
    os.system("pip install openslide-python torch-geometric torchvision -q")
    print("[SYSTEM] Environment built successfully! Proceeding to Mark 3 Execution...\n")

    import site
    from importlib import reload

    reload(site)

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch_geometric.nn import GATv2Conv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from google.colab import drive

# =====================================================================
# 1. PERSISTENT STORAGE & CONFIGURATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INITIATING MARK 3 (100-SLIDE RUN) ON {device.type.upper()} ---")

drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/Mark3_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mid_flight_checkpoint_mk3.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_mark3.pth")

EPOCHS = 10
TRAIN_CHUNKS = [1, 2, 3, 4, 5, 6, 7, 8]  # 80 Slides for Training
VAL_CHUNKS = [9]  # 10 Slides for Validation (Tuning)
TEST_CHUNKS = [10]  # 10 Slides for Final Test (The Exam)


# =====================================================================
# 2. DOMAIN GENERALIZATION (FDA & PHOTOMETRICS)
# =====================================================================
def fourier_amplitude_mix(x, ref, beta=0.08):
    """FDA: Swaps low-frequency color amplitudes to make the model stain-blind."""
    fx = torch.fft.fft2(x, dim=(-2, -1))
    fr = torch.fft.fft2(ref, dim=(-2, -1))
    ax, ph = torch.abs(fx), torch.angle(fx)
    ar = torch.abs(fr)

    ax = torch.fft.fftshift(ax, dim=(-2, -1))
    ar = torch.fft.fftshift(ar, dim=(-2, -1))

    B, C, H, W = x.shape
    b = int(min(H, W) * beta)
    cy, cx = H // 2, W // 2
    y1, y2 = max(0, cy - b), min(H, cy + b)
    x1, x2 = max(0, cx - b), min(W, cx + b)

    ax[:, :, y1:y2, x1:x2] = ar[:, :, y1:y2, x1:x2]
    ax = torch.fft.ifftshift(ax, dim=(-2, -1))

    out = torch.fft.ifft2(ax * torch.exp(1j * ph), dim=(-2, -1)).real
    return out.clamp(0, 1)


photometric_augment = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
])


# =====================================================================
# 3. DATA EXTRACTION & ARCHITECTURE
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


# --- NEW: ReCal Loss for Unsupervised Clustering ---
class ReCalLoss(nn.Module):
    def forward(self, node_embeddings):
        probs = F.softmax(node_embeddings, dim=1)
        log_probs = F.log_softmax(node_embeddings, dim=1)
        classification_entropy = -torch.sum(probs * log_probs) / node_embeddings.shape[0]
        class_prob = probs.mean(dim=0)
        class_entropy = -torch.sum(class_prob * torch.log2(class_prob + 1e-8))
        return classification_entropy - class_entropy


# --- NEW & CORRECTED: Mark 3 GNN (GATv2 + Hydra Heads + Skip Connections) ---
class Mark3GNN(nn.Module):
    def __init__(self, hidden_dim=128, num_clusters=4):
        super().__init__()
        self.pos_encoder = nn.Linear(2, hidden_dim)

        # ---> ADD THIS: The Dynamic Edge Pruning Gate <---
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)

        self.decoder = nn.Linear(hidden_dim, hidden_dim)  # MSE Structure Head
        self.cluster_head = nn.Linear(hidden_dim, num_clusters)  # ReCal Organization Head

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Sigmoid())
        self.attention_weights = nn.Linear(64, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, edge_index, coords):
        # 1. The Transformer Hack (Positional Encoding)
        norm_coords = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-5)
        pos_nodes = nodes + self.pos_encoder(norm_coords)

        # ---> THE SOLUTION: DYNAMIC EDGE PRUNING <---
        row, col = edge_index
        # Look at the features of the two nodes connected by each edge
        edge_features = torch.cat([pos_nodes[row], pos_nodes[col]], dim=1)
        
        # Score the biological relevance of the connection (0.0 to 1.0)
        edge_scores = self.edge_scorer(edge_features).squeeze(-1)
        
        # Snip the edges! Keep only the strong, relevant connections (> 50%)
        mask = edge_scores > 0.5
        pruned_edge_index = edge_index[:, mask]

        # Safety net: If it accidentally prunes everything, fallback to the original
        if pruned_edge_index.shape[1] == 0:
            pruned_edge_index = edge_index

        # 2. GATv2 Spatial Logic
        x1 = F.leaky_relu(self.conv1(pos_nodes, edge_index), 0.01)
        x2 = F.leaky_relu(self.conv2(x1, edge_index), 0.01)

        # 3. The ResNet Hack (Skip Connection)
        x_res = pos_nodes + x1 + x2

        # 4. The Hydra Heads
        x_recon = self.decoder(x_res)
        cluster_embeddings = self.cluster_head(x_res)

        # 5. Multi-Head Attention & Final Diagnosis
        a_v = self.attention_V(x_res)
        a_u = self.attention_U(x_res)
        weights = F.softmax(self.attention_weights(a_v * a_u), dim=0)

        prediction = torch.sigmoid(self.classifier(torch.sum(x_res * weights, dim=0, keepdim=True)))
        return prediction, cluster_embeddings, weights, x_recon

# =====================================================================
# 4. MARK 3 SLIDE PROCESSOR & EXPLAINABILITY VISUALS
# =====================================================================
def export_explainability_maps(slide_name, coords, weights, cluster_logits):
    """Generates and saves the Spatial Attention and Tissue Cluster maps."""
    coords_np = coords.cpu().numpy()
    weights_np = weights.cpu().detach().numpy().squeeze()

    # Normalize weights for heatmap
    weights_norm = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-8)

    # Extract unsupervised classes
    classes_np = torch.argmax(F.softmax(cluster_logits, dim=1), dim=1).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Thermal Attention Heatmap
    sc1 = ax1.scatter(coords_np[:, 0], -coords_np[:, 1], c=weights_norm, cmap='hot', s=10)
    ax1.set_title(f"Thermal Attention Map\n{slide_name}")
    plt.colorbar(sc1, ax=ax1)

    # Tissue Cluster Segmentation
    sc2 = ax2.scatter(coords_np[:, 0], -coords_np[:, 1], c=classes_np, cmap='tab10', s=10)
    ax2.set_title(f"Unsupervised Tissue Clusters\n{slide_name}")

    plt.tight_layout()
    map_path = f"{slide_name.replace('.tif', '')}_explainability.png"
    plt.savefig(map_path)
    plt.close()


def process_slide(slide_path, label, extractor, gnn, criterion_bce, criterion_mse, criterion_recal, is_training=True):
    coords = get_tissue_coordinates(slide_path)
    if not coords: return 0, 0, 0, None, None

    loader = DataLoader(ClinicalWSIDataset(slide_path, coords), batch_size=64, shuffle=False)
    all_nodes, all_coords = [], []

    context = torch.enable_grad() if is_training else torch.no_grad()
    if is_training:
        extractor.train();
        gnn.train()
    else:
        extractor.eval();
        gnn.eval()

    with context:
        for patches, batch_coords in loader:
            patches = patches.to(device)

            # --- FDA & PHOTOMETRIC DOMAIN GENERALIZATION ---
            if is_training:
                patches = photometric_augment(patches)
                if patches.size(0) > 1:
                    ref_idx = torch.randperm(patches.size(0))
                    patches = fourier_amplitude_mix(patches, patches[ref_idx])
            # -----------------------------------------------

            all_nodes.append(extractor(patches))
            all_coords.append(batch_coords.to(device))

    master_nodes = torch.cat(all_nodes)
    master_coords = torch.cat(all_coords)
    dist = torch.cdist(master_coords, master_coords)
    edge_index = ((dist <= 400.0) & (dist > 0)).nonzero(as_tuple=False).t().contiguous()

    prediction, cluster_embeddings, weights, x_recon = gnn(master_nodes, edge_index, master_coords)

    # --- HYDRA MULTI-TASK LOSS ---
    loss_diag = criterion_bce(prediction, torch.tensor([[label]]).to(device))
    loss_recon = criterion_mse(x_recon, master_nodes)
    loss_org = criterion_recal(cluster_embeddings)

    loss = loss_diag + (0.1 * loss_recon) + (0.05 * loss_org)

    if is_training: loss.backward()

    acc = 100 if (prediction.item() >= 0.5) == label else 0
    return loss.item(), acc, prediction.item(), weights, cluster_embeddings, master_coords


# =====================================================================
# 5. CLOUD MANAGER (YOUR EXACT MARK 2 CODE)
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
# 6. THE MARK 3 MASTER LOOP
# =====================================================================
extractor = MultiScaleWaveletExtractor().to(device)
gnn = Mark3GNN().to(device)

optimizer = optim.AdamW(list(extractor.parameters()) + list(gnn.parameters()), lr=0.0002, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# The 3-Headed Loss Functions
criterion_bce = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_recal = ReCalLoss()

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

if start_epoch == 1 and start_chunk == 0 and not os.path.exists(CHECKPOINT_PATH):
    print("\n[HACK 1] INITIATING DINOv2 WARMUP (3 Slides)...")
    manage_cloud_chunk(1, download=True)
    slides = [f for f in os.listdir('.') if f.endswith('.tif')]
    extractor.train()
    for slide in slides[:3]:
        coords = get_tissue_coordinates(slide)
        if coords:
            loader = DataLoader(ClinicalWSIDataset(slide, coords), batch_size=64, shuffle=True)
            for patches, _ in loader:
                optimizer.zero_grad()
                reconstructed, original = extractor.self_supervised_forward(patches.to(device))
                loss = criterion_mse(reconstructed, original)
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
        if epoch == start_epoch and chunk <= start_chunk:
            print(f"[SYSTEM] Fast-forwarding past Chunk {chunk} (Already completed).")
            continue

        if not manage_cloud_chunk(chunk, download=True): continue

        slides = [f for f in os.listdir('.') if f.endswith('.tif')]
        
        # --- NEW: Gradient Accumulation Setup ---
        accumulation_steps = 4 
        slide_count = 0
        optimizer.zero_grad() # Moved OUTSIDE the slide loop
        
        for slide in slides:
            label = 1.0 if "tumor" in slide else 0.0
            
            # 1. Process the slide
            loss, _, pred, _, _, _ = process_slide(slide, label, extractor, gnn, criterion_bce, criterion_mse,
                                                   criterion_recal, is_training=True)
            
            # 2. Scale loss and accumulate gradients
            scaled_loss = loss / accumulation_steps 
            scaled_loss.backward() 
            
            slide_count += 1
            
            # 3. Only step the optimizer every 4 slides
            if slide_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"  Train -> {slide} | Pred: {pred:.4f} | Loss: {loss:.4f}")
            torch.cuda.empty_cache()
            
        # 4. Catch any leftover gradients at the end of the chunk
        if slide_count % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

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

    start_chunk = 0
    scheduler.step()

    print("\n[PHASE 2] VALIDATION")
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for chunk in VAL_CHUNKS:
            if not manage_cloud_chunk(chunk, download=True): continue
            slides = [f for f in os.listdir('.') if f.endswith('.tif')]
            for slide in slides:
                label = 1.0 if "tumor" in slide else 0.0
                loss, acc, pred, _, _, _ = process_slide(slide, label, extractor, gnn, criterion_bce, criterion_mse,
                                                         criterion_recal, is_training=False)
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
true_labels, predictions = [], []

with torch.no_grad():
    for chunk in TEST_CHUNKS:
        if not manage_cloud_chunk(chunk, download=True): continue
        slides = [f for f in os.listdir('.') if f.endswith('.tif')]
        for slide in slides:
            label = 1.0 if "tumor" in slide else 0.0
            _, _, pred, weights, clusters, master_coords = process_slide(slide, label, extractor, gnn, criterion_bce,
                                                                         criterion_mse, criterion_recal,
                                                                         is_training=False)

            true_labels.append(int(label))
            predictions.append(1 if pred >= 0.5 else 0)
            print(
                f"  Test -> {slide} | Actual: {'Tumor' if label == 1.0 else 'Normal'} | Guessed: {'Tumor' if pred >= 0.5 else 'Normal'} ({pred:.4f})")

            # --- GENERATE EXPLAINABILITY VISUALS ---
            export_explainability_maps(slide, master_coords, weights, clusters)

            torch.cuda.empty_cache()
        manage_cloud_chunk(chunk, download=False)

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
plt.xlabel('AI Predicted Diagnosis')
plt.ylabel('Actual Patient Diagnosis')
plt.title('Mark 3 (100 Slides) - Final Accuracy Matrix')
plt.savefig('mark3_confusion_matrix.png')
print(
    "\n[SYSTEM] Run complete! Download 'mark3_confusion_matrix.png' and the generated explainability maps from the Colab file explorer.")

