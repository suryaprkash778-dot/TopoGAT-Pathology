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
    os.system("pip install openslide-python torch-geometric torchvision boto3 -q")
    print("[SYSTEM] Environment built successfully! Proceeding to TopoGAT Execution...\n")
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
from torch.utils.tensorboard import SummaryWriter  # --- THE FIX: Telemetry Engine ---
import torchvision.transforms.functional as TF
from torch_geometric.nn import GATv2Conv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
# --- THE FIX: Environment-Aware Imports ---
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import concurrent.futures
# =====================================================================
# 1. PERSISTENT STORAGE & CONFIGURATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INITIATING MARK 4 (TopoGAT) ON {device.type.upper()} ---")

# --- THE FIX: Dynamic Checkpoint Routing ---
# Automatically route saves to Google Drive if available, otherwise use local disk
if IN_COLAB:
    drive.mount('/content/drive')
    CHECKPOINT_DIR = "/content/drive/MyDrive/TopoGAT_Checkpoints"
else:
    print("[SYSTEM] Non-Colab environment detected. Defaulting to local storage.")
    CHECKPOINT_DIR = "./TopoGAT_Checkpoints"
    
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# THE FIX: Hard reset to Mark 10 for isolated attention temperature testing
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mid_flight_checkpoint_mk10.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_mark10.pth")
# --- THE FIX: Dynamic VRAM Sniffer ---
def get_optimal_batch_size():
    if not torch.cuda.is_available():
        print("[SYSTEM] CPU Mode Detected. Throttling batch size to 16.")
        return 16
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 30.0:
        print(f"[SYSTEM] Titan-Class GPU Detected ({vram_gb:.1f}GB). Maxing throughput (Batch: 256).")
        return 256
    elif vram_gb >= 20.0:
        print(f"[SYSTEM] High-End GPU Detected ({vram_gb:.1f}GB). Scaling throughput (Batch: 128).")
        return 128
    else:
        print(f"[SYSTEM] Standard GPU Detected ({vram_gb:.1f}GB). Setting safe baseline (Batch: 64).")
        return 64

# --- THE MASTER HYPERPARAMETER DIAL ---
CONFIG = {
    "epochs": 10,
    "train_chunks": [1, 2, 3, 4, 5, 6, 7, 8],
    "val_chunks": [9],
    "test_chunks": [10],
    
    "max_patches": 800,        # How many tissue chunks to extract per slide
    "grid_bins": 29,           # Math: int(sqrt(max_patches)) + 1
    "hash_mult": 31,           # Must be slightly larger than grid_bins
    "batch_size": get_optimal_batch_size(),  # --- THE FIX: Auto-Scaling Allocation ---
    
    "gnn_heads": 4,            # Number of multi-core attention heads
    "connect_radius": 1500.0,  # THE FIX: Widened to prevent sparse graph starvation
    

    # --- NOVELTY MATH DIALS ---
    "morpho_thresh": 0.75,     # Minimum biological cosine similarity to allow a connection
    
    
    "lr": 0.0002,              # Learning Rate
    "grad_accum": 4            # How many slides to average before stepping
}

EPOCHS = CONFIG["epochs"]
TRAIN_CHUNKS = CONFIG["train_chunks"]
VAL_CHUNKS = CONFIG["val_chunks"]
TEST_CHUNKS = CONFIG["test_chunks"] 

# =====================================================================
# 2. DOMAIN GENERALIZATION (FDA & PHOTOMETRICS)
# =====================================================================
def fourier_amplitude_mix(x, ref, beta=0.08):
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
    T.ColorJitter(brightness=0.2, contrast=0.2), # Kept Brightness/Contrast
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) # Restored from Mark 3
])

# =====================================================================
# 3. DATA EXTRACTION & MEMORY MANAGEMENT
# =====================================================================
def get_tissue_coordinates(slide_path, level=4, patch_size=256):
    try:
        slide = openslide.OpenSlide(slide_path)
        
        # Safely fallback if a slide has fewer levels than expected
        if level >= slide.level_count:
            level = slide.level_count - 1
            
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
        thumb_np = np.array(thumb)
        
        # --- THE FIX: OpenSlide Alpha-to-White Blending ---
        rgb = thumb_np[:, :, :3]
        alpha = thumb_np[:, :, 3]
        rgb[alpha == 0] = [255, 255, 255]
        
        thumb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        y_coords, x_coords = np.nonzero(mask)

        # Use round() before int() to prevent float-truncation drift on gigapixel coordinates
        scale_factor = int(round(slide.level_downsamples[level]))
        max_w, max_h = slide.dimensions
        
        # --- THE FIX: Out-of-Bounds Boundary Guard ---
        # Strip out any patches that bleed off the glass to prevent black-pixel padding
        coords = []
        for i in range(len(x_coords)):
            x = int(round(x_coords[i] * scale_factor))
            y = int(round(y_coords[i] * scale_factor))
            
            if x + patch_size <= max_w and y + patch_size <= max_h:
                coords.append([x, y])

        if len(coords) > CONFIG["max_patches"]:
            coords_np = np.array(coords)
            x_bins = np.linspace(coords_np[:,0].min(), coords_np[:,0].max(), CONFIG["grid_bins"])
            y_bins = np.linspace(coords_np[:,1].min(), coords_np[:,1].max(), CONFIG["grid_bins"])
            x_idx = np.digitize(coords_np[:,0], x_bins)
            y_idx = np.digitize(coords_np[:,1], y_bins)
            
            grid_keys = x_idx * CONFIG["hash_mult"] + y_idx
            
            _, unique_idx = np.unique(grid_keys, return_index=True)
            coords = coords_np[unique_idx[:CONFIG["max_patches"]]].tolist()
        return coords
    
    # --- THE FIX: Exception Visibility ---
    except Exception as e:
        print(f"  [WARNING] WSI Extraction failed for {slide_path}: {e}")
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

    # CLAUDE FIX: Proper OpenSlide memory cleanup to prevent RAM leaks
    def __del__(self):
        try:
            self.slide.close()
        except:
            pass

class MultiScaleWaveletExtractor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        haar_weights = torch.tensor([[[1., 1.], [1., 1.]], [[-1., -1.], [1., 1.]],
                                     [[-1., 1.], [-1., 1.]], [[1., -1.], [-1., 1.]]]).unsqueeze(1) / 4.0
        self.wavelet = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.wavelet.weight = nn.Parameter(haar_weights, requires_grad=True)

        # --- THE FIX: Dynamic Architecture Scaling ---
        self.pool_sizes = [4, 8, 16]
        self.pool_small = nn.AdaptiveAvgPool2d((self.pool_sizes[0], self.pool_sizes[0]))
        self.pool_med = nn.AdaptiveAvgPool2d((self.pool_sizes[1], self.pool_sizes[1]))
        self.pool_large = nn.AdaptiveAvgPool2d((self.pool_sizes[2], self.pool_sizes[2]))

        # Automatically calculate the flattened tensor dimension: 4 channels * sum(H*W)
        self.raw_feat_dim = 4 * sum(s**2 for s in self.pool_sizes)

        self.compressor = nn.Sequential(
            nn.Linear(self.raw_feat_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.raw_feat_dim)
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

class ReCalLoss(nn.Module):
    def forward(self, node_embeddings):
        probs = F.softmax(node_embeddings, dim=1)
        log_probs = F.log_softmax(node_embeddings, dim=1)
        classification_entropy = -torch.sum(probs * log_probs) / node_embeddings.shape[0]
        class_prob = probs.mean(dim=0)
        class_entropy = -torch.sum(class_prob * torch.log(class_prob + 1e-8))
        
        # --- THE FIX: The Hydra Exploit Clamp ---
        # We shift the entropy to be strictly non-negative so the 
        # uncertainty weighting cannot multiply a negative loss to -infinity.
        max_entropy = torch.log(torch.tensor(node_embeddings.shape[1], dtype=torch.float32, device=node_embeddings.device))
        loss = (classification_entropy - class_entropy) + max_entropy
        
        return torch.clamp(loss, min=0.0)

# =====================================================================
# 4. MARK 4 GNN: TopoGAT
# =====================================================================
class TopoGAT(nn.Module):
    def __init__(self, hidden_dim=128, num_clusters=4):
        super().__init__()
        self.pos_encoder = nn.Linear(2, hidden_dim)

        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # --- THE DARTS UPGRADE: Self-Tuning Biological Dials ---
        self.learned_tau = nn.Parameter(torch.tensor([200.0]))
        self.learned_thresh = nn.Parameter(torch.tensor([0.5]))

        # FIX: Divisible by 4 math alignment
        h = CONFIG["gnn_heads"]
        
        # --- THE FIX: edge_dim=1 allows the DARTS gradients to flow backward! ---
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // h, heads=h, concat=True, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // h, heads=h, concat=True, edge_dim=1)

        self.decoder = nn.Linear(hidden_dim, hidden_dim) 
        self.cluster_head = nn.Linear(hidden_dim, num_clusters) 

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Sigmoid())
        self.attention_weights = nn.Linear(64, 1)
        
        # THE FIX: Temperature to sharpen attention away from uniform collapse
        self.attn_temperature = 0.1
        
        

        # THE FIX: Doubling input size for Dual-Stream MIL (Attention + Max)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

        # --- NEW: Self-Regulating Hydra Loss Controllers ---
        # The AI learns these 3 values to dynamically balance its own loss functions
        self.loss_log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, nodes, edge_index, coords):
        norm_coords = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-5)
        pos_nodes = nodes + self.pos_encoder(norm_coords)

        row, col = edge_index
        edge_features = torch.cat([pos_nodes[row], pos_nodes[col]], dim=1)
        
        # FIX: Added -1 to squeeze to prevent 0-D tensor edge cases
        bio_scores = self.edge_scorer(edge_features).squeeze(-1)
        
        # --- THE DARTS UPGRADE: Continuous Relaxation & Soft Masking ---
        # 1. Capture the active AMP dtype (FP16 or FP32)
        current_dtype = pos_nodes.dtype
        
        # 2. Cast the DARTS parameters to match the context
        safe_tau = torch.clamp(self.learned_tau, min=10.0).to(current_dtype)
        edge_distances = torch.norm(coords[row] - coords[col], dim=1).to(current_dtype)
        spatial_decay = torch.exp(-edge_distances / safe_tau)
        
        decayed_scores = bio_scores * spatial_decay
        
        steepness = 15.0  
        edge_weights = torch.sigmoid((decayed_scores - self.learned_thresh.to(current_dtype)) * steepness)
        
        mask = edge_weights > 0.05  
        pruned_edge_index = edge_index[:, mask]
        
        # 3. Force shape to (N, 1) AND strictly enforce dtype for PyG
        pruned_edge_weights = edge_weights[mask].view(-1, 1).to(current_dtype) 

        if pruned_edge_index.shape[1] == 0:
            pruned_edge_index = edge_index
            # 4. Enforce the same dtype on the fallback tensor
            pruned_edge_weights = torch.ones((edge_index.shape[1], 1), dtype=current_dtype, device=device)

        # Inject the structural weights into the GAT to preserve the gradient graph!
        x1 = F.leaky_relu(self.conv1(pos_nodes, pruned_edge_index, edge_attr=pruned_edge_weights), 0.01)
        x2 = F.leaky_relu(self.conv2(x1, pruned_edge_index, edge_attr=pruned_edge_weights), 0.01)

        x_res = pos_nodes + x1 + x2

        x_recon = self.decoder(x_res)
        cluster_embeddings = self.cluster_head(x_res)

        a_v = self.attention_V(x_res)
        a_u = self.attention_U(x_res)
        
        # THE FIX: Divide by temperature before softmax to sharpen the distribution
        attn_logits = self.attention_weights(a_v * a_u) / self.attn_temperature
        weights = F.softmax(attn_logits, dim=0).view(-1, 1)

        # 1. The Spatial Graph Features (Attention-weighted MIL)
        attn_pooled = torch.sum(x_res * weights, dim=0, keepdim=True)
        
        # 2. THE SKIP CONNECTION: Max Pooling (The Needle Finder)
        # [0] unpacks the max values from the indices returned by torch.max
        max_pooled = torch.max(x_res, dim=0, keepdim=True)[0] 
        
        # 3. Combine both streams and classify
        combined_feat = torch.cat([attn_pooled, max_pooled], dim=1)
        logits = self.classifier(combined_feat)

        # THE FIX: Pack all graph and pooling diagnostics into a single dictionary
        telemetry = {
            'edges_kept': pruned_edge_index.shape[1],
            'edges_total': edge_index.shape[1],
            'attn_norm': torch.norm(attn_pooled, p=2).item(),
            'max_norm': torch.norm(max_pooled, p=2).item()
        }
        
        return logits, cluster_embeddings, weights, x_recon, self.loss_log_vars, telemetry

# =====================================================================
# 5. EXPLAINABILITY & PROCESSING ENGINE
# =====================================================================
def export_explainability_maps(slide_name, coords, weights, cluster_logits):
    coords_np = coords.cpu().numpy()
    weights_np = weights.cpu().detach().numpy().squeeze()

    weights_norm = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-8)
    classes_np = torch.argmax(F.softmax(cluster_logits, dim=1), dim=1).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = ax1.scatter(coords_np[:, 0], -coords_np[:, 1], c=weights_norm, cmap='hot', s=10)
    ax1.set_title(f"Thermal Attention Map\n{slide_name}")
    plt.colorbar(sc1, ax=ax1)

    sc2 = ax2.scatter(coords_np[:, 0], -coords_np[:, 1], c=classes_np, cmap='tab10', s=10)
    ax2.set_title(f"Unsupervised Tissue Clusters\n{slide_name}")

    plt.tight_layout()
    map_path = f"{slide_name.replace('.tif', '')}_explainability.png"
    plt.savefig(map_path)
    plt.close()

def process_slide(slide_path, label, extractor, gnn, criterion_bce, criterion_mse, criterion_recal, is_training=True):
    coords = get_tissue_coordinates(slide_path)
    if not coords: return None, 0, 0, None, None, None, None, {} 

    loader = DataLoader(ClinicalWSIDataset(slide_path, coords), batch_size=CONFIG["batch_size"], shuffle=False)
    all_nodes, all_coords = [], []

    context = torch.enable_grad() if is_training else torch.no_grad()
    if is_training:
        extractor.train(); gnn.train()
    else:
        extractor.eval(); gnn.eval()

    # --- AMP Autocast Context ---
    amp_context = torch.cuda.amp.autocast(enabled=is_training and torch.cuda.is_available())
    
    with context, amp_context:
        for patches, batch_coords in loader:
            patches = patches.to(device)

            if is_training:
                patches = photometric_augment(patches)
                if patches.size(0) > 1:
                    ref_idx = torch.randperm(patches.size(0), device=patches.device)
                    patches = fourier_amplitude_mix(patches, patches[ref_idx])

            all_nodes.append(extractor(patches))
            all_coords.append(batch_coords.to(device))

        # ---------------------------------------------------------
        # ALL OF THIS IS NOW SAFELY INDENTED INSIDE THE CONTEXT!
        # ---------------------------------------------------------
        if len(all_nodes) == 0:
            return None, 0, 0, None, None, None, None, {}

        from scipy.spatial import cKDTree
        master_nodes = torch.cat(all_nodes)
        master_coords = torch.cat(all_coords)
        
        # 1. Find physical neighbors using an ultra-fast CPU KDTree
        coords_np = master_coords.cpu().numpy()
        tree = cKDTree(coords_np)
        pairs = tree.query_pairs(CONFIG["connect_radius"])
        
        if len(pairs) > 0:
            pairs_np = np.array(list(pairs), dtype=np.int64)
            edge_tensor = torch.tensor(pairs_np, dtype=torch.long, device=device)
            
            row, col = edge_tensor[:, 0], edge_tensor[:, 1]
            
            # 2. Compute Biological Similarity
            F_norm = F.normalize(master_nodes, p=2, dim=1)
            sim_scores = (F_norm[row] * F_norm[col]).sum(dim=1)
            
            # 3. Filter connections by biological threshold
            mask = sim_scores >= CONFIG["morpho_thresh"]
            row_filtered = row[mask]
            col_filtered = col[mask]
            
            # THE FIX: Diagnostic-only telemetry for the morpho_thresh filter stage
            morpho_survival = {
                'radius_pairs': len(pairs),
                'morpho_kept': int(mask.sum().item()) if len(pairs) > 0 else 0
            }
            
            # 4. Make surviving edges bidirectional or fallback to k-NN
            if len(row_filtered) > 0:
                edge_index = torch.stack([
                    torch.cat([row_filtered, col_filtered]), 
                    torch.cat([col_filtered, row_filtered])
                ], dim=0)
            else:
                from torch_geometric.nn import knn_graph
                from torch_geometric.utils import to_undirected
                k_val = min(4, master_coords.size(0) - 1)
                edge_index = to_undirected(knn_graph(master_coords, k=k_val)) if k_val > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            from torch_geometric.nn import knn_graph
            from torch_geometric.utils import to_undirected
            k_val = min(4, master_coords.size(0) - 1)
            edge_index = to_undirected(knn_graph(master_coords, k=k_val)) if k_val > 0 else torch.empty((2, 0), dtype=torch.long, device=device)

        # Catch the expanded telemetry dictionary
        logits, cluster_embeddings, weights, x_recon, log_vars, telemetry = gnn(master_nodes, edge_index, master_coords)
        
        loss_diag = criterion_bce(logits, torch.tensor([[label]]).to(device))
        loss_recon = criterion_mse(x_recon, master_nodes)
        loss_org = criterion_recal(cluster_embeddings)

        # AI-Controlled Adaptive Loss Balancing
        safe_log_vars = torch.clamp(log_vars, min=-5.0, max=5.0)

        # THE FIX: Pure, unweighted BCE gradient. Starvation is now impossible.
        loss_0 = loss_diag 
        loss_1 = loss_recon * torch.exp(-safe_log_vars[1]) + safe_log_vars[1]
        loss_2 = loss_org * torch.exp(-safe_log_vars[2]) + safe_log_vars[2]

        loss = loss_0 + loss_1 + loss_2
        
        # Route the expanded telemetry
        raw_metrics = {
            'diag': loss_diag.item(), 'recon': loss_recon.item(), 'org': loss_org.item(),
            'edges_kept': telemetry['edges_kept'], 'edges_total': telemetry['edges_total'],
            'attn_norm': telemetry['attn_norm'], 'max_norm': telemetry['max_norm'],
            'radius_pairs': morpho_survival['radius_pairs'],
            'morpho_kept': morpho_survival['morpho_kept']
        }

        if not is_training:
            loss = loss.detach()
            weights = weights.detach()
            cluster_embeddings = cluster_embeddings.detach()

        pred_prob = torch.sigmoid(logits).item()
        acc = int((pred_prob >= 0.5) == bool(label)) * 100
        
        return loss, acc, pred_prob, weights, cluster_embeddings, master_coords, log_vars, raw_metrics

    

# =====================================================================
# 6. CLOUD MANAGER (PARALLEL I/O)
# =====================================================================
def download_s3_file(bucket_name, s3_key, local_path):
    """Worker function to download a single file safely."""
    if os.path.exists(local_path): 
        return True
    
    # Configure anonymous access to the public bucket
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        print(f"  [I/O ERROR] Failed to download {s3_key}: {e}")
        return False

def manage_cloud_chunk(chunk_id, download=True):
    start_idx = ((chunk_id - 1) * 10) + 1
    end_idx = start_idx + 9
    
    bucket = "camelyon-dataset"
    prefix = "CAMELYON16/images/"

    if download:
        print(f"\n[CLOUD] Parallel-Fetching Chunk {chunk_id} (Slides {start_idx:03d} to {end_idx:03d})...")
        files_to_download = []
        for i in range(start_idx, end_idx + 1):
            files_to_download.append((bucket, f"{prefix}tumor_{i:03d}.tif", f"tumor_{i:03d}.tif"))
            files_to_download.append((bucket, f"{prefix}normal_{i:03d}.tif", f"normal_{i:03d}.tif"))
        
        # Spin up 8 concurrent threads to saturate network bandwidth
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_s3_file, *args) for args in files_to_download]
            concurrent.futures.wait(futures)
            
        # Verify exact expected files are on disk to prevent silent partial-chunk failures
        expected_files = [f[2] for f in files_to_download]
        valid_files = [f for f in expected_files if os.path.exists(f)]
        
        if len(valid_files) < len(expected_files):
            print(f"  [WARNING] I/O Mismatch! Expected {len(expected_files)} slides, but only secured {len(valid_files)}.")
            
        return len(valid_files) > 0
    else:
        # Native Python file cleanup instead of OS sub-shells
        for i in range(start_idx, end_idx + 1):
            try:
                if os.path.exists(f"tumor_{i:03d}.tif"): os.remove(f"tumor_{i:03d}.tif")
                if os.path.exists(f"normal_{i:03d}.tif"): os.remove(f"normal_{i:03d}.tif")
            except Exception:
                pass
        return True

# =====================================================================
# 7. THE MASTER LOOP
# =====================================================================
extractor = MultiScaleWaveletExtractor().to(device)
gnn = TopoGAT().to(device)

# 1. Isolate the "DNA" (The self-optimizing hyperparameters and Hydra dials)
meta_params = [gnn.learned_tau, gnn.learned_thresh, gnn.loss_log_vars]
meta_ids = set(id(p) for p in meta_params)

# 2. Group the standard "Brain" weights (Extractor excluded, as it will be frozen)
base_params = [p for p in gnn.parameters() if id(p) not in meta_ids]

# --- THE FIX: Multi-Speed Meta-Optimizer ---
optimizer = optim.AdamW([
    # LOBE 1: The Neural Network. Cautious learning rate, standard weight decay.
    {'params': base_params, 'lr': CONFIG["lr"], 'weight_decay': 1e-4},
    
    # THE FIX: Throttled Meta-Engine. Slower learning rate, moderate regularization.
    {'params': meta_params, 'lr': 0.002, 'weight_decay': 1e-4}
])

# --- THE FIX: Accurate Warmup Tracking ---
# We must divide by grad_accum so the steps match the optimizer, not the slides!
warmup_steps = (len(CONFIG["train_chunks"]) * 10) // CONFIG["grad_accum"]

def lr_lambda(step):
    return min(1.0, step / max(1, warmup_steps))

warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()
criterion_recal = ReCalLoss()
# --- NEW: Initialize AMP Scaler ---
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
# --- NEW: Boot up TensorBoard ---
writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_DIR, "tensorboard_logs"))

start_epoch = 1
start_chunk = 0
global_step = 0  # --- THE FIX: Explicit Step Tracker ---
best_val_loss = float('inf')

if os.path.exists(CHECKPOINT_PATH):
    print("\n[SYSTEM] Found checkpoint! Recovering brain state...")
    
    # 1. LOAD THE FILE FIRST
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # 2. Extract progression variables
    start_epoch = checkpoint.get('epoch', 1)
    start_chunk = checkpoint.get('chunk', 0)
    global_step = checkpoint.get('global_step', 0) 
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"[SYSTEM] Successfully resumed. Picking up at Epoch {start_epoch}, Chunk {start_chunk + 1}, Global Step {global_step}")
    
    # 3. Load model and optimizer states
    gnn.load_state_dict(checkpoint['model_state_dict'])
    extractor.load_state_dict(checkpoint['extractor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. Explicitly push all internal optimizer momentum buffers to the GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    # 5. Safely load scaler and schedulers
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'warmup_scheduler_state_dict' in checkpoint:
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 6. Restore Deterministic RNG States
    if 'torch_rng' in checkpoint:
        # Pull the PyTorch RNG states back to the CPU where the manager expects them
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['torch_cuda_rng']])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])

if start_epoch == 1 and start_chunk == 0 and not os.path.exists(CHECKPOINT_PATH):
    print("\n[HACK 1] INITIATING WARMUP (3 Slides)...")
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
    
# THE FIX: Universal Extractor Freeze (Applies on both fresh runs and checkpoint resumes)
for param in extractor.parameters():
    param.requires_grad = False
extractor.eval()
print("[SYSTEM] Extractor weights explicitly frozen for this session.")
# ----------------- PHASE 1 & 2: TRAIN AND VALIDATE -----------------
for epoch in range(start_epoch, EPOCHS + 1):
    print(f"\n=========================================\n         EPOCH {epoch}/{EPOCHS}         \n=========================================")
    print("\n[PHASE 1] TRAINING")

    for chunk in TRAIN_CHUNKS:
        if epoch == start_epoch and chunk <= start_chunk:
            print(f"[SYSTEM] Fast-forwarding past Chunk {chunk}.")
            continue

        if not manage_cloud_chunk(chunk, download=True): continue

        slides = [f for f in os.listdir('.') if f.endswith('.tif')]
        
        accumulation_steps = CONFIG["grad_accum"]
        slide_count = 0
        optimizer.zero_grad() 
        
        for slide in slides:
            label = 1.0 if "tumor" in slide else 0.0
            
            # THE FIX: Safely capture 'weights' (4th variable) for the MIL diagnostic
            loss, _, pred, weights, _, _, log_vars, raw_metrics = process_slide(slide, label, extractor, gnn, criterion_bce, criterion_mse, criterion_recal, is_training=True)
            
            if loss is None: continue 
            
            # --- NEW: Write the raw, unweighted diagnostics immediately ---
            writer.add_scalars('Raw_Loss_Components', {
                'Diagnostic_BCE': raw_metrics['diag'],
                'Reconstruction_MSE': raw_metrics['recon'],
                'Clustering_Entropy': raw_metrics['org']
            }, global_step + slide_count) # Staggered slightly for per-slide resolution
            
            
            
            # 1. Scale the loss to prevent vanishing gradients
            scaled_loss = loss / accumulation_steps 
            scaler.scale(scaled_loss).backward()  
            
            slide_count += 1
            
            # Print to terminal
            print(f"  Train -> {slide} | Pred: {pred:.4f} | Diag: {raw_metrics['diag']:.4f} | Thresh: {gnn.learned_thresh.item():.4f} | Edges: {raw_metrics['edges_kept']} / {raw_metrics['edges_total']} | MorphoSurvival: {raw_metrics['morpho_kept']} / {raw_metrics['radius_pairs']}")
            # THE FIX: MIL Attention & Stream Magnitude Diagnostics
            print(f"    ↳ MIL Attn | Max: {weights.max().item():.4f} | Min: {weights.min().item():.4f} | Std: {weights.std().item():.6f}")
            print(f"    ↳ Stream Norms | Attn: {raw_metrics['attn_norm']:.2f} | Max: {raw_metrics['max_norm']:.2f}")
            
            # Master Step & Telemetry Sync
            if slide_count % accumulation_steps == 0:
                scaler.step(optimizer)            
                scaler.update()                   
                
                global_step += 1
                if global_step <= warmup_steps:
                    warmup_scheduler.step()
                optimizer.zero_grad()
                
                # --- Sync Telemetry with Global Steps ---
                writer.add_scalar('Training/Loss', scaled_loss.item(), global_step)
                writer.add_scalar('System/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalars('Hydra_Uncertainty_Dials', {
                    'Diagnostic': log_vars[0].item(),
                    'Reconstruction': log_vars[1].item(),
                    'Clustering': log_vars[2].item()
                }, global_step)
            
            torch.cuda.empty_cache()
        
        # --- THE FIX: Pristine Catch Block ---
        if slide_count > 0 and slide_count % accumulation_steps != 0:
            scaler.step(optimizer)                
            scaler.update()
            
            global_step += 1
            if global_step <= warmup_steps:
                warmup_scheduler.step()
            optimizer.zero_grad()
            
            writer.add_scalar('Training/Loss', scaled_loss.item(), global_step)
            writer.add_scalar('System/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalars('Hydra_Uncertainty_Dials', {
                'Diagnostic': log_vars[0].item(),
                'Reconstruction': log_vars[1].item(),
                'Clustering': log_vars[2].item()
            }, global_step)
            
        torch.cuda.empty_cache()
        # --- THE FIX: Capturing Deterministic RNG States ---
        torch.save({
            'epoch': epoch,
            'chunk': chunk,
            'best_val_loss': best_val_loss,
            'model_state_dict': gnn.state_dict(),
            'extractor_state_dict': extractor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(), # --- NEW: Save AMP state ---
            'scheduler_state_dict': scheduler.state_dict(),
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
            'global_step': global_step,  # --- THE FIX: Save the clock ---
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate()
        }, CHECKPOINT_PATH)
        print(f"[SYSTEM] Chunk {chunk} checkpoint saved.")

        manage_cloud_chunk(chunk, download=False)

    start_chunk = 0
    
    # --- THE FIX: Prevent Scheduler Collisions ---
    # Only allow the Cosine Annealing to take over AFTER the warmup phase is 100% complete
    if global_step > warmup_steps:
        scheduler.step() 

    print("\n[PHASE 2] VALIDATION")
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for chunk in VAL_CHUNKS:
            if not manage_cloud_chunk(chunk, download=True): continue
            slides = [f for f in os.listdir('.') if f.endswith('.tif')]
            for slide in slides:
                label = 1.0 if "tumor" in slide else 0.0
                loss, acc, pred, _, _, _, _, _ = process_slide(slide, label, extractor, gnn, criterion_bce, criterion_mse, criterion_recal, is_training=False)
                
                if loss is not None: # CLAUDE FIX: Explicit None check
                    val_loss += loss.item() # CLAUDE FIX: Extract tensor for tallying
                    if acc == 100: val_correct += 1
                    val_total += 1
            manage_cloud_chunk(chunk, download=False)

    # CLAUDE FIX: Display validation warning if no valid slides were processed
    if val_total == 0:
        print("[WARNING] No valid slides processed in validation. Skipping model save.")
    else:
        val_accuracy = (val_correct / val_total) * 100
        print(f"  --> Validation Accuracy: {val_accuracy:.2f}% | Loss: {val_loss:.4f}")
        # --- NEW: Log Epoch-level Validation ---
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': gnn.state_dict(),
                'extractor_state_dict': extractor.state_dict()
            }, BEST_MODEL_PATH)
            print(f"[SYSTEM] Validation Loss Improved! Saved best weights.")

# ----------------- PHASE 3: THE FINAL HOLD-OUT TEST -----------------
print("\n=========================================\n[PHASE 3] FINAL HOLD-OUT TEST\n=========================================")
if os.path.exists(BEST_MODEL_PATH):
    # --- THE FIX: Hardware-Aware Testing ---
    # Force the elite weights directly onto the active hardware to prevent tensor collisions
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    
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
            _, _, pred, weights, clusters, master_coords, _, _ = process_slide(slide, label, extractor, gnn, criterion_bce, criterion_mse, criterion_recal, is_training=False)
            
            # CLAUDE FIX: The None-Guard to stop empty slides from crashing Explainability Maps
            if master_coords is None: continue 

            true_labels.append(int(label))
            predictions.append(1 if pred >= 0.5 else 0)
            print(f"  Test -> {slide} | Actual: {'Tumor' if label == 1.0 else 'Normal'} | Guessed: {'Tumor' if pred >= 0.5 else 'Normal'} ({pred:.4f})")

            export_explainability_maps(slide, master_coords, weights, clusters)
            torch.cuda.empty_cache()
            
        manage_cloud_chunk(chunk, download=False)

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
plt.xlabel('AI Predicted Diagnosis')
plt.ylabel('Actual Patient Diagnosis')
plt.title('TopoGAT (Mark 4) - Final Accuracy Matrix')
plt.savefig('mark4_confusion_matrix.png')
print("\n[SYSTEM] Run complete! Download 'mark4_confusion_matrix.png' from the Colab file explorer.")
