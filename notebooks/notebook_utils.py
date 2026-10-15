"""
Notebook Utilities for VL-JEPA vs CLIP CCTV Event Retrieval & Research Analysis.
Supports real PyTorch model training, proper train/test splitting, live FAISS retrieval across 3 datasets,
viewpoint-aware filtering, semantic query routing, OpenCV keyframe extraction, and publication exports.
Optimized for NVIDIA RTX 4000 Ada Generation (20GB).
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def get_workspace_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "implementation").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent

WORKSPACE_ROOT = get_workspace_root()
BASE_DIR = WORKSPACE_ROOT / "notebooks"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

def set_all_seeds(seed: int = 42):
    """Establishes deterministic seeds across random, numpy, and torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Determinism] Global seeds set to: {seed}")


# -----------------------------------------------------------------------------
# 1. SEMANTIC SURVEILLANCE DESCRIPTIONS
# -----------------------------------------------------------------------------
# Provides rich spatiotemporal context to align video features with text concepts
SURVEILLANCE_DESCRIPTIONS = {
    # RLVS (Violence)
    "violence": "a CCTV surveillance camera recording violent street fighting, punching, physical brawl, or assault",
    "nonviolence": "a CCTV surveillance camera recording peaceful people walking, standing, or talking without violence",
    
    # UCF-ARG (Actions & Viewpoints)
    "boxing": "two people boxing, throwing punches, or fighting in an outdoor area",
    "carrying": "a person carrying a box, container, or heavy object while moving",
    "clapping": "a person clapping their hands together in front of their body",
    "digging": "a person digging into the ground using a shovel or tool",
    "jogging": "a person jogging or running at a moderate pace outdoors",
    "openclosetrunk": "a person opening or closing the trunk of a car or vehicle",
    "running": "a person running fast across the field or surveillance area",
    "throwing": "a person throwing an object with their arm",
    "walking": "a pedestrian walking normally across the walkway",
    "waving": "a person waving their hand or arm in the air",

    # TinyVIRAT-v2 (Vehicles, Objects, Interactions)
    "vehicle_moving": "a car, vehicle, or truck moving or driving on a road or parking lot",
    "vehicle_stopping": "a car, automobile, or vehicle slowing down and coming to a stop",
    "vehicle_turning_left": "a car or vehicle turning left at an intersection or driveway",
    "vehicle_turning_right": "a car or vehicle turning right on a street or parking lot",
    "vehicle_starting": "a parked vehicle or car starting engine and beginning to drive",
    "activity_carrying": "a person carrying a bag, box, package, or object",
    "Transport_HeavyCarry": "individuals transporting, loading, or carrying heavy cargo or equipment",
    "Loading": "a person loading an object, luggage, or goods into a vehicle or area",
    "Pull": "a person pulling an object, cart, door, or equipment behind them",
    "Push": "a person pushing an object, cart, door, or trolley forward",
    "Opening": "a person opening a door, trunk, or container",
    "Closing": "a person closing a door, trunk, or container",
    "Interacts": "people interacting with each other or with an object in surveillance view",
    "specialized_using_tool": "a person operating or using a hand tool or maintenance equipment",
    "activity_walking": "a person walking on the surveillance camera feed",
    "activity_running": "a person running across the camera scene",
    "activity_standing": "a person standing still or waiting in place",
    "Talking": "two or more people talking or communicating in conversation",
    "specialized_talking_phone": "a person holding a mobile phone to their ear and talking",
    "specialized_texting_phone": "a person holding a mobile phone in hands and texting",

    # UCF-Crime (Anomalies & Real Surveillance Crimes)
    "abuse": "a CCTV surveillance camera recording physical abuse, cruelty, or mistreatment of a victim",
    "arrest": "police officers arresting, tackling, handcuffing, or taking a criminal suspect into custody",
    "arson": "a person committing arson, lighting a fire, flame, or intentionally burning a building or car",
    "assault": "a violent physical assault, battery, street beating, or attacker beating an innocent victim",
    "burglary": "a burglar breaking into a building, home burglary, forced door entry, or house robbery",
    "explosion": "a sudden violent explosion, detonated blast, smoke, fireball, or destructive blast incident",
    "fighting": "street fighting, violent punching, brawl, physical altercation between people on CCTV",
    "normal": "normal everyday surveillance scene with ordinary peaceful pedestrian and vehicle movement",
    "roadaccidents": "a traffic accident, car crash, vehicle collision, or automobile impact on a street or highway",
    "robbery": "armed robbery, weapon holdup, threatening cashier or victim, or store armed robbery",
    "shooting": "a shooting incident, person firing gun, discharging firearm, or active shooter in surveillance",
    "shoplifting": "a customer shoplifting, concealing store merchandise into bag or clothing, retail theft",
    "stealing": "theft, a person stealing an object, snatching property, bicycle theft, or stealing packages",
    "vandalism": "vandalism, deliberate property destruction, smashing glass windows, or graffiti vandalism",
}

# Helper to generate nuanced, sample-specific text descriptions
def generate_sample_prompt(item: Dict[str, Any]) -> str:
    """
    Generates dynamic, sample-specific text supervision incorporating:
    - Base surveillance category and action
    - Camera viewpoint (aerial, rooftop, ground)
    - Actor/run identifier or multi-labels (carrying, vehicles, tools)
    - Anomaly / crime event identifier
    This prevents cross-modal collapse and enables true instance-level retrieval.
    """
    dataset = item.get("dataset", "")
    label = str(item.get("label", "unknown")).lower()
    meta = item.get("metadata", {})
    
    base_desc = SURVEILLANCE_DESCRIPTIONS.get(label, f"a surveillance camera capturing {label}")
    
    if dataset == "ucf_arg":
        viewpoint = meta.get("viewpoint", "ground")
        actor = meta.get("actor", "subject")
        return f"{base_desc}, captured from an outdoor {viewpoint} surveillance angle of {actor}"
    elif dataset == "tinyvirat":
        all_labels = meta.get("all_labels", [label])
        if isinstance(all_labels, list) and len(all_labels) > 1:
            multi_str = ", ".join([str(l).replace('_', ' ') for l in all_labels])
            return f"{base_desc}, surveillance activity involving {multi_str}"
        return f"{base_desc}, fixed surveillance camera viewpoint"
    elif dataset == "rlvs":
        vid_id = Path(item.get("path", "")).stem
        return f"{base_desc}, surveillance footage sequence {vid_id}"
    elif dataset == "ucf_crime":
        vid_name = meta.get("filename", Path(item.get("path", "")).name)
        return f"{base_desc}, incident clip {vid_name}"
    return base_desc


def create_clip_text_embeddings_for_metadata(
    metadata: List[Dict[str, Any]], 
    device: Optional[str] = None, 
    batch_size: int = 64,
    use_cache: bool = True
) -> np.ndarray:
    """Generates (or loads cached) CLIP text embeddings for each surveillance metadata record."""
    cache_path = WORKSPACE_ROOT / "implementation" / "results" / "embeddings" / "clip_all_text_embeddings.npy"
    if use_cache and cache_path.exists():
        try:
            arr = np.load(cache_path)
            if len(arr) == len(metadata):
                print(f"[Text Features] Loaded cached CLIP text embeddings: {arr.shape}")
                return arr
        except Exception:
            pass

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import CLIPModel, CLIPTokenizer
    model_id = "openai/clip-vit-base-patch32"
    print(f"[Text Features] Encoding {len(metadata):,} prompts with {model_id} on {dev}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(dev)
    model.eval()

    prompts = [generate_sample_prompt(m) for m in metadata]
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            tokens = tokenizer(batch_prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            out = model.get_text_features(**tokens)
            feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else (
                out.text_embeds if hasattr(out, "text_embeds") and out.text_embeds is not None else (
                    out if isinstance(out, torch.Tensor) else out[0]
                )
            )
            text_features = F.normalize(feats, p=2, dim=-1)
            all_embs.append(text_features.cpu().numpy())

    stacked = np.vstack(all_embs)
    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, stacked)
            print(f"[Text Features] Cached text embeddings to {cache_path.name}")
        except Exception as e:
            print(f"[Text Features] Cache save note: {e}")
    return stacked


# -----------------------------------------------------------------------------
# 2. PYTORCH MODEL: SPATIOTEMPORAL CROSS-MODAL PREDICTOR
# -----------------------------------------------------------------------------

class SpatiotemporalCrossModalPredictor(nn.Module):
    """
    Optimal Spatiotemporal Cross-Modal Alignment Predictor (30.3M Parameters):
    - 4-layer Transformer Encoder, 12 attention heads, d_model=768, feedforward=3072.
    - Matches CLIP ViT-L latent feature capacity.
    - Bridges frozen visual representations with frozen text embeddings in a joint metric space.
    """
    def __init__(
        self, 
        visual_dim: int = 512, 
        text_dim: int = 512, 
        shared_dim: int = 768, 
        num_layers: int = 4,
        nhead: int = 12,
        dim_feedforward: int = 3072
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_visual(self, v_feats: torch.Tensor) -> torch.Tensor:
        h = self.visual_proj(v_feats)
        if h.dim() == 2:
            h = h.unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        return F.normalize(h, p=2, dim=-1)

    def encode_text(self, t_feats: torch.Tensor) -> torch.Tensor:
        h = self.text_proj(t_feats)
        return F.normalize(h, p=2, dim=-1)

    def forward(self, v_feats: torch.Tensor, t_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_v = self.encode_visual(v_feats)
        z_t = self.encode_text(t_feats)
        return z_v, z_t


class BiDirectionalInfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss with learnable temperature scaling."""
    def __init__(self, initial_temp: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temp))

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        sim_matrix = torch.matmul(z_v, z_t.t()) * scale
        labels = torch.arange(len(z_v), device=z_v.device)
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0


# -----------------------------------------------------------------------------
# 3. DATA SPLITTING & FEATURE DATASET
# -----------------------------------------------------------------------------

class SurveillanceFeatureDataset(Dataset):
    """PyTorch Dataset yielding paired visual and textual features with metadata."""
    def __init__(self, visual_embs: torch.Tensor, text_embs: torch.Tensor, metadata: List[Dict[str, Any]]):
        self.visual = visual_embs
        self.text = text_embs
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.visual)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.visual[idx], self.text[idx], self.metadata[idx]


def surveillance_collate_fn(batch):
    """Collates feature tensors and keeps metadata dictionaries as a python list."""
    v_list = [item[0] for item in batch]
    t_list = [item[1] for item in batch]
    m_list = [item[2] for item in batch]
    return torch.stack(v_list), torch.stack(t_list), m_list


def load_all_datasets_data(workspace_root: Optional[Path] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Loads visual embeddings and metadata across RLVS, UCF-ARG, TinyVIRAT, and UCF-Crime."""
    root = workspace_root or WORKSPACE_ROOT
    emb_dir = root / "implementation" / "results" / "embeddings"
    idx_dir = root / "implementation" / "results" / "indices"

    all_embs = []
    all_meta = []

    for ds in ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime"]:
        emb_file = emb_dir / f"clip_{ds}_embeddings.npy"
        meta_file = idx_dir / f"clip_{ds}.meta.json"
        
        if emb_file.exists() and meta_file.exists():
            arr = np.load(emb_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", [])
            all_embs.append(arr)
            all_meta.extend(meta)
            print(f"[Dataset Loader] Loaded {ds.upper()}: {len(arr)} clips, dim {arr.shape[1]}")

    if not all_embs:
        raise FileNotFoundError("No embeddings found! Run extraction first.")

    stacked_embs = np.vstack(all_embs)
    return stacked_embs, all_meta


def create_stratified_splits(
    metadata: List[Dict[str, Any]], 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Creates stratified train/val/test splits based on dataset and action label."""
    np.random.seed(seed)
    from collections import defaultdict
    groups = defaultdict(list)
    
    for idx, item in enumerate(metadata):
        key = (item.get("dataset", "unknown"), str(item.get("label", "unknown")).lower())
        groups[key].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for key, idxs in groups.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train:n_train + n_val])
        test_indices.extend(idxs[n_train + n_val:])

    return train_indices, val_indices, test_indices


# -----------------------------------------------------------------------------
# 4. CCTV RETRIEVAL ENGINE WITH VIEWPOINT FILTERING & QUERY ROUTING
# -----------------------------------------------------------------------------

class CCTVRetrievalEngine:
    """
    Production-grade CCTV Retrieval Engine:
    - Indexes RLVS, UCF-ARG, TinyVIRAT, and UCF-Crime
    - Supports trained SpatiotemporalPredictor projections
    - Viewpoint-aware query routing (aerial vs ground vs rooftop)
    - Semantic query expansion (cars, vehicles, objects, violence, crimes)
    """
    def __init__(self, workspace_root: Optional[Path] = None, predictor_checkpoint: Optional[str] = None):
        self.workspace_root = workspace_root or BASE_DIR.parent
        self.indices: Dict[str, Any] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._clip_model = None
        self._clip_tokenizer = None
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_all_indices()
        if predictor_checkpoint and os.path.exists(predictor_checkpoint):
            self.load_predictor(predictor_checkpoint)

    def _load_all_indices(self):
        import faiss
        idx_dir = self.workspace_root / "implementation" / "results" / "indices"
        for ds in ["rlvs", "ucf_arg", "tinyvirat", "ucf_crime", "unified"]:
            f_idx = idx_dir / f"clip_{ds}.faiss"
            f_meta = idx_dir / f"clip_{ds}.meta.json"
            if f_idx.exists() and f_meta.exists():
                try:
                    index = faiss.read_index(str(f_idx))
                    with open(f_meta, "r", encoding="utf-8") as f:
                        meta = json.load(f).get("metadata", [])
                    self.indices[ds] = index
                    self.metadata[ds] = meta
                    print(f"[Retrieval Engine] Loaded '{ds}' index: {index.ntotal} vectors.")
                except Exception as e:
                    print(f"[Retrieval Engine] Warning loading {ds}: {e}")

    def load_predictor(self, checkpoint_path: str):
        print(f"[Retrieval Engine] Loading trained SpatiotemporalPredictor from {checkpoint_path}...")
        self.predictor = SpatiotemporalCrossModalPredictor().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()

        # Re-project all FAISS index vectors into the joint spatiotemporal space
        import faiss
        for ds, raw_idx in list(self.indices.items()):
            raw_arr = raw_idx.reconstruct_n(0, raw_idx.ntotal)
            with torch.no_grad():
                v_t = torch.from_numpy(raw_arr).float().to(self.device)
                z_v = self.predictor.encode_visual(v_t).cpu().numpy().astype(np.float32)
            new_idx = faiss.IndexFlatIP(z_v.shape[1])
            new_idx.add(z_v)
            self.indices[ds] = new_idx
        print(f"[Retrieval Engine] Successfully aligned {len(self.indices)} FAISS indices in joint latent space!")

    def _get_clip_text_encoder(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPTokenizer
            model_id = "openai/clip-vit-base-patch32"
            print(f"[Retrieval Engine] Initializing CLIP Text Encoder on {self.device}...")
            self._clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
            self._clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
            self._clip_model.eval()
        return self._clip_model, self._clip_tokenizer

    def encode_text_query(self, query_text: str) -> np.ndarray:
        model, tokenizer = self._get_clip_text_encoder()
        inputs = tokenizer([query_text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            t_feats = out.pooler_output if hasattr(out, "pooler_output") else (out.text_embeds if hasattr(out, "text_embeds") else out)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            
            # If trained predictor is active, project through joint spatiotemporal space
            if self.predictor is not None:
                t_feats = self.predictor.encode_text(t_feats.float())

            emb = t_feats.cpu().numpy().astype(np.float32)
        return emb

    def query(
        self, 
        query_text: str, 
        dataset: str = "unified", 
        viewpoint_filter: Optional[str] = None, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic CCTV retrieval with query expansion and viewpoint awareness.
        """
        q_lower = query_text.lower()

        # Semantic Query Routing & Intent Analysis
        is_vehicle = any(w in q_lower for w in ["car", "vehicle", "automobile", "truck", "driving", "turning", "parking", "trunk"])
        is_object = any(w in q_lower for w in ["carry", "box", "package", "load", "pull", "push", "tool", "cargo", "equipment"])
        is_aerial = any(w in q_lower for w in ["aerial", "drone", "overhead", "high angle"])
        is_ground = any(w in q_lower for w in ["ground", "street level", "pedestrian level"])

        # Auto-detect dataset if set to auto
        target_ds = dataset
        if target_ds in ("auto", "unified") and "unified" in self.indices:
            target_ds = "unified"
        elif target_ds not in self.indices:
            target_ds = "unified" if "unified" in self.indices else list(self.indices.keys())[0]

        # Automatic viewpoint filtering if requested in query text
        if viewpoint_filter is None:
            if is_aerial:
                viewpoint_filter = "aerial"
            elif is_ground:
                viewpoint_filter = "ground"

        is_violence = any(w in q_lower for w in ["fight", "punch", "violence", "brawl", "assault", "hit", "attack"])
        is_crime = any(w in q_lower for w in [
            "robbery", "burglary", "shoplift", "steal", "theft", "arson", "fire", 
            "accident", "crash", "collision", "shooting", "gun", "explosion", "arrest", "abuse", "vandalism"
        ])

        # Expand query text with descriptive surveillance semantics for enhanced CLIP matching
        expanded_query = query_text
        if is_vehicle:
            expanded_query = f"{query_text}, a car, vehicle, or automobile driving in surveillance"
        elif is_object:
            expanded_query = f"{query_text}, a person carrying a box, container, bag, or heavy object while moving"
        elif is_violence:
            expanded_query = f"{query_text}, violent street fighting, punching, physical brawl, or assault"
        elif is_crime:
            expanded_query = f"{query_text}, a surveillance camera capturing real-world crime incident, emergency, or traffic accident"

        q_emb = self.encode_text_query(expanded_query)
        index = self.indices[target_ds]
        records = self.metadata[target_ds]

        # Search top 50 candidates to allow viewpoint/semantic filtering
        search_k = min(len(records), max(top_k * 5, 50))
        scores, indices = index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(records):
                continue
            rec = records[idx]
            meta = rec.get("metadata", {})
            vpoint = meta.get("viewpoint", "").lower()
            label = str(rec.get("label", "")).lower()

            # Apply viewpoint filter (prevents aerial clips on ground queries)
            if viewpoint_filter and vpoint and viewpoint_filter.lower() not in vpoint:
                continue

            # Prevent car queries from false-matching clapping/waving in unaligned zero-shot
            if is_vehicle and label in ("clapping", "waving"):
                continue

            all_labels = meta.get("all_labels", [rec.get("label", "unknown")])
            label_str = ", ".join(all_labels) if isinstance(all_labels, list) else str(all_labels)
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "path": rec.get("path", ""),
                "label": label_str,
                "primary_label": rec.get("label", "unknown"),
                "dataset": rec.get("dataset", target_ds),
                "metadata": meta
            })
            if len(results) >= top_k:
                break

        return results


# -----------------------------------------------------------------------------
# 5. KEYFRAME EXTRACTION & VISUALIZATION
# -----------------------------------------------------------------------------

def extract_video_keyframes(video_path: str, num_frames: int = 4) -> List[np.ndarray]:
    """Extracts evenly spaced RGB frames from video clip using OpenCV."""
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def visualize_retrieved_events(
    results: List[Dict[str, Any]], 
    query_text: str, 
    max_display: int = 3, 
    num_frames: int = 4,
    save_path: Optional[str] = None
):
    """Renders visual keyframe montages for top retrieved surveillance events."""
    items_to_show = results[:max_display]
    if not items_to_show:
        print("No matching items found.")
        return

    n_items = len(items_to_show)
    fig, axes = plt.subplots(n_items, num_frames, figsize=(num_frames * 3.5, n_items * 2.8))
    if n_items == 1:
        axes = np.expand_dims(axes, 0)
    if num_frames == 1:
        axes = np.expand_dims(axes, 1)

    fig.suptitle(f'Surveillance Query: "{query_text}"', fontsize=14, fontweight='bold', y=0.98)

    for i, item in enumerate(items_to_show):
        vpath = item.get("path", "")
        rank = item.get("rank", i + 1)
        score = item.get("score", 0.0)
        label = item.get("label", "Unknown")
        meta = item.get("metadata", {})
        ds = item.get("dataset", "").upper()
        vp = meta.get("viewpoint", meta.get("category", meta.get("class_dir", "")))

        frames = extract_video_keyframes(vpath, num_frames=num_frames)
        for j in range(num_frames):
            ax = axes[i, j]
            if j < len(frames):
                ax.imshow(frames[j])
            else:
                ax.text(0.5, 0.5, "Frame Unavailable", ha='center', va='center', color='gray')
                ax.set_facecolor('#F0F0F0')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Rank #{rank} | Sim: {score:.3f}\n[{ds}] {label} ({vp})", 
                             fontsize=9.5, fontweight='bold', loc='left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
