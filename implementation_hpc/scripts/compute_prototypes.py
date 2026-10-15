import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def compute_prototypes():
    p_meta = Path("results/embeddings/vljepa_ucf_crime_meta.json")
    p_vljepa_embs = Path("results/embeddings/vljepa_ucf_crime_embeddings.npy")
    p_clip_embs = Path("results/embeddings/clip_ucf_crime_embeddings.npy")

    with open(p_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    v_embs = torch.from_numpy(np.load(p_vljepa_embs)).float()
    v_embs = F.normalize(v_embs, p=2, dim=-1)

    c_embs = torch.from_numpy(np.load(p_clip_embs)).float()
    c_embs = F.normalize(c_embs, p=2, dim=-1)

    labels = [m["label"] for m in meta]
    unique_labels = sorted(list(set(labels)))

    vljepa_prototypes = {}
    clip_prototypes = {}

    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        proto_v = F.normalize(v_embs[idx].mean(dim=0), p=2, dim=-1)
        proto_c = F.normalize(c_embs[idx].mean(dim=0), p=2, dim=-1)
        vljepa_prototypes[lbl] = proto_v
        clip_prototypes[lbl] = proto_c
        print(f"Computed prototype for {lbl:15s} from {len(idx)} clips.")

    save_data = {
        "vljepa_prototypes": vljepa_prototypes,
        "clip_prototypes": clip_prototypes,
        "classes": unique_labels,
    }

    out_file = Path("results/ucf_crime_class_prototypes.pt")
    torch.save(save_data, out_file)
    print(f"Saved class prototypes to {out_file}")

if __name__ == "__main__":
    compute_prototypes()
