import hashlib
import json
import os
from pathlib import Path

def get_file_hash_and_size(filepath):
    try:
        p = Path(filepath)
        if not p.exists():
            return None, 0
        size = os.path.getsize(p)
        hasher = hashlib.md5()
        with open(p, "rb") as f:
            chunk = f.read(65536) # first 64KB
            hasher.update(chunk)
        return hasher.hexdigest(), size
    except Exception:
        return None, 0

def build_registry():
    registry = {
        "by_filename": {},
        "by_hash_size": {},
    }

    # 1. UCF-Crime
    p_crime = Path("results/embeddings/vljepa_ucf_crime_meta.json")
    if p_crime.exists():
        with open(p_crime, "r", encoding="utf-8") as f:
            crime_meta = json.load(f)
        for item in crime_meta:
            lbl = item.get("label", "normal")
            vpath = item.get("path", "")
            fname = Path(vpath).name
            entry = {
                "key": lbl,
                "domain": "UCF-Crime (Surveillance Anomalies)",
                "dataset": "ucf_crime",
                "filename": fname,
                "original_path": vpath,
            }
            registry["by_filename"][fname.lower()] = entry
            h, s = get_file_hash_and_size(vpath)
            if h and s:
                registry["by_hash_size"][f"{s}_{h}"] = entry

    # 2. RLVS
    p_rlvs = Path("results/embeddings/clip_rlvs_meta.json")
    if p_rlvs.exists():
        with open(p_rlvs, "r", encoding="utf-8") as f:
            rlvs_meta = json.load(f)
        for item in rlvs_meta:
            lbl = item.get("label", "non_violence")
            if lbl.lower() in ("violence", "v"):
                k = "violence"
            else:
                k = "non_violence"
            vpath = item.get("path", "")
            fname = Path(vpath).name
            entry = {
                "key": k,
                "domain": "RLVS (Violence Recognition)",
                "dataset": "rlvs",
                "filename": fname,
                "original_path": vpath,
            }
            registry["by_filename"][fname.lower()] = entry
            h, s = get_file_hash_and_size(vpath)
            if h and s:
                registry["by_hash_size"][f"{s}_{h}"] = entry

    # 3. UCF-ARG
    p_arg = Path("results/embeddings/clip_ucf_arg_meta.json")
    if p_arg.exists():
        with open(p_arg, "r", encoding="utf-8") as f:
            arg_meta = json.load(f)
        for item in arg_meta:
            lbl = item.get("label", "walking")
            vpath = item.get("path", "")
            fname = Path(vpath).name
            entry = {
                "key": lbl,
                "domain": "UCF-ARG (Cross-Camera Actions)",
                "dataset": "ucf_arg",
                "filename": fname,
                "original_path": vpath,
            }
            registry["by_filename"][fname.lower()] = entry
            h, s = get_file_hash_and_size(vpath)
            if h and s:
                registry["by_hash_size"][f"{s}_{h}"] = entry

    # 4. TinyVIRAT
    p_virat = Path("results/embeddings/clip_tinyvirat_meta.json")
    if p_virat.exists():
        with open(p_virat, "r", encoding="utf-8") as f:
            virat_meta = json.load(f)
        for item in virat_meta:
            lbl = item.get("label", "activity_standing")
            vpath = item.get("path", "")
            fname = Path(vpath).name
            entry = {
                "key": lbl,
                "domain": "TinyVIRAT (Facility Surveillance)",
                "dataset": "tinyvirat",
                "filename": fname,
                "original_path": vpath,
            }
            # For virat, filename might be 000000.mp4, so include parent folder
            parent_name = Path(vpath).parent.name
            registry["by_filename"][f"{parent_name}_{fname}".lower()] = entry
            h, s = get_file_hash_and_size(vpath)
            if h and s:
                registry["by_hash_size"][f"{s}_{h}"] = entry

    out_path = Path("results/benchmark_ground_truth_registry.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry built successfully! Indexed {len(registry['by_filename'])} filenames, {len(registry['by_hash_size'])} hashes.")

if __name__ == "__main__":
    build_registry()
