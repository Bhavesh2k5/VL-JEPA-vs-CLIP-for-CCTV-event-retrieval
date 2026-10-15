import json
from pathlib import Path

with open("data/test_prompts_500.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

md_lines = [
    "# Comprehensive Catalog: 500 Fine-Grained Surveillance Benchmark Prompts",
    "",
    "This catalog contains **500 curated, realistic, fine-grained surveillance queries** specifically engineered to stress-test **VL-JEPA** vs. **CLIP** across the 4 benchmark datasets.",
    "",
    "Each prompt is calibrated to evaluate:",
    "1. **Fine-grained subject specificity** (e.g. gender, clothing, roles, victim vs. aggressor).",
    "2. **Spatiotemporal dynamic actions** (e.g. punching, tackling, carrying, throwing, digging, breaking in).",
    "3. **Geometric perspective & camera angle** (aerial drone, elevated rooftop, ground CCTV).",
    "4. **Non-violent controls & subtle boundary conditions** (calm walking, hugging, friendly sporting).",
    "",
    "---",
    "",
    "## Dataset Summary",
    "",
    "| Dataset | Corpus Scope | Prompt Count | Primary Focus |",
    "| :--- | :--- | :---: | :--- |",
    "| **RLVS** | 2,000 clips | **150** | Violence vs. Non-Violence, Gender/Age Demographics, Fighting Dynamics |",
    "| **UCF-Crime** | 1,000 clips | **150** | 14 Real-world Anomaly Classes (Robbery, Shoplifting, Arson, Accidents, Normal) |",
    "| **UCF-ARG** | 1,410 clips | **100** | Cross-Camera Viewpoint Invariance (Aerial, Rooftop, Ground) |",
    "| **TinyVIRAT-v2** | 3,000 clips | **100** | Complex Person-Vehicle Interactions & Spatiotemporal Progression |",
    "| **Total** | **7,410 clips** | **500** | Full Surveillance Intelligence Benchmark Suite |",
    "",
    "---",
    ""
]

current_ds = ""
for p in prompts:
    ds = p["dataset"]
    if ds != current_ds:
        if current_ds != "":
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        current_ds = ds
        md_lines.append(f"## {current_ds} Benchmark Prompts")
        md_lines.append("")
        md_lines.append("| ID | Category | Surveillance Query Prompt | Specificity | Target Context |")
        md_lines.append("| :---: | :--- | :--- | :---: | :--- |")
    
    md_lines.append(f"| **{p['id']}** | {p['category']} | `{p['query']}` | {p['specificity']} | {p['target_context']} |")

md_lines.append("")

target_file = Path("C:/Users/HPCLAB/.gemini/antigravity/brain/f4dbd9c2-0474-4397-8ee4-ede428b62eed/surveillance_test_prompts_500.md")
with open(target_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Wrote {len(md_lines)} lines to {target_file}")
