"""Task 4 hybrid pipeline: U-Net mask → regionprops → LLM JSON → narrative → CSV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .classical import density_class, region_table, summarise_table, table_to_text
from .config import TABLE_DIR, ensure_dirs
from .data import list_image_ids, load_rgb, to_grayscale_256
from .llm import generate
from .prompts import HYBRID_PROMPT
from .train_eval import predict_mask


def features_from_mask(image_id: str, gray, mask) -> Dict[str, Any]:
    df = region_table(gray, mask)
    text, stats = table_to_text(df, image_id)
    stats["density_class_rule"] = density_class(int(stats["n_objects"]), float(stats["area_fraction"]))
    stats["feature_text"] = text
    stats["region_table"] = df
    return stats


def llm_record_from_features(stats: Dict[str, Any], cache_key: str) -> Dict[str, Any]:
    """Ask the local LLM for a structured JSON record + narrative."""
    user = (
        HYBRID_PROMPT
        + "\n\nInput statistics:\n"
        + stats["feature_text"]
        + f"\nimage_id={stats['image_id']}"
        + f"\nmean_area={stats['mean_area']:.3f}"
        + f"\nn_objects={stats['n_objects']}"
    )
    rec = generate(user, image_path=None, temperature=0.1, force_json=True, cache_key=cache_key)
    parsed = rec.get("json") or {}
    # Keep computed numbers as the source of truth; LLM may only label/narrate.
    parsed["image_id"] = stats["image_id"]
    parsed["n_objects"] = int(stats["n_objects"])
    parsed["mean_area"] = round(float(stats["mean_area"]), 3)
    parsed["llm_density_class"] = parsed.get("density_class")
    parsed["density_class"] = stats["density_class_rule"]
    if parsed.get("quality_flag") not in {"ok", "low_contrast", "fragmented", "uncertain"}:
        parsed["quality_flag"] = "ok" if stats["n_objects"] > 0 else "uncertain"
    if not parsed.get("narrative"):
        parsed["narrative"] = rec.get("raw", "")[:800]
    parsed["_llm_raw"] = rec.get("raw")
    return parsed


def run_test_pipeline(model, device, skip_llm: bool = False) -> pd.DataFrame:
    """Run the full hybrid pipeline on the unseen test split."""
    ensure_dirs()
    rows: List[Dict[str, Any]] = []
    narratives: List[Dict[str, Any]] = []
    for image_id in list_image_ids("test"):
        gray = to_grayscale_256(load_rgb(image_id))
        mask = predict_mask(model, gray, device)
        stats = features_from_mask(image_id, gray, mask)
        if skip_llm:
            parsed = {
                "image_id": image_id,
                "n_objects": int(stats["n_objects"]),
                "mean_area": round(float(stats["mean_area"]), 3),
                "density_class": stats["density_class_rule"],
                "quality_flag": "ok" if stats["n_objects"] else "uncertain",
                "narrative": (
                    f"{image_id} contains {stats['n_objects']} segmented objects "
                    f"with mean area {stats['mean_area']:.1f} px "
                    f"(density {stats['density_class_rule']})."
                ),
            }
        else:
            parsed = llm_record_from_features(stats, cache_key=f"hybrid_{image_id}")
        row = {k: parsed[k] for k in ("image_id", "n_objects", "mean_area", "density_class", "quality_flag")}
        row["mean_eccentricity"] = round(float(stats["mean_eccentricity"]), 4)
        row["mean_solidity"] = round(float(stats["mean_solidity"]), 4)
        row["area_fraction"] = round(float(stats["area_fraction"]), 5)
        rows.append(row)
        narratives.append({"image_id": image_id, "narrative": parsed.get("narrative", ""), "json": parsed})
        print(f"[pipeline] {image_id}: n={row['n_objects']} density={row['density_class']}")

    df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "test_pipeline_records.csv"
    df.to_csv(csv_path, index=False)
    (TABLE_DIR / "test_pipeline_narratives.json").write_text(json.dumps(narratives, indent=2))
    return df
