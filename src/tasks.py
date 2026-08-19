"""Task 1 (VLM) and Task 2 (classical + numbers-first LLM)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .classical import otsu_segment, region_table, table_to_text
from .config import LLM_DIR, REPRESENTATIVE_ID, TABLE_DIR, ensure_dirs
from .data import load_rgb, raw_image_path, split_of, to_grayscale_256
from .figures import fig_eda_samples, fig_intensity_histogram, fig_otsu_example
from .llm import generate
from .prompts import NAIVE_VLM_PROMPT, NUMBERS_FIRST_PROMPT, OPTIMISED_VLM_PROMPT


def run_task1(skip_llm: bool = False) -> Dict[str, Any]:
    """Preprocess note + EDA + naive vs engineered VLM + run-to-run variability."""
    ensure_dirs()
    fig_eda_samples()
    fig_intensity_histogram()

    image_id = REPRESENTATIVE_ID
    image_path = raw_image_path(split_of(image_id), image_id)
    result: Dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path),
        "naive_prompt": NAIVE_VLM_PROMPT,
        "optimised_prompt": OPTIMISED_VLM_PROMPT,
    }

    if skip_llm:
        result["skipped"] = True
        (TABLE_DIR / "task1_vlm.json").write_text(json.dumps(result, indent=2))
        return result

    naive = generate(
        NAIVE_VLM_PROMPT,
        image_path=image_path,
        temperature=0.2,
        force_json=False,
        cache_key="task1_naive",
    )
    optimised = generate(
        OPTIMISED_VLM_PROMPT,
        image_path=image_path,
        temperature=0.2,
        force_json=True,
        cache_key="task1_optimised",
    )
    repeats = []
    for i in range(3):
        repeats.append(
            generate(
                OPTIMISED_VLM_PROMPT,
                image_path=image_path,
                temperature=0.8,
                force_json=True,
                cache_key=f"task1_repeat_{i}",
            )
        )

    result["naive"] = {"raw": naive["raw"], "json": naive["json"]}
    result["optimised"] = {"raw": optimised["raw"], "json": optimised["json"]}
    result["repeats"] = [{"raw": r["raw"], "json": r["json"]} for r in repeats]
    result["repeats_identical"] = _all_equal([r["json"] for r in repeats])
    (TABLE_DIR / "task1_vlm.json").write_text(json.dumps(result, indent=2))
    return result


def run_task2(skip_llm: bool = False) -> Dict[str, Any]:
    """Otsu + regionprops on the Task-1 image; numbers-only LLM interpretation."""
    ensure_dirs()
    image_id = REPRESENTATIVE_ID
    gray = to_grayscale_256(load_rgb(image_id))
    mask = otsu_segment(gray)
    df = region_table(gray, mask)
    text, stats = table_to_text(df, image_id)
    fig_otsu_example(image_id)
    df.to_csv(TABLE_DIR / f"regionprops_{image_id}.csv", index=False)

    payload: Dict[str, Any] = {
        "image_id": image_id,
        "feature_text": text,
        "stats": {k: v for k, v in stats.items() if k != "image_id"},
        "n_rows": int(len(df)),
        "prompt": NUMBERS_FIRST_PROMPT,
    }

    if skip_llm:
        payload["skipped"] = True
        payload["llm_json"] = {
            "n_objects": stats["n_objects"],
            "density_class": "normal" if 15 <= stats["n_objects"] < 45 else "sparse",
            "shape_regularity": "mixed",
            "quality_flag": "ok",
        }
        payload["narrative"] = text
        (TABLE_DIR / "task2_numbers_first.json").write_text(json.dumps(payload, indent=2, default=str))
        return payload

    rec = generate(
        NUMBERS_FIRST_PROMPT + "\n\nFeature summary:\n" + text,
        image_path=None,
        temperature=0.2,
        force_json=True,
        cache_key="task2_numbers_first",
    )
    payload["raw"] = rec["raw"]
    payload["llm_json"] = rec["json"]
    parsed = rec.get("json") or {}
    payload["narrative"] = parsed.get("narrative") or _paragraph_after_json(rec["raw"])
    (TABLE_DIR / "task2_numbers_first.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def compare_vision_models(skip_llm: bool = False) -> Dict[str, Any]:
    """Extra credit: same engineered prompt on llama3.2-vision vs llava."""
    from .config import SECOND_VLM_MODEL, VLM_MODEL
    from .llm import OllamaError

    ensure_dirs()
    image_id = REPRESENTATIVE_ID
    image_path = raw_image_path(split_of(image_id), image_id)
    comparison: Dict[str, Any] = {
        "image_id": image_id,
        "prompt": OPTIMISED_VLM_PROMPT,
        "models": [VLM_MODEL, SECOND_VLM_MODEL],
    }
    if skip_llm:
        # Still reuse cached JSON if a previous run stored it.
        from .config import LLM_DIR
        cached_a = LLM_DIR / "task1_optimised.json"
        cached_b = LLM_DIR / "task1_optimised_llava.json"
        if cached_a.exists() or cached_b.exists():
            skip_llm = False
        else:
            comparison["skipped"] = True
            (TABLE_DIR / "vlm_model_comparison.json").write_text(json.dumps(comparison, indent=2))
            return comparison

    a = generate(
        OPTIMISED_VLM_PROMPT,
        image_path=image_path,
        model=VLM_MODEL,
        temperature=0.2,
        force_json=True,
        cache_key="task1_optimised",
    )
    try:
        b = generate(
            OPTIMISED_VLM_PROMPT,
            image_path=image_path,
            model=SECOND_VLM_MODEL,
            temperature=0.2,
            force_json=True,
            cache_key="task1_optimised_llava",
        )
    except OllamaError as exc:
        comparison["error"] = str(exc)
        comparison["llama3.2-vision"] = {"json": a.get("json"), "raw": a.get("raw")}
        (TABLE_DIR / "vlm_model_comparison.json").write_text(json.dumps(comparison, indent=2))
        return comparison

    comparison["llama3.2-vision"] = {"json": a.get("json"), "raw": a.get("raw"), "model": a.get("model")}
    comparison["llava"] = {"json": b.get("json"), "raw": b.get("raw"), "model": b.get("model")}
    comparison["json_identical"] = json.dumps(a.get("json"), sort_keys=True) == json.dumps(
        b.get("json"), sort_keys=True
    )
    (TABLE_DIR / "vlm_model_comparison.json").write_text(json.dumps(comparison, indent=2))
    return comparison


def _all_equal(objs) -> bool:
    if not objs:
        return True
    return all(json.dumps(o, sort_keys=True) == json.dumps(objs[0], sort_keys=True) for o in objs)


def _paragraph_after_json(raw: str) -> str:
    end = raw.rfind("}")
    if end == -1:
        return raw.strip()
    rest = raw[end + 1 :].strip()
    return rest if rest else raw.strip()
