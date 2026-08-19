#!/usr/bin/env python3
"""End-to-end .

Usage
-----
    source .venv/bin/activate
    python run_all.py              # full pipeline (needs Ollama + llama3.2-vision)
    python run_all.py --skip-llm   # train U-Net / figures only
    python run_all.py --epochs 20

All numerical values and figures used in the report are written under outputs/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python run_all.py` from the repo root without installing the package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    EPOCHS,
    LOSSES,
    PRIMARY_LOSS,
    TABLE_DIR,
    ensure_dirs,
)
from src.data import list_image_ids, load_mask, load_rgb, preprocess_split, to_grayscale_256  # noqa: E402
from src.classical import otsu_segment  # noqa: E402
from src.train_eval import (  # noqa: E402
    collect_val_triplets,
    get_device,
    load_trained,
    otsu_vs_unet_on_val,
    train_one_loss,
)
from src.tasks import run_task1, run_task2, compare_vision_models  # noqa: E402
from src.pipeline import run_test_pipeline  # noqa: E402
from src.robustness import trace_corruption  # noqa: E402
from src.medsam_cmp import compare_medsam_on_val  # noqa: E402
from src.figures import fig_loss_curves, fig_otsu_vs_unet, fig_unet_panels, fig_medsam_panels  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assignment 3 hybrid nuclei pipeline")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="U-Net epochs per loss")
    p.add_argument("--skip-llm", action="store_true", help="Skip Ollama calls (uses cached JSON if present)")
    p.add_argument("--skip-train", action="store_true", help="Reuse saved U-Net checkpoints")
    p.add_argument("--skip-medsam", action="store_true", help="Skip MedSAM download/inference")
    p.add_argument("--extras-only", action="store_true", help="Only run VLM-vs-VLM and MedSAM extras")
    p.add_argument("--primary-loss", default=PRIMARY_LOSS, choices=list(LOSSES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    device = get_device()
    print(f"Device: {device}")

    if args.extras_only:
        model = load_trained(args.primary_loss, device=device)
        _run_extras(model, device, skip_llm=args.skip_llm, skip_medsam=args.skip_medsam)
        return

    # --- Task 1: grayscale 256×256 + EDA + VLM --------------------------------
    print("\n=== Task 1: preprocess + EDA + VLM ===")
    for split in ("train", "val", "test"):
        paths = preprocess_split(split)
        print(f"  processed {len(paths)} {split} images → outputs/processed/{split}/")
    t1 = run_task1(skip_llm=args.skip_llm)
    print("  representative image:", t1["image_id"])
    if t1.get("optimised"):
        print("  optimised JSON:", t1["optimised"]["json"])

    # --- Task 2: Otsu + regionprops + numbers-first LLM -----------------------
    print("\n=== Task 2: classical features + numbers-first LLM ===")
    t2 = run_task2(skip_llm=args.skip_llm)
    print("  n_objects (regionprops):", t2["stats"]["n_objects"])
    print("  LLM JSON:", t2.get("llm_json"))

    # --- Task 3: U-Net training + loss ablation -------------------------------
    print("\n=== Task 3: U-Net training / evaluation ===")
    histories = {}
    ablation = []
    if not args.skip_train:
        for loss_name in LOSSES:
            print(f"\n--- training loss={loss_name} ---")
            result = train_one_loss(loss_name, epochs=args.epochs, device=device)
            histories[loss_name] = result["history"]
            ablation.append(
                {
                    "loss": loss_name,
                    "best_epoch": result["best_epoch"],
                    "val_dice": result["val_dice"],
                    "val_iou": result["val_iou"],
                }
            )
    else:
        for loss_name in LOSSES:
            path = TABLE_DIR / f"unet_{loss_name}_metrics.json"
            result = json.loads(path.read_text())
            histories[loss_name] = result["history"]
            ablation.append(
                {
                    "loss": loss_name,
                    "best_epoch": result["best_epoch"],
                    "val_dice": result["val_dice"],
                    "val_iou": result["val_iou"],
                }
            )

    fig_loss_curves(histories)
    (TABLE_DIR / "loss_ablation.json").write_text(json.dumps(ablation, indent=2))

    model = load_trained(args.primary_loss, device=device)
    triplets = collect_val_triplets(model, device, n=3)
    fig_unet_panels(triplets)

    cmp_rows = otsu_vs_unet_on_val(model, device)
    (TABLE_DIR / "otsu_vs_unet.json").write_text(json.dumps(cmp_rows, indent=2))
    otsu_mean = sum(r["otsu_dice"] for r in cmp_rows) / len(cmp_rows)
    unet_mean = sum(r["unet_dice"] for r in cmp_rows) / len(cmp_rows)
    print(f"  mean val Dice  Otsu={otsu_mean:.4f}  U-Net={unet_mean:.4f}")

    # one image where each method is better (fall back to extremes of the gap)
    otsu_win = max(cmp_rows, key=lambda r: r["otsu_dice"] - r["unet_dice"])
    unet_win = max(cmp_rows, key=lambda r: r["unet_dice"] - r["otsu_dice"])
    examples = []
    for row, title in ((otsu_win, "smallest U-Net Dice margin"), (unet_win, "largest U-Net Dice margin")):
        iid = row["image_id"]
        gray = to_grayscale_256(load_rgb(iid))
        from src.train_eval import predict_mask

        examples.append(
            {
                "image_id": iid,
                "gray": gray,
                "gt": load_mask(iid),
                "otsu": otsu_segment(gray),
                "unet": predict_mask(model, gray, device),
                "title": title,
            }
        )
    fig_otsu_vs_unet(cmp_rows, examples)

    # --- Task 4: hybrid pipeline on unseen test images ------------------------
    print("\n=== Task 4: hybrid pipeline on test ===")
    df = run_test_pipeline(model, device, skip_llm=args.skip_llm)
    print(df.to_string(index=False))

    # --- Extra credit: robustness --------------------------------------------
    print("\n=== Extra: robustness (heavy blur) ===")
    rob = trace_corruption(model, device, skip_llm=args.skip_llm)
    print("  earliest detectable stage:", rob["earliest_detectable_stage"])
    print("  n_objects clean/corr:", rob["mask_delta"]["n_objects_clean"], rob["mask_delta"]["n_objects_corr"])

    _run_extras(model, device, skip_llm=args.skip_llm, skip_medsam=args.skip_medsam)


def _run_extras(model, device, skip_llm: bool, skip_medsam: bool) -> None:
    print("\n=== Extra: two vision models on the same prompt ===")
    cmp = compare_vision_models(skip_llm=skip_llm)
    print("  llama3.2-vision JSON:", (cmp.get("llama3.2-vision") or {}).get("json"))
    print("  llava JSON:", (cmp.get("llava") or {}).get("json"))
    if cmp.get("error"):
        print("  llava error:", cmp["error"])

    if skip_medsam:
        print("\n=== Extra: MedSAM skipped ===")
        return
    print("\n=== Extra: MedSAM vs U-Net (validation) ===")
    result = compare_medsam_on_val(model, device)
    s = result["summary"]
    print(
        f"  mean Dice  U-Net={s['mean_unet_dice']:.4f}  "
        f"MedSAM-fullbox={s['mean_medsam_fullbox_dice']:.4f}  "
        f"MedSAM-otsubox={s['mean_medsam_otsubox_dice']:.4f}"
    )
    if result["panels"]:
        fig_medsam_panels(result["panels"])


if __name__ == "__main__":
    main()
