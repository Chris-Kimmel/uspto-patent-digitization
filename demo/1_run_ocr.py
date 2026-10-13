# Demo, step 1: find the title page of each patent PDF, detect its layout,
# isolate the bibliographic block, and OCR it.
#
# Runs the production OCR script (pipeline/ocr_title_page.py) unchanged on the
# example patents in demo/patents/. Two production defaults are switched for
# the demo (debug images on, intermediate files kept), so you can see what the
# pipeline sees. At 2M-patent scale both are off.
#
# Requires: the packages in requirements.txt, Tesseract (used only to find the
# title page), and the OCR/layout models (one-time: python pipeline/setup_models.py)
#
# Run:    python demo/1_run_ocr.py
# Output: demo/output/<patent>/
#           1_page_with_detected_layout.png   layout analysis (boxes) on the title page
#           2_title_block_for_ocr.png         the block that is OCR'd, rest masked
#           3_ocr_text.txt                    OCR text, the input to step 2

import os
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DEMO_DIR.parent / "pipeline"
OUTPUT_DIR = DEMO_DIR / "output"
WORKING_DIR = OUTPUT_DIR / "working_files"   # the pipeline's own run folders and logs

# The pipeline finds its models relative to the working directory, as in
# production. A plain import (not importlib) keeps the module picklable for the
# worker processes, which re-import it by name.
os.chdir(PIPELINE_DIR)
sys.path.insert(0, str(PIPELINE_DIR))

import ocr_title_page as pipeline

_original_worker = pipeline._worker_process_pdf


def _worker_with_debug_images(pdf_path, run_root, pdf_extraction_config):
    # save_debug_img is read inside each worker process, so it is switched on here
    cfg = dict(pdf_extraction_config)
    cfg["save_debug_img"] = True
    return _original_worker(pdf_path, run_root, cfg)


pipeline._worker_process_pdf = _worker_with_debug_images
pipeline._cleanup_aggregated_patent_folders = lambda *a, **kw: None  # keep per-patent images


def collect_results(run_root):
    """Copy the interesting files of each patent into demo/output/<patent>/."""
    text_dir = Path(run_root).parent.parent / "STEP_1_2_RESULTS" / "Step1_2_Output"
    for pdf in sorted((DEMO_DIR / "patents").glob("*.pdf")):
        pn = pdf.stem
        images = Path(run_root) / pn / "cropped_masked_images"
        out = OUTPUT_DIR / pn
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(images / f"{pn}_debug_boxes.png", out / "1_page_with_detected_layout.png")
        shutil.copy(images / f"{pn}.png", out / "2_title_block_for_ocr.png")
        shutil.copy(text_dir / f"{pn}.txt", out / "3_ocr_text.txt")
        print(f"  {pn} -> {out.relative_to(DEMO_DIR.parent)}/")


if __name__ == "__main__":
    shutil.rmtree(WORKING_DIR, ignore_errors=True)  # fresh run (the pipeline skips already-processed PDFs)
    result = pipeline.main()
    print("\nCollecting results:")
    collect_results(result["run_root"])
