'''
This is for steps 1 and 2 of the pipeline:
Step1: Multipage .pdf -> singlepage .pdf
Step2: singlepage .pdf -> relevant section .txt (Masking/cropping logic and OCR with DocTR) 
(Step 3, LLM extraction of structured JSON from the .txt, is shown in demo/2_run_extraction.py)

This is the script used for the full 1945-1980 production run; only the
configuration block below (paths, worker count) was changed to point at demo/.
'''

print("importing dependencies...")

import os

# --- OFFLINE + LOCAL MODEL SETUP --------------------------------------------

# 1. Enable OFFLINE mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# !! CRITICAL CHANGE: REMOVE os.environ['HF_HOME'] = "models_backup" !!
# By NOT setting HF_HOME, the script defaults to the standard system cache location 
# where 'setup_models.py' downloaded YOLOX.

# Set the custom cache for Doctr models (this is still necessary for Doctr)
os.environ["DOCTR_CACHE_DIR"] = "models_backup/doctr"

# 2. Paths to your local models
BASE_MODEL_DIR = "models_backup"
DOCTR_DIR = os.path.join(BASE_MODEL_DIR, "doctr")

# 3. Verify that the Doctr models exist
doctr_files = ["fast_base-688a8b34.pt", "crnn_vgg16_bn-0417f351.pt"]
fpath = os.path.join(DOCTR_DIR, "models", doctr_files[0]) # Check one file
if not os.path.exists(fpath):
    raise FileNotFoundError(f"Doctr model not found at {fpath}. Run setup_models.py first.")
else:
    print(f"✅ Found Doctr models in {DOCTR_DIR}")

# --- Model Loading (Offline from default system cache) ---

from unstructured_inference.inference.layout import get_model
from langchain_unstructured import UnstructuredLoader

# get_model("yolox") will now look in the DEFAULT system cache.
# Since HF_HUB_OFFLINE is "1" and the model was previously downloaded, it will load successfully.
yolox_model = get_model("yolox")

print("✅ YOLOX model loaded ")

# 4. Load YOLOX model locally
#yolox_model_name = "yolox"

# Core / standard library
import re
import tarfile
import tempfile
import shutil
import copy
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from time import time
from dotenv import load_dotenv
import shutil
#print("Core/Standard IMPORTS DONE")

# Typing
from typing import List
# Import TypedDict and NotRequired from typing_extensions for Python 3.10 compatibility
from typing_extensions import TypedDict, NotRequired 
#print("Typing IMPORTS DONE")

# Third-party libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from tqdm import tqdm
from PIL import Image as PilImage
from PIL import ImageFile
from PIL import ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytesseract
from fpdf import FPDF

from pdf2image import convert_from_bytes, convert_from_path
from pypdf import PdfReader, PdfWriter # ppdf instead of PyPDF2

from doctr.models import ocr_predictor

# LangChain / LangGraph
from langgraph.graph import StateGraph, START, END
#print("LangChain/LangGraph IMPORTS DONE")

# load from .env into environment
load_dotenv()

# for DocTR
# Setup model for OCR: Load pretrained OCR model
predictor = ocr_predictor(pretrained=True)
#print("predictor done")

# --- DEFAULT CONFIGURATION CONSTANTS ---

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

DEFAULT_INPUT_PATH = str(_DEMO_DIR / "patents") # input a folder containing the .tar files or .pdf

DEFAULT_OUTPUT_ROOT = str(_DEMO_DIR / "output" / "working_files" / "runs")
# Define a central aggregation folder for text results. This will be created next to the 'output_root' folder
OCR_MASTER_FOLDER = os.path.join(os.path.dirname(DEFAULT_OUTPUT_ROOT), "Step1_2_Output")
DEFAULT_MASTER_LOG = str(_DEMO_DIR / "output" / "working_files" / "master_log_step1_2_processed.txt")

# Number of worker processes for parallel step1+step2 processing.
# CPU-bound (no GPU code). The production run used 48 worker processes on a
# 64-core server; a laptop-sized default is used here.
MAX_WORKERS = 4

# -----------------------------------------------------------------------------


class PipelineState(TypedDict):
    # --- Step 1: Multipage PDF -> singlepage PDFs ---
    pdf_extraction: NotRequired[dict]  # contains tar/pdf directory info, output folder, configurables etc.
    # --- Step 2: OCR ---
    ocr: NotRequired[dict]  # contains predictor, results, etc.



# ==========================================================================================================
# ==========================================================================================================
# ========================================                          ========================================
# ======================================== Helper/Utility functions ========================================
# ========================================                          ========================================
# ==========================================================================================================
# ==========================================================================================================



# --- For finding relevant pdf page from multipage.pdf

# Keywords and DPI config
KEYWORDS = ["united", "states", "office", "patented", "patonted"]
DPI_FAST = 100

def find_relevant_page_index_fast(pdf_bytes, keywords=KEYWORDS, dpi=DPI_FAST):
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    for i, one_image in enumerate(images):
        text = pytesseract.image_to_string(one_image.convert("L"), config="--psm 6").lower()
        print(f"Page {i}: extracted text (first 100 chars): '{text[:100]}'")
        if any(k in text for k in keywords):
            print("KEYWORD FOUND:", [ k for k in keywords if k in text ])
            return i
    print("NO KEYWORD FOUND ON ANY PAGE")
    return None

# --- Utility: single page extraction
def single_page_pdf_to_tempfile(pdf_path: str, page_index: int = 0) -> str:
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    with open(tmp.name, "wb") as f:
        writer.write(f)
    return tmp.name

# --- Utility: deskewing with Hough transform
def deskew_with_hough(pil_img, debug=False, rot_threshold=1.3,
                        min_rel_len=0.20, max_angle=4.5, margin_frac=0.05):
    """
    Deskew using probabilistic Hough transform (HoughLinesP).
    - pil_img: input PIL image
    - debug: if set to True, you can plot detected lines and deskew process info (good to see how it works)
    - rot_threshold: degrees; only rotate if |angle| >= this angle. (Dont want to deskew unnecessarily, but if pdf is too skewed (~ > 1.3 degrees), it can mess up the masking process)
    - min_rel_len: minimum line length as fraction of image width to consider when determening skew (to filter out short lines/noise)
    - max_angle: discard lines steeper than this (deg from horizontal). A "top threshold" to avoid line detection on very steep lines from patent drawings that can be on the page
    - margin_frac: discard lines that live only in page margins (fraction of width) (noise)
    """
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape
    min_len = int(w * min_rel_len)

    # run HoughLinesP
    linesp = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=80,
        minLineLength=min_len, maxLineGap=20
    )

    if linesp is None or len(linesp) == 0:
        if debug:
            print("[deskew] No lines detected.")
        return pil_img, 0.0

    # compute weighted angle
    angles, lengths = [], []

    # Sort lines by their vertical position (top-to-bottom)
    lines_sorted = sorted(linesp.reshape(-1, 4), key=lambda l: (l[1] + l[3]) / 2)

    for x1, y1, x2, y2 in lines_sorted[:5]:  # use only first 5 lines, top to bottom
        dx, dy = x2-x1, y2-y1
        length = (dx*dx + dy*dy)**0.5
        if length < min_len:
            continue
        # ignore lines in margins
        if (x1 < w*margin_frac and x2 < w*margin_frac) or \
           (x1 > w*(1-margin_frac) and x2 > w*(1-margin_frac)):
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # normalize to [-90,90]
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        if abs(ang) > max_angle:
            continue
        angles.append(ang)
        lengths.append(length)

    if not angles:
        if debug:
            print("[deskew] No usable lines after filtering.")
        return pil_img, 0.0

    angle = float(np.sum(np.array(angles)*np.array(lengths)) / np.sum(lengths))

    # if skewdness under threshold, skip rotation (i.e. deskewing)
    if abs(angle) < rot_threshold:
        if debug:
            print(f"[deskew] Angle {angle:.2f}° < threshold {rot_threshold}, skipping rotation")
        return pil_img, 0.0

    # rotate to deskew (undo skew)
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (w,h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    pil_out = PilImage.fromarray(rotated)

    if debug:
        # print images showing the deskew process
        print(f"[deskew] Estimated angle: {angle:.2f}° from {len(angles)} lines")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Panel 1: Original + overlay lines
        overlay = img_np.copy()
        for x1, y1, x2, y2 in linesp.reshape(-1, 4):
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        axs[0].imshow(overlay)
        axs[0].set_title("Original with detected lines")
        axs[0].axis("off")

        # Panel 2: Deskewed result
        axs[1].imshow(rotated)
        axs[1].set_title(f"Deskewed ({angle:.2f}°)")
        axs[1].axis("off")

        plt.show()


    return pil_out, angle


# --- Utility: Classifier if centered metadata is centered or in two cols
def detect_centered_metadata_wholepage(
    first_page_docs: List,  # the page elements (called "documents") from unstructured
    layout_w: float,
    layout_h: float,
    *,
    debug: bool = False,
    # Tunable parameters
    center_tol_frac: float = 0.10,      # how far from center (p/m frac of page width) counts as "centered"
    min_box_area_rel: float = 0.00025,   # filter tiny boxes (relative to page area) (basically to filter our the line-count numbers that in the patent text)
    min_boxes_in_stack: int = 2,        # min. number of distinct documents (= text elements) to make up a vertical (= condidate for stacked metadata section).
    #max_v_gap_frac: float = 0.00125,       # max vertical gap between stacked boxes (fraction of page height)
    max_v_gap: float = 23,      # max. allowed gap between documents in a vertical stack
    #wide_frac: float = 0.55,            # box width fraction to count as a single-wide centered block
    min_stack_mean_width_frac: float = 0.005, # mean width fraction for a stacked centered block
    min_block_height_frac: float = 0.01,     # minimum vertical extent fraction for a block to matter
    min_y_loc_on_page: float = 3/5,          # maximum vertical location on page (frac) to count as a condidate for centered metadata section    
):
    """
    Detect whether a centered metadata section exists anywhere on the page. This deermines the masking logic for the OCR later.
    Returns a dict.
    """

    # 1. Build box list (layout coordinates)
    boxes = []
    for i, d in enumerate(first_page_docs):
        pts = d.metadata["coordinates"]["points"]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        w = x1 - x0
        h = y1 - y0
        area = w * h
        boxes.append({
            "idx": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
            "w_rel": w / layout_w, "h_rel": h / layout_h, "area_rel": area / (layout_w * layout_h),
            "text": d.page_content.strip() if d.page_content else ""
        })

    if not boxes:
        #print("yikes")
        # fallback to full-page ocr
        return {
            "metadata_centered": False,
            "classification": "double_column",
            "centered_blocks": [],
            "reason": "no_boxes_found"
        }

    # 2. Filter unwanted tiny boxes (noise) and huge boxes (imaged/drawings)
    filtered = []
    for b in boxes:
        if b["area_rel"] < min_box_area_rel:
            continue
        if (b["y1"] - b["y0"]) > 0.2 * layout_h:  # taller than 20% of page height
            continue
        text = first_page_docs[b["idx"]].page_content.strip().lower()
        if "united states patent office" in text:
            continue
        filtered.append(b)

    if debug:
        print("filtered boxes:")
        for b in filtered:
            print(f"idx={b['idx']} x0={b['x0']:.1f} x1={b['x1']:.1f} "
                f"w_rel={b['w_rel']:.3f} area_rel={b['area_rel']:.5f} "
                f"ooo TEXT ooo='{b['text'][:50]}'")

    if not filtered:
        #print("no boxes filtered")
        filtered = boxes[:]  # conservative fallback

    # 3. Find centered candidate boxes (whole page)
    center_x = 0.5 * layout_w
    centered_candidates = [
        b for b in filtered
        if abs((b["cx"] / layout_w) - 0.5) <= center_tol_frac
    ]

    if debug:
        print(f"[detect_centered] total_boxes={len(boxes)}, filtered={len(filtered)}, centered_candidates={len(centered_candidates)}")

    centered_blocks = []

    # 4. If no centered candidates -> definitely no centered metadata
    if not centered_candidates:
        return {
            "metadata_centered": False,
            "classification": "double_column",
            "centered_blocks": [],
            "reason": "no_centered_candidates"
        }

    # 5. Sort centered candidates by vertical position and group into vertical blocks
    centered_sorted = sorted(centered_candidates, key=lambda b: b["y0"])

    if debug:
        print("centered_sorted boxes:")
        for b in centered_sorted:
            print(f"idx={b['idx']} cx={b['cx']:.1f} "
                f"w_rel={b['w_rel']:.3f} "
                f"ooooooo TEXT ooooooo='{b['text'][:50]}'")

    current_block = [centered_sorted[0]]
    for b in centered_sorted[1:]:
        prev = current_block[-1]
        gap = b["y0"] - prev["y1"]
        #print("=============== vertical gap between centered_sorted elements:", gap)
        #print("=============== max vertical gap allowed:", max_v_gap)
        if gap <= max_v_gap:
            current_block.append(b)
        else:
            # finalize prev block
            ys0 = min(bb["y0"] for bb in current_block)
            ys1 = max(bb["y1"] for bb in current_block)
            mean_w = float(np.mean([bb["w_rel"] for bb in current_block]))
            block_area = float(np.sum([bb["area_rel"] for bb in current_block]))
            centered_blocks.append({
                "box_indices": [bb["idx"] for bb in current_block],
                "y0": ys0, "y1": ys1, "height_frac": (ys1 - ys0) / layout_h,
                "mean_w_frac": mean_w, "area_frac": block_area, "y0_frac": ys0/layout_h,
                "n_boxes": len(current_block)
            })
            current_block = [b]

    # finalize last block
    if current_block:
        ys0 = min(bb["y0"] for bb in current_block)
        ys1 = max(bb["y1"] for bb in current_block)
        mean_w = float(np.mean([bb["w_rel"] for bb in current_block]))
        block_area = float(np.sum([bb["area_rel"] for bb in current_block]))
        centered_blocks.append({
            "box_indices": [bb["idx"] for bb in current_block],
            "y0": ys0, "y1": ys1, "height_frac": (ys1 - ys0) / layout_h,
            "mean_w_frac": mean_w, "area_frac": block_area, "y0_frac": ys0/layout_h,
            "n_boxes": len(current_block)
        })

    if debug:
        print("[detect_centered] found centered_blocks:", centered_blocks)

    # 6. Decide: block-level rules
    chosen_block = None
    for block in centered_blocks:
        # --- require multiple centered boxes ---
        # checks each vertical clustered of centered blocks against the set thresholds
        if block["n_boxes"] > min_boxes_in_stack and \
           block["mean_w_frac"] >= min_stack_mean_width_frac and \
           block["height_frac"] >= min_block_height_frac and \
           block["y0_frac"] <= min_y_loc_on_page:
            chosen_block = {**block, "reason": "stacked_centered_boxes"}
            break

    if chosen_block is not None:
        if debug:
            print("centered metadata found: ", chosen_block["reason"])
        return {
            "metadata_centered": True,
            "classification": "centered",
            "centered_blocks": [chosen_block],
            "reason": chosen_block["reason"]
        }

    # 7. No block passed the rules -> default to double-column (i.e. no centered metadata section found on page)
    return {
        "metadata_centered": False,
        "classification": "double_column",
        "centered_blocks": centered_blocks,
        "reason": "no_block_passed_rules"
    }



# --- Utility: crop and mask first page based on centered/two-column metadata
def crop_and_mask_first_page(docs, doctr_doc, predictor, pdf_basename_pat_number, save_dir, debug=False, save_debug_img=False):
    """
    Mask and crop the first page according to single/double-column logic.
    Returns OCR text, applied case, and path to saved image.
    """
    first_page_docs = docs
    meta0 = first_page_docs[0].metadata
    layout_w = float(meta0["coordinates"]["layout_width"])
    layout_h = float(meta0["coordinates"]["layout_height"])

    page_np = doctr_doc[0]
    page_img = PilImage.fromarray(page_np)
    img_w, img_h = page_img.size
    sx = img_w / layout_w
    sy = img_h / layout_h

    # --- DEBUG: draw bounding boxes ---
    debug_img = page_img.copy()
    draw_debug = ImageDraw.Draw(debug_img)
    for d in first_page_docs:
        pts = d.metadata["coordinates"]["points"]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x0, y0, x1, y1 = min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy
        draw_debug.rectangle([x0, y0, x1, y1], outline="red", width=2)

    debug_img_out_path = os.path.join(save_dir, f"{pdf_basename_pat_number}_debug_boxes.png")
    os.makedirs(save_dir, exist_ok=True)
    if save_debug_img:
        debug_img.save(debug_img_out_path)

    # --- Detect centered metadata ---
    res = detect_centered_metadata_wholepage(first_page_docs, layout_w, layout_h, debug=False)
    metadata_centered = res["metadata_centered"]
    
    if debug:
        print("--------- metadata centered?:", metadata_centered)

    # --- Step 2: Initialize mask image ---
    cropped_img = page_img.copy()
    draw = ImageDraw.Draw(cropped_img)

    # --- Step 3: Define patterns to find the y-cutoff of the page (only the text above this cutoff is important). Everything after the "claims" section of the patent is not so important (does not contain inventor/assignee location information)---
    target_pattern = re.compile(
        r"(?:\d+\s*)?[\(\[\{]?\s*(claims?|cl\.?|ci\.?)\b[\)\]\}\.:]?|"
        r"(?<!\w)[\(\[\{]\s*52\s*[\)\]\}](?!\w)", # 52 indicates the claim section on newer patets (INID code)
        re.IGNORECASE
    )
    target_pattern_T0 = re.compile(r"\bdrawings?\b\.?", re.IGNORECASE)
    case_applied = "fallback"

    # ---------------- Single-column (centered metadata) ----------------
    if metadata_centered:
        # Choose which pattern to use (Defensive patents T0 have distinct format)
        if "T0" in pdf_basename_pat_number:
            target_boxes = [d for d in first_page_docs if target_pattern_T0.search((d.page_content or "").lower())]
        else:
            target_boxes = [d for d in first_page_docs if target_pattern.search((d.page_content or "").lower())]

        # Initialize variables
        d_target = None
        y_crop = layout_h
        other_boxes = []
        max_x = 0

        if target_boxes:
            # Select target box
            d_target = target_boxes[-1] if "T0" in pdf_basename_pat_number else target_boxes[0]
            pts = d_target.metadata["coordinates"]["points"]
            y_crop = max([float(p[1]) for p in pts]) * sy + 100

        # Prepare other boxes
        for d in first_page_docs:
            if d is d_target:
                continue
            pts = d.metadata["coordinates"]["points"]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            other_boxes.append((x0*sx, y0*sy, x1*sx, y1*sy))

            # Update max_x for boxes below claim
            if d_target and y0 * sy >= y_crop-100 and x0 < layout_w/4 and x1 < 0.75*layout_w and y0 * sy <= y_crop + 200:
                max_x = max(max_x, x1 * sx)

        # Draw masks if target exists
        if d_target:
            draw.rectangle([max_x, y_crop-100, img_w, img_h], fill="red")
            # Mask boxes starting near max_x
            for (x0_px, y0_px, x1_px, y1_px) in other_boxes:
                if max_x - 10 <= x0_px <= max_x and y0_px >= y_crop-100:
                    draw.rectangle([x0_px, y0_px, x1_px, y1_px], fill="red")
            # Mask boxes ending in last quarter
            for x0_px, y0_px, x1_px, y1_px in other_boxes:
                if y0_px >= y_crop-100 and x1_px > 0.75 * img_w:
                    draw.rectangle([x0_px, y0_px, img_w, img_h], fill="red")

        case_applied = "Single-Col (Metadata in single col)" if d_target else "Single-Col (No target box found)"

    # ---------------- Double-column ----------------
    else:
        target_boxes = [d for d in first_page_docs if target_pattern.search((d.page_content or "").lower())]
        

        # Initialize variables
        d_target = None
        y_crop = layout_h * sy  # Default: no crop (end of page)
        y1_target_px = layout_h * sy
        other_boxes = []
        max_x = 0

        if target_boxes:
            # --- pick first target box that starts in first quarter of the page ---
            for d in target_boxes:
                pts = d.metadata["coordinates"]["points"]
                xs = [float(p[0]) for p in pts]
                x0 = min(xs)
                if x0 <= layout_w / 4:  # starts in first quarter
                    d_target = d
                    break

            # Only apply cropping if we found a valid target box
            if d_target:
                pts = d_target.metadata["coordinates"]["points"]
                y_crop = max([float(p[1]) for p in pts]) * sy + 100
                y1_target_px = max([float(p[1]) for p in pts]) * sy

            # --- Find reference y-coordinate (top section of patent) ---
            keyword_pattern = re.compile(
                r"(date of patent|patented|united|states|patent|"
                r"(?<!\w)[\(\[\{]\s*45\s*[\)\]\}](?!\w)|"
                r"(?<!\w)[\(\[\{]\s*11\s*[\)\]\}](?!\w)|"
                r"(?<!\w)[\(\[\{]\s*19\s*[\)\]\}](?!\w))",
                re.IGNORECASE
            )
            y_ref = 0
            first_page_docs_sorted = sorted(
                first_page_docs,
                key=lambda d: (min(float(p[1]) for p in d.metadata["coordinates"]["points"]),
                               min(float(p[0]) for p in d.metadata["coordinates"]["points"]))
            )
            for d in first_page_docs_sorted:
                text = d.page_content or ""
                match = keyword_pattern.search(text)
                if match:
                    pts = d.metadata["coordinates"]["points"]
                    y_val = max([float(p[1]) for p in pts])
                    matched_str = match.group(0).lower()
                    y_ref = y_val * sy + (15 if any(k in matched_str for k in ["date of patent", "patented", "[45"]) else 80)
                    print("------- y_ref set to:", y_ref, "from matched text:", matched_str)
                    break

            # --- Masking logic ---
            for d in first_page_docs_sorted:
                if d is d_target:
                    continue
                pts = d.metadata["coordinates"]["points"]
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                other_boxes.append((x0*sx, y0*sy, x1*sx, y1*sy))

                if y0 * sy >= y_ref and x0 < layout_w/4 and x1 < 0.75*layout_w and y0 * sy <= y1_target_px + 300:
                    max_x = max(max_x, x1*sx)

            # First mask
            draw.rectangle([max_x, y_ref+15, img_w, img_h], fill="blue")
            # Second mask: boxes starting near max_x (boxes along the center of the page (basically the line numbering numbers))
            for (x0_px, y0_px, x1_px, y1_px) in other_boxes:
                if max_x - 10 <= x0_px <= max_x and y0_px > y_ref:
                    draw.rectangle([x0_px, y0_px, x1_px, y1_px], fill="blue")
            # Third mask: boxes ending in last quarter
            for x0_px, y0_px, x1_px, y1_px in other_boxes:
                if y0_px > y_ref and x1_px > 0.75 * img_w and (d_target):
                    draw.rectangle([x0_px, y0_px, img_w, img_h], fill="blue")

        case_applied = "Double-Col (Metadata in double col)" if d_target else "Double-Col (No target box found)"

    # --- Step 4: Crop image ---
    crop_box = (0, 0, img_w, min(y_crop + 160, img_h))
    cropped_img = cropped_img.crop(crop_box)

    # --- Step 5: Save cropped/masked image ---
    os.makedirs(save_dir, exist_ok=True)
    img_out_path = os.path.join(save_dir, f"{pdf_basename_pat_number}.png")
    if save_debug_img:
        cropped_img.save(img_out_path)

    # --- Step 6: OCR with DocTR ---
    cropped_np = np.array(cropped_img)
    result = predictor([cropped_np])
    text = result.render()

    return text, case_applied, img_out_path




# ==========================================================================================================
# ==========================================================================================================
# ===============================================            ===============================================
# =============================================== NODE LOGIC ===============================================
# ===============================================            ===============================================
# ==========================================================================================================
# ==========================================================================================================



# --- STEP 1: Convert PDFs to single-page PDFs ---
def run_step1(input_list, output_pdf_folder, log_paths):
    """
    Convert each PDF in input_list to a single-page PDF (relevant page only).
    input_list: list of file paths (already extracted from tar/pdf directory if needed)
    """
    success_log = open(log_paths["success"], "w", encoding="utf-8")
    failure_log = open(log_paths["failure"], "w", encoding="utf-8")
    error_log   = open(log_paths["error"], "w", encoding="utf-8")

    pdf_entries = []

    for pdf_path in input_list:
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            page_index = find_relevant_page_index_fast(pdf_bytes)
            if page_index is None:
                pdf_basename = os.path.basename(pdf_path)
                failure_log.write(f"{pdf_basename}\n")
                continue

            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            writer.add_page(reader.pages[page_index])

            out_pdf_name = f"{Path(pdf_path).stem}.pdf"
            out_pdf_path = os.path.join(output_pdf_folder, out_pdf_name)
            os.makedirs(output_pdf_folder, exist_ok=True)
            with open(out_pdf_path, "wb") as f_out:
                writer.write(f_out)

            pdf_entries.append({
                "source_pdf": pdf_path,
                "single_page_pdf": out_pdf_path
            })
            success_log.write(f"{out_pdf_name}\n")

        except Exception as e:
            error_log.write(f"{pdf_path} -> {e}\n")

    success_log.close()
    failure_log.close()
    error_log.close()

    return pdf_entries


# --- STEP 1 NODE ---
def step1_node(state: PipelineState) -> PipelineState:
    """
    Step 1: Convert PDFs in input_list to single-page PDFs.
    """
    start_time = time()

    # make alias for quicker access
    pe = state["pdf_extraction"]
    
    input_list = pe.get("input_list", [])
    output_pdf_folder = os.path.join(pe["pdf_root"], "single_page_pdfs")
    os.makedirs(output_pdf_folder, exist_ok=True)

    # --- Log paths ---
    log_dir = os.path.join(pe["pdf_root"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_paths = {
        "success": os.path.join(log_dir, "step1_success.txt"),
        "failure": os.path.join(log_dir, "step1_failure.txt"),
        "error":   os.path.join(log_dir, "step1_error.txt"),
    }

    # --- Run Step 1 ---
    pdf_entries = run_step1(input_list, output_pdf_folder, log_paths)

    # --- Update state ---
    pe.update({
        "output_pdf_folder": output_pdf_folder,
        "log_paths": log_paths,
        "pdf_entries": pdf_entries,
    })

    print(f"✅ Step 1 complete. {len(pdf_entries)} PDFs extracted into {output_pdf_folder}")
    total_time = time() - start_time
    avg_runtime = total_time / len(pdf_entries) if pdf_entries else 0
    print(f"⏱️ Average time per PDF STEP1: {avg_runtime:.2f} seconds")

    return state


# --- STEP 2 NODE ---
def step2_node(state: PipelineState) -> PipelineState:
    """
    Step 2: Convert PDFs to images, deskew (if needed), crop/mask first page, and run OCR.
    Ensures docs coordinates match deskewed image.
    """
    start_time = time()
    predictor = state["ocr"]["predictor"]
    #run_ts = state["pdf_extraction"]["run_ts"]

    # --- Use batch folder from Step 1 ---
    pdf_root = state["pdf_extraction"]["pdf_root"]
    input_pdf_folder = state["pdf_extraction"]["output_pdf_folder"]

    # --- Subfolders inside batch folder ---
    output_text_folder = os.path.join(pdf_root, f"txt_results")
    output_img_folder = os.path.join(pdf_root, "cropped_masked_images")
    log_dir = os.path.join(pdf_root, "logs")

    os.makedirs(output_text_folder, exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Logs ---
    timestamp = datetime.now(ZoneInfo("Europe/Vienna")).strftime("%Y%m%d_%H%M%S")
    success_log_path = os.path.join(log_dir, f"step2_success.txt")
    failure_log_path = os.path.join(log_dir, f"step2_failure.txt")

    pdf_files = sorted(glob(os.path.join(input_pdf_folder, "*.pdf")))
    processing_results = []

    # make pattern for extracting patent number from file name
    pat_num_pattern = re.compile(r"(?:02|03|04|PP|T0|D0|RE)\d{6}")

    with open(success_log_path, "w", encoding="utf-8") as success_log, \
         open(failure_log_path, "w", encoding="utf-8") as failure_log:

        success_log.write("file, case, deskew_angle, time_s\n")
        failure_log.write("file, error\n")

        for pdf_path in pdf_files:
            pdf_basename = os.path.basename(pdf_path)
            print(f"------------------- Step 2 Processing: {pdf_basename} -------------------")
            match = pat_num_pattern.search(pdf_basename)
            pdf_basename_pat_number = match.group(0) if match else os.path.splitext(pdf_basename)[0]

            try:
                # --- Step 1: convert to temp single-page PDF ---
                tmp_page = single_page_pdf_to_tempfile(pdf_path, page_index=0)

                # --- Step 2: Convert first page to image for deskewing ---
                pages = convert_from_path(tmp_page, dpi=300, first_page=1, last_page=1)
                if not pages:
                    raise ValueError(f"No pages extracted from PDF {pdf_basename}")
                pil_page = pages[0]

                # --- Step 3: Deskew ---
                pil_deskewed, deskew_angle = deskew_with_hough(pil_page, debug=False)

                # --- Step 4: Decide if we need to rebuild PDF ---
                if abs(deskew_angle) >= 1:  # threshold matches deskew_with_hough
                    # Save deskewed page as temporary PDF for UnstructuredLoader
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                        tmp_img_path = tmp_img_file.name
                    pil_deskewed.save(tmp_img_path, format="PNG")

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf_file:
                        tmp_pdf_path = tmp_pdf_file.name

                    pdf_tmp = FPDF(unit="pt", format=[pil_deskewed.width, pil_deskewed.height])
                    pdf_tmp.add_page()
                    pdf_tmp.image(tmp_img_path, 0, 0, pil_deskewed.width, pil_deskewed.height)
                    pdf_tmp.output(tmp_pdf_path)

                    loader_input = tmp_pdf_path
                    doctr_doc = [np.array(pil_deskewed)]
                else:
                    # No deskew -> reuse original PDF + image
                    loader_input = tmp_page
                    doctr_doc = [np.array(pil_page)]

                # --- Step 5: Load documents with UnstructuredLoader ---
                loader = UnstructuredLoader(loader_input, strategy="hi_res", detection_model=yolox_model) # model= yolox_model
                docs = list(loader.lazy_load())
                if not docs:
                    raise ValueError("No docs found in page")

                # --- Step 6: Crop + OCR ---
                text, case_applied, img_out_path = crop_and_mask_first_page(
                    docs=docs,
                    doctr_doc=doctr_doc,
                    predictor=predictor,
                    pdf_basename_pat_number=pdf_basename_pat_number,
                    save_dir=output_img_folder,
                    debug=False,        # set True for info and debug images
                    save_debug_img=state.get("pdf_extraction", {}).get("save_debug_img", False)
                )

                # --- Step 7: Save OCR text ---
                txt_out = os.path.join(output_text_folder, f"{pdf_basename_pat_number}.txt")
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(text)
                

                elapsed = time() - start_time
                success_log.write(f"{pdf_basename},{case_applied},{deskew_angle:.2f},{elapsed:.2f}\n")

                processing_results.append({
                    "pdf": pdf_path,
                    "txt_out": txt_out,
                    "img_out": img_out_path,
                    "case": case_applied,
                    "deskew_angle": deskew_angle,
                    "status": "success"
                })

            except Exception as e:
                failure_log.write(f"{pdf_basename}, error: {e}, traceback: {traceback.format_exc()}\n")
                print(f"❌ Failed on {pdf_basename}:\n{traceback.format_exc()}")
                processing_results.append({
                    "pdf": pdf_path,
                    "error": str(e),
                    "status": "failure"
                })

    # --- Update state ---
    state["ocr"].update({
        "input_pdf_folder": input_pdf_folder,
        "output_text_folder": output_text_folder,
        "output_img_folder": output_img_folder,
        "log_paths": {"success": success_log_path, "failure": failure_log_path},
        "pdf_files": pdf_files,
        "processing_results": processing_results,
        "timestamp": timestamp
    })

    print(f"✅ Step 2 complete. {len([r for r in processing_results if r['status']=='success'])} files processed into {output_text_folder}")
    runtime = time() - start_time
    avg_time = runtime/len(processing_results) if processing_results else 0
    print(f"⏱️ Average time per PDF STEP2: {avg_time:.2f} seconds")

    return state


# Define graph for steps 1 and 2 (set up as a langgraph because it's easier to manage this way and easily attachable to step 3 for the full pipeline)
workflow = StateGraph(PipelineState)
workflow.add_node("step1_pdf_extraction", step1_node)
workflow.add_node("step2_ocr", step2_node)
# optional place to add step3_llm here

workflow.set_entry_point("step1_pdf_extraction")
workflow.add_edge("step1_pdf_extraction", "step2_ocr")
workflow.add_edge("step2_ocr", END)

graph_step1_2 = workflow.compile()


# --- Helper functions for the runner/main function ---


def _extract_single_member(tar_path: str, member_path: str, tmp_dir: str) -> str:
    """Extracts a single member from a tar file to a temporary directory."""
    
    # 1. Ensure the temporary directory exists
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Extract the single member
    try:
        with tarfile.open(tar_path, mode="r:*") as tar:
            # Get the member info
            member = tar.getmember(member_path)
            # Extract the member to the temporary directory
            tar.extract(member, path=tmp_dir)
            
            # The extracted file will be located at: tmp_dir/member_path
            # Since member_path often includes internal folder structure (e.g., '02/patent.pdf'), we return the full, absolute path to the extracted file.
            return os.path.join(tmp_dir, member_path)
    except Exception as e:
        # Re-raise the exception for the caller to handle as an extraction failure
        raise RuntimeError(f"Tar extraction failed for {member_path}: {e}")

def run_pipeline_single_pdf(
    pdf_path_id: str, 
    run_root: str, 
    initial_state: dict, 
    graph_step1_2, 
    *,
    cleanup_tmp: bool = True
):
    
    tmp_dir_for_pdf = None
    status = "failure"
    error_message = None
    
    # 1. Determine Input Mode and Path Components
    is_tar_member = '|' in pdf_path_id
    
    # A) Determine file identification and output location
    try:
        if is_tar_member:
            # --- TAR MEMBER MODE: ID is "tar_path|member_path" ---
            tar_path, member_path = pdf_path_id.split('|', 1) 
            pdf_stem = Path(member_path).stem
            
            # 2. Extract the PDF file to a temporary location
            tmp_dir_for_pdf = tempfile.mkdtemp(prefix=f"tmp_extract_{pdf_stem}_")
            
            # Call the helper function to extract the single file
            extracted_file_path = _extract_single_member(tar_path, member_path, tmp_dir_for_pdf)

        else:
            # --- PDF FOLDER MODE: ID is the full path on disk ---
            pdf_stem = Path(pdf_path_id).stem
            extracted_file_path = pdf_path_id # File is already on disk
            # No temporary directory created, so cleanup is skipped later
            tmp_dir_for_pdf = None
            
        # Define Output Folder (Unique per patent ID)
        pdf_root = os.path.join(run_root, pdf_stem)
        os.makedirs(pdf_root, exist_ok=True)
        print(f"======== Processing: {pdf_stem}")
        
    except Exception as e:
        # Handles errors in path parsing or file extraction
        return {"pdf_stem": Path(pdf_path_id).stem, "status": "setup_failure", "error": str(e)}

    
    # 3. Prepare state for ONE PDF
    state_single = copy.deepcopy(initial_state)
    pe = state_single.setdefault("pdf_extraction", {})
    
    # Inject the actual path on disk
    pe["pdf_path"] = extracted_file_path    # The path on disk (temp dir or original path)
    pe["input_list"] = [extracted_file_path]    # pass as a list because step1_node expects it like this
    pe["pdf_stem"] = pdf_stem   # unique ID
    pe["output_root"] = run_root
    pe["pdf_root"] = pdf_root   # The unique output folder
    
    # 4. Invoke the graph and handle logging
    try:
        state_after_2 = graph_step1_2.invoke(state_single)

        status = "success"
        with open(os.path.join(pdf_root, "SUCCESS.log"), "w", encoding="utf-8") as s_log:
             s_log.write(f"Processed successfully: {pdf_stem}\n")

    except Exception as e:
        status = "error"
        error_message = str(e)
        error_log_path = os.path.join(pdf_root, "ERROR.log")
        with open(error_log_path, "a", encoding="utf-8") as err:
            err.write(f"\n[{datetime.now(ZoneInfo('Europe/Vienna'))}] Processing failed:\n")
            err.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            err.write("\n" + "="*80 + "\n")
        print(f"❌ Processing {pdf_stem} failed (see {error_log_path}): {e}")

    finally:
        # 5. Cleanup temporary extraction directory only if it was created (tar mode)
        if cleanup_tmp and tmp_dir_for_pdf and os.path.isdir(tmp_dir_for_pdf):
            try:
                shutil.rmtree(tmp_dir_for_pdf, ignore_errors=True)
            except Exception as e:
                print(f"WARNING: Failed to cleanup temp dir {tmp_dir_for_pdf}: {e}")

        # 6. Cleanup single_page_pdfs — pure intermediate scratch, never needed
        # again once the .txt exists. Note: step2_node swallows its own
        # per-file exceptions internally (graph_step1_2.invoke() can return
        # normally even if OCR failed for this file), so we check for the
        # actual .txt output rather than trusting `status` alone.
        cleanup_enabled = initial_state.get("pdf_extraction", {}).get("cleanup_intermediate_files", False)
        txt_out_path = os.path.join(pdf_root, "txt_results", f"{pdf_stem}.txt")
        if cleanup_enabled and os.path.isfile(txt_out_path):
            single_page_dir = os.path.join(pdf_root, "single_page_pdfs")
            if os.path.isdir(single_page_dir):
                shutil.rmtree(single_page_dir, ignore_errors=True)

    # 7. Return the result
    return {"pdf_stem": pdf_stem, "status": status, "error": error_message}


def _worker_process_pdf(pdf_path_id: str, run_root: str, pdf_extraction_config: dict) -> dict:
    """
    Entry point for a ProcessPoolExecutor worker. Runs in a separate process.

    Deliberately does NOT receive `predictor` or `graph_step1_2` as
    arguments — those are live model/graph objects and would have to be
    pickled to cross the process boundary, which is slow (or fails outright)
    for objects like this. Instead, each worker builds its own state using
    the `predictor` / `graph_step1_2` globals that already exist in ITS OWN
    process because this module was imported when the worker started (or,
    on Linux with the default 'fork' start method, inherited directly from
    the parent's already-loaded copies via copy-on-write). Only plain,
    picklable arguments (strings, dicts of primitives) cross the boundary.
    """
    worker_initial_state = {
        "pdf_extraction": pdf_extraction_config,
        "ocr": {"predictor": predictor},
    }
    return run_pipeline_single_pdf(
        pdf_path_id=pdf_path_id,
        run_root=run_root,
        initial_state=worker_initial_state,
        graph_step1_2=graph_step1_2,
    )


def _discover_pdfs_with_checkpoint(initial_state: dict, run_root: str) -> list[str]:
    """
    Discovers all available PDFs from the input directory.
    Supports a directory containing loose pdfs or a directory containing .tar files.
    Generates a unique path string for each pdf and applies checkpoint filter.
    """
    input_path = initial_state["pdf_extraction"]["input_path"]
    master_log_file = initial_state["pdf_extraction"]["master_log_file_path"]
    allowed_dirs = initial_state["pdf_extraction"].get("allowed_dirs", None)
    
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is not a valid directory: {input_path!r}")

    all_pdf_paths = set()
    input_mode = "unknown"
    
    # Check if the input directory contains any .tar files
    tar_file_paths = glob(os.path.join(input_path, "*.tar"), recursive=True)
    
    if tar_file_paths:
        # --- MODE 1: Directory of .tar files ---
        input_mode = "tar_directory"
        
        for tar_path in tar_file_paths:
            try:
                with tarfile.open(tar_path, mode="r:*") as tar:
                    for m in tar.getmembers():
                        name = m.name # The path inside the tar file
                        
                        if not name.lower().endswith(".pdf"):
                            continue
                            
                        # Apply allowed_dirs filter
                        if allowed_dirs:
                            parts = Path(name).parts
                            if not any(p in allowed_dirs for p in parts):
                                continue
                                
                        if m.isfile() or m.size > 0:
                            # Unique ID: <TAR_PATH>|<MEMBER_PATH>
                            unique_id = f"{tar_path}|{name}"
                            all_pdf_paths.add(unique_id)
            except Exception as e:
                print(f"WARNING: Could not open or process tar file {tar_path}: {e}")
                
    else:
        # --- MODE 2: Directory of loose pdfs ---
        input_mode = "pdf_directory"
        
        # Recursively find all PDFs in the input directory
        # We don't apply allowed_dirs here unless the PDF paths contain the directory structure
        loose_pdf_paths = [str(p) for p in Path(input_path).rglob("*.pdf")]
        
        # For consistency, the unique ID is just the full path. 
        for p in loose_pdf_paths:
            # Unique ID: <FULL_PATH>|NO_MEMBER
            all_pdf_paths.add(p)


    all_pdf_paths = sorted(list(all_pdf_paths))
    total_available = len(all_pdf_paths)
    
    # 3. Checkpoint filtering
    checkpoint_enabled = initial_state.get("pdf_extraction", {}).get("start_from_checkpoint", False)
    if checkpoint_enabled and os.path.exists(master_log_file):
        print(f"Applying checkpoint from: {master_log_file}")
        with open(master_log_file, "r", encoding="utf-8") as f:
            processed_files = {
                line.strip().split(",")[0]
                for line in f.readlines() if line.strip()
            }
        
        # Filter based on the patent ID stem (i.e. unique patent number)
        all_pdf_paths = [
            p for p in all_pdf_paths 
            if Path(p).stem not in processed_files
        ]
        
    # 4. Apply limit_nr_of_pdfs_to_process limit
    limit_nr_of_pdfs_to_process = initial_state.get("pdf_extraction", {}).get("limit_nr_of_pdfs_to_process", None)
    if limit_nr_of_pdfs_to_process is not None:
        all_pdf_paths = all_pdf_paths[:limit_nr_of_pdfs_to_process]
    
    print(f"Run root: {run_root}")
    print(f"Input mode: {input_mode}, total available: {total_available}, available after checkpoint: {len(all_pdf_paths)}")
    
    return all_pdf_paths



def _aggregate_logs_and_master_log(results: list[dict], run_root: str, master_log_file: str):
    """
    Aggregates results across runs into shared log file
    """
    run_log_dir = os.path.join(run_root, "unified_logs")
    os.makedirs(run_log_dir, exist_ok=True)
    agg_success_log = os.path.join(run_log_dir, "success_all.txt")
    agg_failure_log = os.path.join(run_log_dir, "failure_all.txt")
    agg_error_log   = os.path.join(run_log_dir, "error_all.txt")
    
    # 1. Read existing master log to prevent duplicates during aggregation
    if os.path.exists(master_log_file):
        with open(master_log_file, "r", encoding="utf-8") as f:
            existing_stems = {
                line.strip().split(",")[0]
                for line in f.readlines()[1:] if line.strip()
            }
    else:
        existing_stems = set()

    # 2. Write to all logs
    processed_count = 0
    with open(agg_success_log, "a", encoding="utf-8") as s_log, \
         open(agg_failure_log, "a", encoding="utf-8") as f_log, \
         open(agg_error_log, "a", encoding="utf-8") as e_log, \
         open(master_log_file, "a", encoding="utf-8") as m_log:
        
        for r in results:
            pdf_stem = r["pdf_stem"]
            if pdf_stem in existing_stems:
                continue # Already logged in a previous run
            
            processed_count += 1
            
            if r["status"] == "success":
                s_log.write(f"{pdf_stem}\n")
            elif r["status"] == "error":
                f_log.write(f"{pdf_stem}, Error: {r.get('error')[:100]}...\n")
                
                # Retrieve the full error log from the unique pdf folder
                pdf_error_log_path = os.path.join(run_root, pdf_stem, "ERROR.log")
                if os.path.exists(pdf_error_log_path):
                    with open(pdf_error_log_path, "r", encoding="utf-8") as pdf_err:
                        e_log.write(f"\n[{datetime.now(ZoneInfo('Europe/Vienna'))}] PDF: {pdf_stem} - Full Error Log:\n")
                        e_log.write(pdf_err.read())
                        e_log.write("\n" + "="*80 + "\n")
                
            # Update the master log regardless of success/failure
            m_log.write(f"{pdf_stem}\n")
            
    print(f"\n✅ Log aggregation finished. {processed_count} new entries added.")
    return {"processed_total": processed_count}


def _aggregate_ocr_results(current_run_root: str, master_text_root: str) -> None:
    """
    Collects all individual OCR .txt results from the current run and copies 
    them to a central master folder, only adding files that do not yet exist
    to prevent duplicates and accidental overwrites.
    
    Args:
        current_run_root: The root folder of the current pipeline run.
        master_text_root: The path to the central folder for all aggregated .txt results.
    """
    
    # 1. Ensure the master folder exists
    Path(master_text_root).mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    total_skipped = 0
    
    # Use glob to find all *.txt files nested two directories deep 
    search_pattern = Path(current_run_root) / "**" / "txt_results" / "*.txt"
    
    for src_path in glob(str(search_pattern), recursive=True):
        src_path = Path(src_path)
        file_name = src_path.name 
        dst_path = Path(master_text_root) / file_name
        
        # --- Duplicate check ---
        if dst_path.exists():
            # If the file already exists in the master folder, skip copying.
            total_skipped += 1
            continue        
        try:
            # Copy the file only if it did not exist
            shutil.copy2(src_path, dst_path)
            total_copied += 1
        except Exception as e:
            print(f"WARNING: Could not copy {file_name}. Error: {e}")
            
    #print(f"✅ Aggregation complete. Copied {total_copied} file(s), skipped {total_skipped} existing file(s).")


def _cleanup_aggregated_patent_folders(results: list[dict], run_root: str, master_text_root: str) -> None:
    """
    Deletes each patent's per-run working folder (run_root/<pdf_stem>/) once
    its .txt output is confirmed present in the master aggregation folder.
    Leaves folders for anything not confirmed in the master folder (failures,
    aggregation misses) so they remain available for debugging.
    """
    removed = 0
    kept = 0
    for r in results:
        pdf_stem = r["pdf_stem"]
        master_txt_path = os.path.join(master_text_root, f"{pdf_stem}.txt")
        patent_folder = os.path.join(run_root, pdf_stem)

        if os.path.isfile(master_txt_path) and os.path.isdir(patent_folder):
            try:
                shutil.rmtree(patent_folder)
                removed += 1
            except Exception as e:
                print(f"WARNING: Could not remove {patent_folder}: {e}")
                kept += 1
        else:
            kept += 1

    print(f"✅ Cleanup complete. Removed {removed} per-patent folder(s), kept {kept} (no confirmed .txt in master folder).")


def main():
    print("Initializing pipeline controller...")
    run_ts = datetime.now(ZoneInfo("Europe/Vienna")).strftime("%Y%m%d_%H%M%S")

    # 1. Build initial state with relative path defaults
    initial_state = {
        "pdf_extraction": {
            "input_path": DEFAULT_INPUT_PATH, # relative path for input data: Assume 'test_data' folder exists next to script
            "output_root": DEFAULT_OUTPUT_ROOT, # relative path for output: Creates 'results' folder next to script
            "master_log_file_path": DEFAULT_MASTER_LOG,  # tracks all processed pdfs across runs
            "limit_nr_of_pdfs_to_process": None, # Limiting the run size. If you want to process all: enter "None"
            "start_from_checkpoint": True, # skips pdfs that are in master_log_step1_2_processed.txt (already processed pdfs)
            "allowed_dirs": ["01", "02", "03", "04", "PP", "D0", "T0", "RE"], # Select the type of patent to be extracted (utility: "02" or "03" or "04", plant: "PP", design: "D0", defensive: "T0", reissue: "RE")
            "save_debug_img": False, # save intermediary and debug images from step 2 (only turn on for manual debugging of a small batch)
            "cleanup_intermediate_files": True # delete per-patent single_page_pdfs/ and working folders once their .txt is safely aggregated, so the run leaves behind only the master txt_results folder + logs
        },
        "ocr": {
            "predictor": predictor
        }
    }
    
     # --- SETUP RUN DIRECTORIES ---
    input_path = initial_state.get("pdf_extraction", {}).get("input_path", DEFAULT_INPUT_PATH)
    output_root = initial_state.get("pdf_extraction", {}).get("output_root", DEFAULT_OUTPUT_ROOT)
    master_log_file = initial_state.get("pdf_extraction", {}).get("master_log_file_path", DEFAULT_MASTER_LOG)
    
    # Define the unique run folder for this execution. Use of input foler name plus timestamp
    run_root_base = Path(input_path).name
    run_root = os.path.join(output_root, f"{run_root_base}_{run_ts}")
    
    # Define a central aggregation folder for all .txt results
    OCR_MASTER_FOLDER = os.path.join(os.path.dirname(output_root), "STEP_1_2_RESULTS/Step1_2_Output") 
    
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(os.path.dirname(master_log_file), exist_ok=True)
    
    # Initialize master log if it doesn't exist
    if not os.path.exists(master_log_file):
        with open(master_log_file, "w", encoding="utf-8") as f:
            pass # no header
            
    # 2. Discover available pdfs (across all .tar files)
    all_pdf_paths = _discover_pdfs_with_checkpoint(initial_state, run_root)
    
    if not all_pdf_paths:
        print("No PDFs to process after applying checkpoint / filters. Exiting.")
        return {"run_root": run_root, "processed_total": 0}
    
    results = []
    total_pdfs = len(all_pdf_paths)

    # 3. Parallel execution across worker processes.
    # Each worker only receives plain, picklable arguments (path strings,
    # a config dict of primitives) — never the live `predictor` or
    # `graph_step1_2` objects. See `_worker_process_pdf` for why.
    pdf_extraction_config = initial_state["pdf_extraction"]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {
            executor.submit(_worker_process_pdf, pdf_path, run_root, pdf_extraction_config): pdf_path
            for pdf_path in all_pdf_paths
        }

        completed = 0
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            pdf_stem = Path(pdf_path.split('|')[-1]).stem

            try:
                result = future.result()
            except Exception as e:
                # Catch catastrophic errors (e.g. worker process crashed) and
                # log them as a result item instead of losing them silently.
                error_message = f"FATAL error outside worker (likely setup/init): {e}"
                print(f"❌ {pdf_stem}: {error_message}")
                result = {
                    "pdf_stem": pdf_stem,
                    "status": "fatal_init_error",
                    "error": error_message
                }

            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == total_pdfs:
                print(f"--- Completed {completed}/{total_pdfs} (last: {pdf_stem}, status={result['status']}) ---")
    
    # Synchronization point after parallelization
    # 4. Final Aggregation and Checkpoint Update
    _aggregate_logs_and_master_log(results, run_root, master_log_file)
    
    # 5. Aggregate all OCR text files to a central location
    processed_count = len(results)
    if processed_count > 0:
        _aggregate_ocr_results(run_root, OCR_MASTER_FOLDER)

    # 6. Cleanup per-patent working folders now that .txt output is safely
    # aggregated (only runs if cleanup_intermediate_files is enabled)
    cleanup_enabled = initial_state.get("pdf_extraction", {}).get("cleanup_intermediate_files", False)
    if cleanup_enabled and processed_count > 0:
        _cleanup_aggregated_patent_folders(results, run_root, OCR_MASTER_FOLDER)

    print(f"\n✅ Run completed. Results root: {run_root}")
    return {"run_root": run_root, "processed_total": processed_count}


if __name__ == "__main__":
    main()