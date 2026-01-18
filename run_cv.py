# run_cv.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# -------------------------
# IO
# -------------------------
def imread_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imread_query(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Returns (bgr, alpha_mask_or_none).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, None

    if img.shape[2] == 4:
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        return bgr, mask

    return img[:, :, :3].copy(), None


# -------------------------
# Trim template
# -------------------------
def trim_template(bgr: np.ndarray, alpha_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = bgr.shape[:2]

    if alpha_mask is not None:
        ys, xs = np.where(alpha_mask > 0)
        if xs.size == 0:
            return bgr, alpha_mask
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        return bgr[y1 : y2 + 1, x1 : x2 + 1], alpha_mask[y1 : y2 + 1, x1 : x2 + 1]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    content = gray < 250
    ys, xs = np.where(content)
    if xs.size == 0:
        return bgr, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = 2
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return bgr[y1 : y2 + 1, x1 : x2 + 1], None


# -------------------------
# Hue / color helpers
# -------------------------
def _count_red_green(hsv: np.ndarray, sat_min: int = 40, val_min: int = 40) -> tuple[int, int, int]:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    colored = (s >= sat_min) & (v >= val_min)

    # OpenCV hue: 0..179
    red = colored & ((h <= 10) | (h >= 170))
    green = colored & (h >= 35) & (h <= 85)

    c = int(np.count_nonzero(colored))
    r = int(np.count_nonzero(red))
    g = int(np.count_nonzero(green))
    return c, r, g


def dominant_color_label(bgr: np.ndarray, alpha_mask: np.ndarray | None) -> str:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if alpha_mask is not None:
        m = alpha_mask > 0
        if not np.any(m):
            return "unknown"
        hsv = hsv[m].reshape(-1, 3)

        # reshape back to 2D-ish for counting via masks
        # easiest: count directly
        h = hsv[:, 0]
        s = hsv[:, 1]
        v = hsv[:, 2]
        colored = (s >= 40) & (v >= 40)
        if colored.sum() < 20:
            return "unknown"
        red = colored & ((h <= 10) | (h >= 170))
        green = colored & (h >= 35) & (h <= 85)
        r = int(red.sum())
        g = int(green.sum())
        if r > g * 1.2:
            return "red"
        if g > r * 1.2:
            return "green"
        return "unknown"

    c, r, g = _count_red_green(hsv, 40, 40)
    if c < 50:
        return "unknown"
    if r > g * 1.2:
        return "red"
    if g > r * 1.2:
        return "green"
    return "unknown"


def candidate_color_ok(candidate_bgr: np.ndarray, want: str, gate_frac: float) -> bool:
    if want == "unknown":
        return True
    hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)
    c, r, g = _count_red_green(hsv, 45, 45)
    if c < 30:
        return False
    rf = r / max(1, c)
    gf = g / max(1, c)
    if want == "red":
        return rf >= gate_frac
    if want == "green":
        return gf >= gate_frac
    return True


# -------------------------
# Edges + mask for matching
# -------------------------
def edges_and_mask(bgr: np.ndarray, alpha_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    k = np.ones((3, 3), np.uint8)
    m = cv2.dilate(edges, k, iterations=1)
    m = (m > 0).astype(np.uint8) * 255

    if alpha_mask is not None:
        m = cv2.bitwise_and(m, alpha_mask)

    # avoid empty masks
    if int(np.count_nonzero(m)) < 20:
        m = np.ones_like(m, dtype=np.uint8) * 255

    return edges, m


def score_patch(patch_bgr: np.ndarray, q_edges: np.ndarray, q_mask: np.ndarray) -> float:
    ph, pw = patch_bgr.shape[:2]
    qh, qw = q_edges.shape[:2]
    if ph < 8 or pw < 8 or qh < 8 or qw < 8:
        return -1.0

    p_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    p_gray = cv2.GaussianBlur(p_gray, (3, 3), 0)
    p_edges = cv2.Canny(p_gray, 50, 150)

    # normalize size to query
    p_edges_r = cv2.resize(p_edges, (qw, qh), interpolation=cv2.INTER_AREA)

    # masked correlation (1x1)
    res = cv2.matchTemplate(p_edges_r, q_edges, cv2.TM_CCORR_NORMED, mask=q_mask)
    return float(res[0, 0])


# -------------------------
# Candidate placard detection (exclude blue callout lines)
# -------------------------
def find_candidate_boxes(
    target_bgr: np.ndarray,
    *,
    sat_min: int,
    val_min: int,
    blue_hmin: int,
    blue_hmax: int,
    min_side: int,
    max_side: int,
    aspect_tol: float,
    fill_min: float,
) -> list[list[int]]:
    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    colored = (s >= sat_min) & (v >= val_min)
    is_blue = (h >= blue_hmin) & (h <= blue_hmax)

    # keep saturated pixels but throw out blue callout lines
    mask = colored & (~is_blue)

    # also keep red even if saturation is slightly lower (thin borders)
    mask_red = ((h <= 10) | (h >= 170)) & (s >= max(25, sat_min - 20)) & (v >= val_min)
    mask = mask | mask_red

    m = (mask.astype(np.uint8) * 255)

    k3 = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[list[int]] = []
    H, W = target_bgr.shape[:2]

    for c in contours:
        x, y, w, h2 = cv2.boundingRect(c)
        if w < min_side or h2 < min_side or w > max_side or h2 > max_side:
            continue

        ar = w / float(h2 + 1e-9)
        if ar < (1.0 - aspect_tol) or ar > (1.0 + aspect_tol):
            continue

        area = float(cv2.contourArea(c))
        fill = area / float(w * h2 + 1e-9)
        if fill < fill_min:
            continue

        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W - 1, x + w + pad)
        y2 = min(H - 1, y + h2 + pad)
        boxes.append([x1, y1, x2, y2])

    # NMS to dedupe
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    scores = np.array([(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in boxes], dtype=np.float32)
    keep = nms(b, scores, 0.35)
    return [boxes[int(i)] for i in keep]


# -------------------------
# NMS
# -------------------------
def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= iou_thresh]

    return np.array(keep, dtype=np.int64)


# -------------------------
# Draw
# -------------------------
def color_for_label(label: str) -> tuple[int, int, int]:
    h = (hash(label) & 0xFFFFFFFF)
    b = 60 + (h & 0x7F)
    g = 60 + ((h >> 8) & 0x7F)
    r = 60 + ((h >> 16) & 0x7F)
    return int(b), int(g), int(r)


def draw_dets(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        label = d["query"]
        score = d["score"]
        color = color_for_label(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_candidates(img: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(
            out,
            str(i),
            (x1, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return out


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="data/targets/target.png")
    ap.add_argument("--queries", default="data/queries")
    ap.add_argument("--out", default="data/outputs/cv")

    ap.add_argument("--score-threshold", type=float, default=0.82)
    ap.add_argument("--nms", type=float, default=0.30)
    ap.add_argument("--max-per-query", type=int, default=50)

    # candidate detection (tune if needed)
    ap.add_argument("--sat-min", type=int, default=45)
    ap.add_argument("--val-min", type=int, default=55)
    ap.add_argument("--blue-hmin", type=int, default=85)
    ap.add_argument("--blue-hmax", type=int, default=135)

    ap.add_argument("--min-side", type=int, default=14)
    ap.add_argument("--max-side", type=int, default=90)
    ap.add_argument("--aspect-tol", type=float, default=0.45)
    ap.add_argument("--fill-min", type=float, default=0.08)

    # hue gating
    ap.add_argument("--hue-gate-frac", type=float, default=0.55)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug-candidates", action="store_true")
    args = ap.parse_args()

    target_path = Path(args.target)
    queries_dir = Path(args.queries)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bgr = imread_color(target_path)

    query_paths = sorted([p for p in queries_dir.glob("*.png") if p.is_file()])
    if not query_paths:
        raise FileNotFoundError(f"No .png queries in {queries_dir}")

    cand_boxes = find_candidate_boxes(
        target_bgr,
        sat_min=int(args.sat_min),
        val_min=int(args.val_min),
        blue_hmin=int(args.blue_hmin),
        blue_hmax=int(args.blue_hmax),
        min_side=int(args.min_side),
        max_side=int(args.max_side),
        aspect_tol=float(args.aspect_tol),
        fill_min=float(args.fill_min),
    )

    print(f"Candidate boxes found: {len(cand_boxes)}")
    if args.debug_candidates:
        cand_overlay = draw_candidates(target_bgr, cand_boxes)
        cv2.imwrite(str(out_dir / "candidate_boxes_overlay.png"), cand_overlay)

    all_dets: list[dict] = []
    by_query: dict[str, dict] = {}
    combined_overlay = target_bgr.copy()

    for qp in tqdm(query_paths, desc="Queries", unit="img"):
        q_bgr, q_alpha = imread_query(qp)
        q_bgr, q_alpha = trim_template(q_bgr, q_alpha)

        want = dominant_color_label(q_bgr, q_alpha)
        q_edges, q_mask = edges_and_mask(q_bgr, q_alpha)

        dets: list[dict] = []
        for (x1, y1, x2, y2) in cand_boxes:
            patch = target_bgr[y1:y2, x1:x2]
            if not candidate_color_ok(patch, want, float(args.hue_gate_frac)):
                continue

            sc = score_patch(patch, q_edges, q_mask)
            if sc >= float(args.score_threshold):
                dets.append(
                    {
                        "query": qp.stem,
                        "score": float(sc),
                        "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "query_color": want,
                    }
                )

        dets.sort(key=lambda d: d["score"], reverse=True)
        if len(dets) > int(args.max_per_query):
            dets = dets[: int(args.max_per_query)]

        if dets:
            boxes = np.array([d["box_xyxy"] for d in dets], dtype=np.float32)
            scores = np.array([d["score"] for d in dets], dtype=np.float32)
            keep = nms(boxes, scores, float(args.nms))
            dets = [dets[int(i)] for i in keep]

        by_query[qp.stem] = {
            "count": len(dets),
            "top_score": (dets[0]["score"] if dets else None),
            "query_color": want,
        }
        print(f"{qp.stem}: color={want} hits={len(dets)} top={by_query[qp.stem]['top_score']}")

        per_overlay = draw_dets(target_bgr, dets)
        cv2.imwrite(str(out_dir / f"{qp.stem}__overlay.png"), per_overlay)

        combined_overlay = draw_dets(combined_overlay, dets)
        all_dets.extend(dets)

    all_dets.sort(key=lambda d: d["score"], reverse=True)

    payload = {
        "target": str(target_path),
        "queries_dir": str(queries_dir),
        "params": {
            "score_threshold": float(args.score_threshold),
            "nms_iou": float(args.nms),
            "max_per_query": int(args.max_per_query),
            "candidate": {
                "sat_min": int(args.sat_min),
                "val_min": int(args.val_min),
                "blue_hmin": int(args.blue_hmin),
                "blue_hmax": int(args.blue_hmax),
                "min_side": int(args.min_side),
                "max_side": int(args.max_side),
                "aspect_tol": float(args.aspect_tol),
                "fill_min": float(args.fill_min),
            },
            "hue_gate_frac": float(args.hue_gate_frac),
        },
        "summary": {"total": len(all_dets), "by_query": by_query},
        "detections": all_dets,
    }

    json_path = out_dir / "all_matches.json"
    overlay_path = out_dir / "all_matches_overlay.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cv2.imwrite(str(overlay_path), combined_overlay)

    print(f"TOTAL detections: {len(all_dets)}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {overlay_path}")

    if args.show:
        cv2.imshow("all_matches_overlay", combined_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
