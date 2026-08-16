"""
06_image_comparison.py
-----------------------
Compares two fundus images of the SAME eye taken at different times to
help visualize disease progression.

Pipeline:
  1. Align the follow-up image onto the baseline image (ORB feature
     matching + homography) — corrects for differences in camera angle,
     zoom, or eye position between visits.
  2. Compute an SSIM (structural similarity) map between the aligned
     images — low-SSIM regions indicate areas that changed.
  3. Produce a difference heatmap overlay highlighting the changed regions.
  4. Report a single "change score" (1 - mean SSIM) as a rough progression
     indicator.

NOTE: this is a classical image-processing tool, not a trained model — ODIR
has no longitudinal data to train/validate a learned progression detector.
Frame it in your report as a visualization aid for a clinician, not as a
disease-specific progression classifier.
"""

import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_and_preprocess(path, size=(512, 512)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    return img


def align_images(baseline, followup, max_features=2000, good_match_pct=0.15):
    """Align `followup` onto `baseline` using ORB keypoints + homography."""
    gray1 = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(followup, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    num_good = max(4, int(len(matches) * good_match_pct))
    matches = matches[:num_good]

    if len(matches) < 4:
        # Not enough reliable matches — return the image unaligned rather
        # than crashing, and let the caller know via the returned flag.
        return followup, False

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return followup, False

    aligned = cv2.warpPerspective(followup, H, (baseline.shape[1], baseline.shape[0]))
    return aligned, True


def compute_difference(baseline, aligned_followup):
    gray1 = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(aligned_followup, cv2.COLOR_BGR2GRAY)

    score, diff_map = ssim(gray1, gray2, full=True)
    diff_map = (1 - diff_map)  # invert so higher = more change
    diff_map_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    heatmap = cv2.applyColorMap(diff_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(baseline, 0.6, heatmap, 0.4, 0)

    change_score = 1 - score  # 0 = identical, 1 = completely different
    return change_score, heatmap, overlay


def compare_progression(baseline_path, followup_path, out_prefix="progression"):
    baseline = load_and_preprocess(baseline_path)
    followup = load_and_preprocess(followup_path)

    aligned, ok = align_images(baseline, followup)
    if not ok:
        print("WARNING: could not reliably align images (too few keypoint "
              "matches) — proceeding with unaligned comparison. Results "
              "may be noisy.")

    change_score, heatmap, overlay = compute_difference(baseline, aligned)

    cv2.imwrite(f"{out_prefix}_heatmap.png", heatmap)
    cv2.imwrite(f"{out_prefix}_overlay.png", overlay)

    print(f"Change score (0=no change, 1=max change): {change_score:.4f}")
    print(f"Saved: {out_prefix}_heatmap.png, {out_prefix}_overlay.png")
    return change_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="earlier fundus image")
    ap.add_argument("--followup", required=True, help="later fundus image, same eye")
    ap.add_argument("--out_prefix", default="progression")
    args = ap.parse_args()

    compare_progression(args.baseline, args.followup, args.out_prefix)