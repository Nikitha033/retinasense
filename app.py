"""
app.py
------
RetinaSense — Professional Medical AI Dashboard (Streamlit).

Run locally:
    streamlit run app.py

Deploy for a shareable link: push this folder to a public GitHub repo and
deploy free on https://share.streamlit.io or a Hugging Face Space
(Streamlit SDK), pointed at this file. Use Git LFS for the checkpoint if
it's over 100MB.

Honesty note on the "Performance Metrics" section: it reads real numbers
from models/metrics.json, which 05_evaluate.py writes after you run it on
your test set. Until you've run 05_evaluate.py, that section shows
"Not yet evaluated" rather than a made-up accuracy figure.
"""

import os
import json
import time
import importlib.util
import tempfile

import cv2
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load sibling project modules (numeric-prefixed filenames aren't valid
# Python import names, so we load them by file path instead)
# ---------------------------------------------------------------------------

def _load(module_file, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
ds_mod = _load(os.path.join(_here, "02_dataset.py"), "ds_mod")
model_mod = _load(os.path.join(_here, "03_model.py"), "model_mod")
cmp_mod = _load(os.path.join(_here, "06_image_comparison.py"), "cmp_mod")

get_eval_transforms = ds_mod.get_eval_transforms
RetinaSenseModel = model_mod.RetinaSenseModel
align_images = cmp_mod.align_images
compute_difference = cmp_mod.compute_difference
load_and_preprocess = cmp_mod.load_and_preprocess

DISEASE_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]
DISEASE_NAMES = {
    "N": "Normal", "D": "Diabetic Retinopathy", "G": "Glaucoma",
    "C": "Cataract", "A": "Age-related Macular Degeneration",
    "H": "Hypertensive Retinopathy", "M": "Myopia", "O": "Other",
}
SEVERITY_NAMES = ["No/Unspecified", "Mild", "Moderate", "Severe", "Proliferative"]

DISEASE_INFO = {
    "Diabetic Retinopathy": {
        "what": "Damage to the retina's blood vessels caused by long-term "
                "high blood sugar, ranging from mild non-proliferative "
                "changes to sight-threatening proliferative disease.",
        "symptoms": "Often none in early stages. Later: blurred vision, "
                    "floaters, dark areas of vision, difficulty seeing at night.",
        "treatment": "Blood sugar control, regular eye exams, laser "
                     "photocoagulation, anti-VEGF injections, or vitrectomy "
                     "in advanced cases.",
        "risk_factors": "Duration of diabetes, poor blood sugar control, "
                        "high blood pressure, high cholesterol, pregnancy.",
    },
    "Glaucoma": {
        "what": "A group of conditions that damage the optic nerve, often "
                "linked to elevated pressure inside the eye, that can lead "
                "to irreversible vision loss if untreated.",
        "symptoms": "Usually none until noticeable vision loss occurs "
                    "(peripheral vision goes first); sudden severe cases "
                    "can cause eye pain, headache, blurred vision, halos.",
        "treatment": "Pressure-lowering eye drops, laser therapy, or "
                     "surgery to improve fluid drainage from the eye.",
        "risk_factors": "Age over 60, family history, high eye pressure, "
                        "thin corneas, some medical conditions like diabetes.",
    },
    "Cataract": {
        "what": "Clouding of the eye's natural lens that blurs vision, "
                "usually developing slowly with age.",
        "symptoms": "Cloudy or blurry vision, faded colors, glare "
                    "sensitivity, poor night vision, seeing halos around lights.",
        "treatment": "Early on, updated glasses may help; the definitive "
                     "treatment is outpatient surgery to replace the "
                     "clouded lens with an artificial one.",
        "risk_factors": "Aging, diabetes, smoking, prolonged UV exposure, "
                        "steroid use, previous eye injury or surgery.",
    },
    "Age-related Macular Degeneration": {
        "what": "Deterioration of the macula (central retina), causing "
                "loss of sharp, central vision needed for reading and "
                "recognizing faces.",
        "symptoms": "Blurred or reduced central vision, straight lines "
                    "appearing wavy, difficulty recognizing faces.",
        "treatment": "Anti-VEGF injections for the wet form, nutritional "
                     "supplements for some dry-form cases, low-vision aids.",
        "risk_factors": "Age over 50, smoking, family history, "
                        "cardiovascular disease, obesity.",
    },
    "Hypertensive Retinopathy": {
        "what": "Retinal blood vessel damage caused by long-term high "
                "blood pressure.",
        "symptoms": "Often none until advanced; can include headaches "
                    "and vision changes in severe/acute hypertension.",
        "treatment": "Primarily blood pressure control; regular retinal "
                     "monitoring in patients with chronic hypertension.",
        "risk_factors": "Chronic high blood pressure, long duration of "
                        "hypertension, coexisting diabetes.",
    },
    "Myopia": {
        "what": "Nearsightedness — the eye focuses light in front of the "
                "retina, making distant objects look blurry. High myopia "
                "can cause retinal thinning and stretching.",
        "symptoms": "Blurred distance vision, squinting, eye strain, headaches.",
        "treatment": "Glasses, contact lenses, refractive surgery; high "
                     "myopia needs regular retinal monitoring for "
                     "complications like retinal detachment.",
        "risk_factors": "Family history, extensive near-work, limited "
                        "outdoor time in childhood.",
    },
    "Other": {
        "what": "Findings that don't fit the other specific categories in "
                "this model's training labels (ODIR-5K's 'Other' class "
                "covers a broad mix of miscellaneous retinal findings).",
        "symptoms": "Varies widely depending on the underlying finding.",
        "treatment": "Depends entirely on the specific condition — "
                     "clinical evaluation is needed to identify it.",
        "risk_factors": "Varies widely depending on the underlying finding.",
    },
}

st.set_page_config(page_title="RetinaSense", page_icon="👁", layout="wide")

# ---------------------------------------------------------------------------
# Theme: dark mode, blue accent, rounded cards
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
.card {
    background: linear-gradient(145deg, #1a1f2b, #161a24);
    border: 1px solid #2a3142;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.card h3, .card h4 { margin-top: 0; color: #4da3ff; }
.big-stat { font-size: 2.1rem; font-weight: 700; color: #ffffff; }
.stat-label { font-size: 0.85rem; color: #9aa5b1; text-transform: uppercase; letter-spacing: 0.05em; }
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; margin-left: 8px;
}
.pill-green { background: #14361f; color: #4ade80; }
.pill-yellow { background: #3a3418; color: #facc15; }
.pill-red { background: #3a1818; color: #f87171; }
hr { border-color: #2a3142; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaSenseModel(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict(image_bgr, model, device, threshold):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = get_eval_transforms()
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        disease_logits, severity_logits = model(tensor)
        probs = torch.sigmoid(disease_logits).cpu().numpy()[0]
        severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
    elapsed = time.time() - start

    results = {DISEASE_NAMES[c]: float(probs[i]) for i, c in enumerate(DISEASE_COLS)}
    results_sorted = dict(sorted(results.items(), key=lambda kv: -kv[1]))

    severity_result = None
    if probs[DISEASE_COLS.index("D")] >= threshold:
        sev_idx = int(severity_probs.argmax())
        severity_result = {
            "grade": SEVERITY_NAMES[sev_idx],
            "confidence": float(severity_probs[sev_idx]),
        }
    return results_sorted, severity_result, elapsed


def status_for(prob, threshold):
    """Returns (label, pill_css_class) for the disease status table."""
    if prob >= threshold:
        return "Detected", "pill-red" if prob >= 0.7 else "pill-yellow"
    elif prob >= threshold * 0.5:
        return "Possible", "pill-yellow"
    else:
        return "Low", "pill-green"


def risk_level(change_score):
    if change_score < 0.20:
        return "Minimal Change", "pill-green"
    elif change_score < 0.50:
        return "Moderate Change", "pill-yellow"
    else:
        return "High Progression", "pill-red"


def plot_probability_bars(results, threshold):
    names = list(results.keys())        # already sorted highest -> lowest
    probs = list(results.values())
    colors = ["#4da3ff" if p >= threshold else "#3a4152" for p in probs]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    y_pos = np.arange(len(names))
    ax.barh(y_pos, probs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#e6e6e6")
    ax.tick_params(colors="#e6e6e6")
    ax.set_xlim(0, 1)
    ax.axvline(threshold, color="#f87171", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted probability", color="#e6e6e6")
    for spine in ax.spines.values():
        spine.set_color("#2a3142")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def load_metrics(checkpoint_path):
    metrics_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 👁 RetinaSense")
    st.caption("AI-powered Retinal Disease Screening")
    st.markdown("---")

    st.markdown("**📁 Model**")
    st.markdown("✔ EfficientNet-B0")
    checkpoint_path = st.text_input("Checkpoint", value="models/best_model.pt", label_visibility="collapsed")

    st.markdown("**🎯 Confidence Threshold**")
    threshold = st.slider("threshold", 0.0, 1.0, 0.30, 0.05, label_visibility="collapsed")

    st.markdown("**📌 Screening Mode**")
    mode = st.radio("mode", ["Single Image", "Progression Analysis"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**ℹ Dataset**")
    st.caption("ODIR-5K")

    st.markdown("---")
    st.warning("⚠ Research use only.\nNot intended for clinical diagnosis.")

# ---------------------------------------------------------------------------
# Main page — header
# ---------------------------------------------------------------------------

st.markdown("# RetinaSense AI Dashboard")
st.markdown("Upload a retinal fundus image for automated disease screening.")

if not os.path.isfile(checkpoint_path):
    st.warning(f"Checkpoint not found at `{checkpoint_path}`. Train a model "
               f"first (`04_train.py`) or update the path in the sidebar.")
    st.stop()

model, device = load_model(checkpoint_path)
metrics = load_metrics(checkpoint_path)
last_inference_time = None

# ===========================================================================
# SINGLE IMAGE MODE
# ===========================================================================

if mode == "Single Image":
    uploaded = st.file_uploader("Fundus image (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img_path = save_uploaded_file(uploaded)
        image_bgr = cv2.imread(img_path)
        h, w = image_bgr.shape[:2]

        with st.spinner("Analyzing..."):
            results, severity, elapsed = predict(image_bgr, model, device, threshold)
        last_inference_time = elapsed

        top_name, top_prob = next(iter(results.items()))
        top_status, top_pill = status_for(top_prob, threshold)

        col_img, col_summary = st.columns([1, 1.3])

        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Original Fundus Image")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"**Image Size**  \n{w} × {h} (model input: 224 × 224)")
            st.markdown("**Status**  \n✔ Successfully Loaded")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_summary:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Prediction Summary")
            st.markdown("**Primary Prediction**")
            st.markdown(f"### 🟢 {top_name}" if top_prob >= threshold else f"### 🟡 {top_name}")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="stat-label">Confidence</div><div class="big-stat">{top_prob*100:.0f}%</div>', unsafe_allow_html=True)
            sev_text = severity["grade"] if severity else "—"
            c2.markdown(f'<div class="stat-label">Severity</div><div class="big-stat">{sev_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["Results", "Probability", "Progression", "About Disease"])

        # ---------------- Results tab ----------------
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Top Predictions")
            medals = ["1️⃣", "2️⃣", "3️⃣"]
            for i, (name, prob) in enumerate(list(results.items())[:3]):
                st.markdown(f"{medals[i]} **{name}** — {prob*100:.0f}%")
            if top_prob < threshold:
                st.info(
                    f"**Most Likely Diagnosis:** {top_name}  \n"
                    f"**Confidence:** {top_prob*100:.0f}% — below screening threshold.  \n"
                    f"Clinical examination recommended."
                )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Disease Status Table")
            st.markdown(
                "| Disease | Probability | Status |\n|---|---|---|\n" +
                "\n".join(
                    f"| {name} | {prob*100:.0f}% | "
                    f'<span class="pill {status_for(prob, threshold)[1]}">{status_for(prob, threshold)[0]}</span> |'
                    for name, prob in results.items()
                ),
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### AI Interpretation")
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"**Primary finding**  \n{top_name}")
                st.markdown(f"**Confidence**  \n{top_prob*100:.0f}%")
            with ic2:
                st.markdown(f"**Disease severity**  \n{severity['grade'] if severity else 'Not applicable'}")
                action = "Consult an ophthalmologist." if top_prob >= threshold else "Routine follow-up recommended."
                st.markdown(f"**Suggested Action**  \n{action}")
            st.markdown('</div>', unsafe_allow_html=True)

            if severity:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Diabetic Retinopathy Severity")
                sc1, sc2 = st.columns(2)
                sc1.markdown(f'<div class="stat-label">Grade</div><div class="big-stat">{severity["grade"]}</div>', unsafe_allow_html=True)
                sc2.markdown(f'<div class="stat-label">Confidence</div><div class="big-stat">{severity["confidence"]*100:.0f}%</div>', unsafe_allow_html=True)
                st.caption("Severity is derived from ODIR's diagnostic-keyword text during "
                           "training and is validated only for DR, not other diseases.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### DR Severity")
                st.markdown("**Not Applicable** — Diabetic Retinopathy probability is below threshold.")
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- Probability tab ----------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Disease Probability Chart")
            st.pyplot(plot_probability_bars(results, threshold))
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- Progression tab (placeholder in single-image mode) ----------------
        with tab3:
            st.info("Switch the sidebar to **Progression Analysis** and upload a "
                    "baseline + follow-up image to compare disease change over time.")

        # ---------------- About Disease tab ----------------
        with tab4:
            for name, info in DISEASE_INFO.items():
                with st.expander(f"{name}"):
                    st.markdown(f"**What is it?**  \n{info['what']}")
                    st.markdown(f"**Symptoms**  \n{info['symptoms']}")
                    st.markdown(f"**Treatment**  \n{info['treatment']}")
                    st.markdown(f"**Risk factors**  \n{info['risk_factors']}")

        os.remove(img_path)

# ===========================================================================
# PROGRESSION ANALYSIS MODE
# ===========================================================================

else:
    c1, c2 = st.columns(2)
    with c1:
        baseline_file = st.file_uploader("Baseline (earlier) image", type=["jpg", "jpeg", "png"], key="baseline")
    with c2:
        followup_file = st.file_uploader("Follow-up (later) image", type=["jpg", "jpeg", "png"], key="followup")

    if baseline_file and followup_file:
        baseline_path = save_uploaded_file(baseline_file)
        followup_path = save_uploaded_file(followup_file)

        baseline_img = load_and_preprocess(baseline_path)
        followup_img = load_and_preprocess(followup_path)

        with st.spinner("Aligning images and computing change map..."):
            aligned, ok = align_images(baseline_img, followup_img)
            if not ok:
                st.warning("Could not reliably align the two images (too few "
                           "matching features) — showing an unaligned comparison.")
            change_score, heatmap, overlay = compute_difference(baseline_img, aligned)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB), caption="Baseline", use_container_width=True)
        with colB:
            st.image(cv2.cvtColor(followup_img, cv2.COLOR_BGR2RGB), caption="Follow-up", use_container_width=True)
        with colC:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Heatmap", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        level_label, level_pill = risk_level(change_score)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        pc1.markdown(f'<div class="stat-label">Progression Score</div><div class="big-stat">{change_score:.2f}</div>', unsafe_allow_html=True)
        pc2.markdown(f'<div class="stat-label">Risk Level</div><div class="big-stat">{level_label}</div>', unsafe_allow_html=True)
        st.caption(
            "0.00 – 0.20 Minimal Change  •  0.20 – 0.50 Moderate Change  •  0.50 – 1.00 High Progression"
        )
        st.caption(
            "This is a classical image-registration + structural-similarity "
            "comparison, not a trained progression classifier — ODIR-5K has "
            "no same-patient, multiple-time-point data to train one on. "
            "Treat this as a visualization aid, not a diagnosis of worsening."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Running disease predictions on both images..."):
            results_base, sev_base, elapsed_base = predict(baseline_img, model, device, threshold)
            results_follow, sev_follow, elapsed_follow = predict(followup_img, model, device, threshold)
        last_inference_time = (elapsed_base + elapsed_follow) / 2

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Disease Probability Comparison")
        names = list(results_base.keys())
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, [results_base[n] for n in names], width, label="Baseline", color="#3a4152")
        ax.bar(x + width / 2, [results_follow[n] for n in names], width, label="Follow-up", color="#4da3ff")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", color="#e6e6e6")
        ax.tick_params(colors="#e6e6e6")
        ax.axhline(threshold, color="#f87171", linestyle="--", linewidth=1)
        ax.legend(facecolor="#161a24", labelcolor="#e6e6e6")
        for spine in ax.spines.values():
            spine.set_color("#2a3142")
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        os.remove(baseline_path)
        os.remove(followup_path)
    else:
        st.info("Upload both a baseline and a follow-up image to run the progression comparison.")

# ===========================================================================
# Performance metrics + footer (always visible)
# ===========================================================================

st.markdown("---")
st.markdown("#### Performance Metrics")

if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macro AUC", f"{metrics['macro_auc']*100:.0f}%")
    m2.metric("Diseases", "8")
    m3.metric("DR Grades", "5")
    dr_acc = metrics.get("dr_severity_accuracy")
    m4.metric("DR Severity Acc.", f"{dr_acc*100:.0f}%" if dr_acc is not None else "N/A")
else:
    st.info("Not yet evaluated — run `05_evaluate.py` on your test set to populate real "
            "accuracy/AUC/F1/Kappa figures here (saved to `models/metrics.json`).")

st.markdown("---")
f1, f2, f3, f4 = st.columns(4)
f1.markdown(f"**Model**  \nEfficientNet-B0")
f2.markdown(f"**Dataset**  \nODIR-5K")
f3.markdown(f"**Framework**  \nPyTorch")
f4.markdown(f"**Inference Time**  \n{last_inference_time:.2f} sec" if last_inference_time is not None else "**Inference Time**  \n—")