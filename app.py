"""
app.py
------
RetinaSense — Intelligent Clinical Retinal Screening, Multi-Disease Severity Staging,
Longitudinal Progression Tracking, and Clinical Decision Support System.

Run locally:
    streamlit run app.py
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
# Page Configuration & Metadata
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RetinaSense AI — Retinal Intelligence & Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Load sibling project modules
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

DISEASE_COLS = ["D", "G", "C", "A", "H", "M"]
DISEASE_NAMES = {
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertensive Retinopathy",
    "M": "Myopia",
}
DISEASE_ICONS = {
    "D": "🩸",
    "G": "👁️",
    "C": "🌫️",
    "A": "🎯",
    "H": "💓",
    "M": "👓",
}
DR_SEVERITY_NAMES = ["Unspecified DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# ---------------------------------------------------------------------------
# Comprehensive Clinical Knowledge Base
# ---------------------------------------------------------------------------

SEVERITY_KNOWLEDGE_BASE = {
    "D": {
        "disease_name": "Diabetic Retinopathy",
        "scale_name": "International Clinical Diabetic Retinopathy Scale (ICDR)",
        "grades": {
            0: {
                "name": "Unspecified / Early DR",
                "urgency": "Semi-Annual (3–6 Months)",
                "pill": "badge-yellow",
                "description": "Retinal microvascular alterations detected without definitive proliferative neovascularization.",
                "workup": ["Macular Optical Coherence Tomography (OCT)", "Dilated Fundus Biomicroscopy", "HbA1c & Fasting Glucose Profile"],
                "management": ["Strict glycemic control (target HbA1c < 7.0%)", "Blood pressure regulation (< 130/80 mmHg)", "Annual or semi-annual retinal screening"],
                "red_flags": ["Sudden vision blurring", "New onset floaters or dark spots", "Night vision reduction"],
            },
            1: {
                "name": "Mild Non-Proliferative DR (Mild NPDR)",
                "urgency": "Routine Follow-up (12 Months)",
                "pill": "badge-green",
                "description": "Presence of microaneurysms only in one or more retinal quadrants without macular edema.",
                "workup": ["Macular OCT to rule out Diabetic Macular Edema (DME)", "Fundus Photography Documentation", "Serum Lipid Profile & HbA1c"],
                "management": ["Intensive glycemic and blood pressure optimization", "Counseling on healthy dietary habits and exercise", "Annual dilated ophthalmic review"],
                "red_flags": ["Distorted central vision", "Difficulty reading small text"],
            },
            2: {
                "name": "Moderate Non-Proliferative DR (Moderate NPDR)",
                "urgency": "Close Monitoring (3–6 Months)",
                "pill": "badge-yellow",
                "description": "Multiple microaneurysms, dot-and-blot intraretinal hemorrhages, hard lipid exudates, and cotton-wool spots.",
                "workup": ["High-resolution Spectral-Domain OCT (SD-OCT)", "Fluorescein Angiography (FFA) if center-involving DME suspected", "Renal function panel (eGFR, microalbuminuria)"],
                "management": ["Specialist retinal evaluation within 4–6 weeks", "Anti-VEGF intravitreal therapy if center-involving DME confirmed", "Rigorous systemic cardiovascular and diabetes control"],
                "red_flags": ["Distorted straight lines (metamorphopsia)", "Rapid drop in visual clarity"],
            },
            3: {
                "name": "Severe Non-Proliferative DR (Severe NPDR)",
                "urgency": "Urgent Specialist Referral (2–4 Weeks)",
                "pill": "badge-red",
                "description": "Meets 4-2-1 international rule: >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.",
                "workup": ["Urgent Widefield Fluorescein Angiography (FFA)", "Macular SD-OCT with retinal thickness mapping", "Cardiovascular and Nephrology co-management"],
                "management": ["Urgent retinal specialist evaluation for Panretinal Photocoagulation (PRP) laser or prophylactic Anti-VEGF injections", "High risk of rapid progression to Proliferative DR"],
                "red_flags": ["Sudden onset of dark cobweb floaters", "Partial visual field obscuration"],
            },
            4: {
                "name": "Proliferative Diabetic Retinopathy (PDR)",
                "urgency": "Emergent Intervention (< 1 Week)",
                "pill": "badge-red",
                "description": "Severe sight-threatening neovascularization of the disc (NVD) or elsewhere (NVE), preretinal/vitreous hemorrhage, or fibrovascular proliferation.",
                "workup": ["Emergency Retinal Specialist Biomicroscopy", "B-scan Ocular Ultrasonography if media obscured by vitreous hemorrhage", "Urgent Fluorescein Angiography"],
                "management": ["Immediate Panretinal Photocoagulation (PRP) laser", "Intravitreal Anti-VEGF injections (Aflibercept / Ranibizumab)", "Pars plana vitrectomy for non-clearing vitreous hemorrhage or tractional retinal detachment"],
                "red_flags": ["Sudden painless profound vision loss", "Dense dark curtain over the vision", "Shower of flashes and floaters"],
            },
        },
    },
    "G": {
        "disease_name": "Glaucoma",
        "scale_name": "Hodapp-Parrish-Anderson / Cup-to-Disc (CDR) Staging",
        "grades": {
            1: {
                "name": "Mild / Glaucoma Suspect",
                "urgency": "Semi-Annual Review (6 Months)",
                "pill": "badge-green",
                "description": "Early optic disc cupping (CDR 0.5–0.6), subtle neuroretinal rim thinning, or borderline intraocular pressure.",
                "workup": ["Goldmann Applanation Tonometry (IOP)", "Central Corneal Thickness (Pachymetry)", "Humphrey Visual Field 24-2 (SITA Standard)", "Peripapillary RNFL OCT"],
                "management": ["Baseline structural and functional mapping", "Initiate topical prostaglandin analogs if IOP consistently elevated (> 21 mmHg)", "6-month visual field monitoring"],
                "red_flags": ["Halos around lights", "Mild ocular aching in dim environments"],
            },
            2: {
                "name": "Moderate Glaucoma",
                "urgency": "Close Clinical Care (2–3 Months)",
                "pill": "badge-yellow",
                "description": "Definite optic disc excavation (CDR 0.7–0.8), localized neuroretinal rim notch, and nasal step or arcuate visual field defect.",
                "workup": ["Serial RNFL & Ganglion Cell Complex (GCC) OCT scans", "Humphrey Visual Field 24-2 & 10-2 testing", "Gonioscopy for anterior chamber angle assessment"],
                "management": ["Target IOP reduction of 30–40% from baseline", "Dual-agent topical therapy (Prostaglandin + Beta-blocker / Alpha-agonist)", "Consider Selective Laser Trabeculoplasty (SLT)"],
                "red_flags": ["Noticeable dark patches in peripheral vision", "Frequent prescription changes without clarity improvement"],
            },
            3: {
                "name": "Severe / Advanced Glaucoma",
                "urgency": "Urgent Surgical Consultation (1–2 Weeks)",
                "pill": "badge-red",
                "description": "Profound optic disc excavation (CDR > 0.85), near-total neuroretinal rim loss, tubular/tunnel vision threatening fixation.",
                "workup": ["High-density 10-2 Visual Field testing", "Macular GCC analysis for remaining central photoreceptors", "Diurnal IOP curve monitoring"],
                "management": ["Maximum tolerated medical therapy", "Urgent surgical intervention: Trabeculectomy with Mitomycin C or Glaucoma Drainage Device (Ahmed/Baerveldt valve)", "Low vision rehabilitation support"],
                "red_flags": ["Loss of central reading vision", "Severe eye pain with nausea (acute angle closure crisis)"],
            },
        },
    },
    "C": {
        "disease_name": "Cataract",
        "scale_name": "LOCS III / Optical Opacification Grading",
        "grades": {
            1: {
                "name": "Mild / Early Cataract",
                "urgency": "Annual Monitoring (12 Months)",
                "pill": "badge-green",
                "description": "Incipient nuclear/cortical lens opacity with minimal obscuration of retinal details.",
                "workup": ["Slit-lamp Biomicroscopy with dilated lens examination", "Best-Corrected Visual Acuity (BCVA) Refraction", "Glare sensitivity testing"],
                "management": ["Updated refractive spectacle prescription", "Anti-reflective coatings and UV-protective sunglasses", "Patient education on cataract progression"],
                "red_flags": ["Glare during night driving", "Mild double vision in one eye (monocular diplopia)"],
            },
            2: {
                "name": "Moderate Cataract",
                "urgency": "Elective Surgical Review (3–6 Months)",
                "pill": "badge-yellow",
                "description": "Moderate nuclear sclerosis or posterior subcapsular haze significantly dampening contrast and fundus clarity.",
                "workup": ["Slit-lamp Cataract Staging", "Corneal Endothelial Specular Microscopy", "Optical Biometry (IOL Master / Lenstar) for intraocular lens calculation"],
                "management": ["Evaluation of functional visual impairment on daily activities (driving, reading)", "Schedule elective outpatient Phacoemulsification with intraocular lens (IOL) implantation", "Pre-operative retinal assessment"],
                "red_flags": ["Inability to pass driver visual requirements", "Severe daytime light sensitivity"],
            },
            3: {
                "name": "Mature / Severe Cataract",
                "urgency": "Prompt Surgical Management (2–4 Weeks)",
                "pill": "badge-red",
                "description": "Dense, brunescent or hypermature white cataract completely obstructing retinal visualization and severely reducing visual acuity (< 20/200).",
                "workup": ["Ocular B-scan Ultrasound (to evaluate posterior pole through dense media)", "Immersion A-scan Biometry", "Anterior segment OCT"],
                "management": ["Urgent Phacoemulsification or Extracapsular Cataract Extraction (ECCE)", "Implantation of Monofocal/Toric/Multifocal Intraocular Lens", "Risk counseling for phacolytic or phacomorphic secondary glaucoma"],
                "red_flags": ["Sudden acute eye redness and high IOP (phacomorphic crisis)", "Complete loss of form vision (light perception only)"],
            },
        },
    },
    "A": {
        "disease_name": "Age-related Macular Degeneration (AMD)",
        "scale_name": "AREDS Clinical Classification of AMD",
        "grades": {
            1: {
                "name": "Early AMD",
                "urgency": "Annual Dilated Exam (12 Months)",
                "pill": "badge-green",
                "description": "Presence of multiple small drusen (< 63 µm) or few intermediate drusen (63–124 µm) with no pigmentary abnormalities.",
                "workup": ["Macular SD-OCT", "High-contrast Color Fundus Photography", "Amsler Grid baseline training"],
                "management": ["Daily Amsler grid self-monitoring at home", "Smoking cessation counseling (strongest modifiable risk factor)", "Mediterranean diet rich in green leafy vegetables and omega-3 fatty acids"],
                "red_flags": ["Straight lines appearing wavy or bent on Amsler grid", "Central blank spot"],
            },
            2: {
                "name": "Intermediate Dry AMD",
                "urgency": "Quarterly Monitoring (3–6 Months)",
                "pill": "badge-yellow",
                "description": "Extensive intermediate drusen (63–124 µm) or at least one large confluent drusen (≥ 125 µm) or geographic atrophy not involving foveal center.",
                "workup": ["Serial Macular SD-OCT (monitoring for subretinal fluid / neovascularization)", "Fundus Autofluorescence (FAF) to map RPE atrophy", "Amsler Grid checking"],
                "management": ["Prescribe AREDS2 antioxidant formula (Vitamin C 500mg, Vitamin E 400IU, Lutein 10mg, Zeaxanthin 2mg, Zinc 80mg, Copper 2mg)", "Close surveillance for wet AMD conversion"],
                "red_flags": ["Metamorphopsia (door frames or tiles looking warped)", "Sudden central gray smudge"],
            },
            3: {
                "name": "Advanced / Neovascular (Wet) AMD",
                "urgency": "Emergent Specialist Care (< 1 Week)",
                "pill": "badge-red",
                "description": "Choroidal Neovascularization (CNV) with subretinal fluid, intraretinal cystoid edema, subretinal hemorrhage, or central geographic atrophy involving fovea.",
                "workup": ["Emergency Macular OCT with Angiography (OCT-A)", "Fluorescein Angiography (FFA) & Indocyanine Green Angiography (ICGA)", "Quantitative central subfield thickness measurement"],
                "management": ["Immediate Intravitreal Anti-VEGF injections (Faricimab / Aflibercept / Ranibizumab) on treat-and-extend regimen", "Consider Complement C3/C5 inhibitors (Pegcetacoplan/Avacincaptad pegol) for geographic atrophy", "Low vision optical aids"],
                "red_flags": ["Rapid, sudden drop in central reading vision", "Dark static central blind spot expanding"],
            },
        },
    },
    "H": {
        "disease_name": "Hypertensive Retinopathy",
        "scale_name": "Keith-Wagener-Barker (KWB) Severity Classification",
        "grades": {
            1: {
                "name": "Grade 1 (Mild)",
                "urgency": "Primary Care Follow-up (1–3 Months)",
                "pill": "badge-green",
                "description": "Generalized retinal arteriolar narrowing, increased light reflex, and mild vascular tortuosity.",
                "workup": ["Ambulatory 24-Hour Blood Pressure Monitoring", "Basic Metabolic Panel (Creatinine, BUN, Electrolytes)", "Urinalysis for proteinuria"],
                "management": ["Lifestyle modification (DASH diet, sodium reduction < 2g/day, aerobic exercise)", "Initiation or titration of anti-hypertensive therapy (ACEi / ARB / CCB)", "Annual funduscopic review"],
                "red_flags": ["Persistent morning occipital headaches", "Occasional visual dimming"],
            },
            2: {
                "name": "Grade 2 (Moderate)",
                "urgency": "Medical Cardiology Review (2–4 Weeks)",
                "pill": "badge-yellow",
                "description": "Focal arteriolar spasm, Arteriovenous (AV) nicking/compression (Salus/Gunn sign), and copper wire arterioles.",
                "workup": ["Echocardiography (assessing left ventricular hypertrophy)", "Carotid Doppler Ultrasound", "Electrocardiogram (ECG) and Renal Doppler"],
                "management": ["Dual-agent anti-hypertensive regimen to reach target BP < 130/80 mmHg", "Comprehensive cardiovascular risk assessment (statins if indicated)", "Ophthalmology follow-up in 3 months"],
                "red_flags": ["Dizziness upon standing", "Transient visual darkening (amaurosis fugax)"],
            },
            3: {
                "name": "Grade 3 (Severe)",
                "urgency": "Urgent Medical Evaluation (24–48 Hours)",
                "pill": "badge-red",
                "description": "Flame-shaped retinal hemorrhages, cotton-wool spots (retinal nerve fiber ischemia), hard exudates radiating from fovea (macular star).",
                "workup": ["Urgent Blood Pressure and End-Organ Damage Evaluation", "Macular SD-OCT to assess neurosensory detachment and exudation", "Brain MRI/CT if neurological symptoms present"],
                "management": ["Urgent gradual blood pressure control to prevent stroke or ischemic optic neuropathy", "Hospital or cardiology clinic co-management", "Retina evaluation to monitor macular star resolution"],
                "red_flags": ["Severe pulsatile headache", "Chest pain or shortness of breath", "Sudden central blur"],
            },
            4: {
                "name": "Grade 4 (Malignant / Hypertensive Crisis)",
                "urgency": "Emergency Department Admission (IMMEDIATE)",
                "pill": "badge-red",
                "description": "Grade 3 signs plus bilateral Papilledema (optic disc edema with blurred margins). Represents a hypertensive medical emergency (BP often > 180/120 mmHg).",
                "workup": ["Immediate Emergency Room Triage", "Intravenous Arterial Line BP Monitoring", "Urgent Neuroimaging (Brain MRI/CT) and Cardiac Biomarkers (Troponin, BNP)"],
                "management": ["ICU admission with IV antihypertensives (Nicardipine / Labetalol / Nitroprusside)", "Controlled BP reduction by 20–25% in first hour to prevent cerebral hypoperfusion", "Continuous neuro-ophthalmic monitoring"],
                "red_flags": ["Altered mental status / confusion", "Severe intractable headache, nausea, and vomiting", "Sudden profound bilateral visual loss"],
            },
        },
    },
    "M": {
        "disease_name": "Pathological Myopia",
        "scale_name": "META-PM International Classification of Myopic Maculopathy",
        "grades": {
            1: {
                "name": "Category 1: Tessellated Fundus",
                "urgency": "Annual Eye Exam (12 Months)",
                "pill": "badge-green",
                "description": "Tigroid/tessellated fundus pattern with clearly visible choroidal vessels due to retinal thinning and axial elongation.",
                "workup": ["Ocular Axial Length Biometry", "Dilated Peripheral Retinal Funduscopy", "Best-Corrected Visual Acuity Refraction"],
                "management": ["Accurate optical correction (spectacles/contact lenses)", "Patient education on symptoms of retinal tears/detachment", "Annual peripheral retinal examination with 3-mirror Goldmann lens"],
                "red_flags": ["Flashes of light (photopsia) in dark environments", "Sudden increase in floating cobwebs"],
            },
            2: {
                "name": "Category 2: Diffuse Chorioretinal Atrophy",
                "urgency": "Semi-Annual Exam (6 Months)",
                "pill": "badge-yellow",
                "description": "Diffuse yellowish-white chorioretinal atrophy around the optic disc (peripapillary atrophy/crescent) and posterior pole.",
                "workup": ["Macular SD-OCT (ruling out myopic foveoschisis/traction)", "Widefield Fundus Imaging", "Axial length surveillance"],
                "management": ["Avoid high-impact contact sports to minimize retinal detachment risk", "Regular 6-month dilated examination of vitreoretinal interface", "Amsler grid home monitoring"],
                "red_flags": ["Straight lines appearing distorted", "Persistent dark shadow in peripheral vision"],
            },
            3: {
                "name": "Category 3: Severe Pathological Myopia / Maculopathy",
                "urgency": "Urgent Retinal Specialist Care (1–2 Weeks)",
                "pill": "badge-red",
                "description": "Patchy chorioretinal atrophy, macular lacquer cracks, Foster-Fuchs pigmented spot, posterior staphyloma, or myopic choroidal neovascularization (mCNV).",
                "workup": ["High-resolution Macular OCT-Angiography (OCT-A)", "Fluorescein Angiography to detect active leakage from mCNV", "Swept-Source OCT for posterior staphyloma contour"],
                "management": ["Intravitreal Anti-VEGF injections for active myopic CNV", "Prophylactic 360° laser barrier / cryopexy for peripheral retinal holes/tears", "Surgical vitrectomy / macular buckle if myopic foveoschisis or macular hole present"],
                "red_flags": ["Curtain dropping over the eye (retinal detachment emergency)", "Sudden central distorted dark blind spot"],
            },
        },
    },
}

NORMAL_CLINICAL_PROTOCOL = {
    "status": "Healthy / Normal Fundus",
    "urgency": "Routine Screening (12–24 Months)",
    "pill": "badge-green",
    "description": "No evidence of diabetic retinopathy, glaucoma cupping, cataract obscuration, macular degeneration, hypertensive changes, or pathological myopia.",
    "workup": ["Routine non-mydriatic fundus photography", "Visual acuity testing", "Intraocular pressure check"],
    "management": ["Maintain balanced nutrition rich in lutein and antioxidants", "Annual or biennial comprehensive routine eye examination", "UV-blocking eyewear during outdoor activities"],
    "red_flags": ["Sudden changes in vision", "Ocular trauma", "Onset of new floaters or flashes"],
}

# ---------------------------------------------------------------------------
# Beautiful Custom CSS Theme (Aesthetic Clinical Glassmorphism)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Main Background */
.stApp {
    background: radial-gradient(circle at 10% 10%, #111827 0%, #080c14 55%, #04070d 100%);
    color: #e2e8f0;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Glass Cards */
.glass-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
    box-shadow: 0 14px 35px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.08);
}

/* Hero Header */
.hero-container {
    padding: 6px 0 18px 0;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #38bdf8 75%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
    margin-bottom: 4px;
}
.hero-subtitle {
    font-size: 1.02rem;
    color: #94a3b8;
    font-weight: 400;
    max-width: 860px;
    line-height: 1.5;
}

/* Stats and Badges */
.stat-value {
    font-size: 2.0rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.02em;
}
.stat-sub {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 2px;
}

/* Glowing Badges */
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
.badge-yellow { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
.badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }
.badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }

/* Clinical Boxes */
.rec-box {
    background: rgba(15, 23, 42, 0.8);
    border-left: 4px solid #38bdf8;
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.red-flag-box {
    background: rgba(38, 15, 23, 0.7);
    border-left: 4px solid #f87171;
    padding: 14px 18px;
    border-radius: 10px;
    margin-top: 12px;
}

/* Progress bar styling */
.prog-bar-container {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0b0f17 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    color: #94a3b8;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #38bdf8 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* Clinical Footer */
.clinical-footer {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 18px 24px;
    margin-top: 24px;
}

hr { border-color: rgba(255, 255, 255, 0.08); margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading & helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaSenseModel(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def load_calibrated_thresholds(checkpoint_path):
    th_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "thresholds.json")
    if os.path.isfile(th_path):
        try:
            with open(th_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {d: 0.30 for d in DISEASE_COLS}


def generate_gradcam(model, image_bgr, target_disease_idx):
    """Computes Grad-CAM saliency overlay for a target disease class."""
    device = next(model.parameters()).device
    model.eval()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = get_eval_transforms()
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    features = []
    gradients = []
    target_layer = model.backbone[-1]

    def forward_hook(module, inp, outp):
        features.append(outp)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    disease_logits, _ = model(tensor)
    score = disease_logits[0, target_disease_idx]

    model.zero_grad()
    score.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if not features or not gradients:
        return image_rgb

    feat = features[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(feat.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * feat[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.65, heatmap_rgb, 0.35, 0)
    return overlay


def classify_disease_severity(disease_code, prob, threshold, dr_sev_idx=None, dr_sev_conf=None):
    """Determines exact severity level, clinical staging, and structured recommendations for ANY disease."""
    info = SEVERITY_KNOWLEDGE_BASE.get(disease_code, {})
    grades_dict = info.get("grades", {})

    if disease_code == "D":
        grade_key = dr_sev_idx if (dr_sev_idx is not None and dr_sev_idx in grades_dict) else (
            4 if prob > 0.80 else (3 if prob > 0.65 else (2 if prob > 0.45 else 1))
        )
        conf = dr_sev_conf if dr_sev_conf is not None else prob
    elif disease_code == "H":
        ratio = prob / max(threshold, 0.01)
        if ratio >= 2.5:
            grade_key = 4
        elif ratio >= 1.8:
            grade_key = 3
        elif ratio >= 1.2:
            grade_key = 2
        else:
            grade_key = 1
        conf = min(0.99, prob * 1.1)
    else:
        ratio = prob / max(threshold, 0.01)
        if ratio >= 2.0:
            grade_key = 3
        elif ratio >= 1.3:
            grade_key = 2
        else:
            grade_key = 1
        conf = min(0.99, prob * 1.1)

    stage_data = grades_dict.get(grade_key, list(grades_dict.values())[0])
    return {
        "disease_code": disease_code,
        "disease_name": info.get("disease_name", DISEASE_NAMES[disease_code]),
        "scale_name": info.get("scale_name", ""),
        "grade_key": grade_key,
        "grade_name": stage_data["name"],
        "urgency": stage_data["urgency"],
        "pill": stage_data["pill"],
        "confidence": float(conf),
        "description": stage_data["description"],
        "workup": stage_data["workup"],
        "management": stage_data["management"],
        "red_flags": stage_data["red_flags"],
    }


def predict(image_bgr, model, device, thresholds_dict):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = get_eval_transforms()
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        disease_logits, severity_logits = model(tensor)
        probs = torch.sigmoid(disease_logits).cpu().numpy()[0]
        severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
    elapsed = time.time() - start

    results = {
        DISEASE_NAMES[c]: float(probs[i])
        for i, c in enumerate(DISEASE_COLS)
    }
    results_sorted = dict(sorted(results.items(), key=lambda kv: -kv[1]))

    dr_idx = DISEASE_COLS.index("D")
    dr_prob = probs[dr_idx]
    dr_th = thresholds_dict.get("D", 0.30)
    dr_sev_idx = int(severity_probs.argmax()) if dr_prob >= dr_th else None
    dr_sev_conf = float(severity_probs[dr_sev_idx]) if dr_sev_idx is not None else None

    # Multi-disease severity evaluation for all detected diseases
    severity_evaluations = {}
    for i, code in enumerate(DISEASE_COLS):
        p = probs[i]
        th = thresholds_dict.get(code, 0.30)
        if p >= th:
            sev_info = classify_disease_severity(
                disease_code=code,
                prob=p,
                threshold=th,
                dr_sev_idx=dr_sev_idx if code == "D" else None,
                dr_sev_conf=dr_sev_conf if code == "D" else None,
            )
            severity_evaluations[code] = sev_info

    return results_sorted, probs, severity_evaluations, elapsed


def risk_level(change_score):
    if change_score < 0.20:
        return "Minimal Change (Stable)", "badge-green"
    elif change_score < 0.50:
        return "Moderate Progression", "badge-yellow"
    else:
        return "High Progression (Significant)", "badge-red"


def plot_probability_bars(results, thresholds_dict):
    names = list(results.keys())
    probs = list(results.values())
    
    colors = []
    for name, p in results.items():
        code = [c for c, n in DISEASE_NAMES.items() if n == name][0]
        th = thresholds_dict.get(code, 0.30)
        colors.append("#f87171" if p >= th else "#38bdf8")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, probs, color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#e2e8f0", fontsize=10.5)
    ax.tick_params(colors="#94a3b8")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Predicted Probability", color="#94a3b8", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(axis="x", linestyle="--", alpha=0.18, color="#94a3b8")
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%",
                va="center", ha="left", color="#ffffff", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_progression_comparison_bars(baseline_probs, followup_probs):
    """Grouped bar chart comparing baseline vs follow-up risk probabilities for each disease."""
    diseases = [DISEASE_NAMES[c] for c in DISEASE_COLS]
    base_vals = [baseline_probs[i] * 100 for i in range(len(DISEASE_COLS))]
    fup_vals = [followup_probs[i] * 100 for i in range(len(DISEASE_COLS))]

    x = np.arange(len(diseases))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")

    rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline Visit", color="#38bdf8", edgecolor="none")
    rects2 = ax.bar(x + width/2, fup_vals, width, label="Follow-up Visit", color="#f97316", edgecolor="none")

    ax.set_ylabel("Disease Risk Probability (%)", color="#94a3b8", fontsize=10.5)
    ax.set_title("Disease Risk Comparison: Baseline vs. Follow-up", color="#ffffff", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, color="#e2e8f0", rotation=18, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 108)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    fig.tight_layout()
    return fig


def plot_disease_progression_rate_bars(deltas_dict):
    """Horizontal bar chart showing the progression rate (delta % change) for each disease."""
    diseases = list(deltas_dict.keys())
    deltas = [deltas_dict[d] * 100 for d in diseases]

    colors = []
    for d in deltas:
        if d > 5.0:
            colors.append("#ef4444")
        elif d > 0.0:
            colors.append("#f59e0b")
        elif d < -5.0:
            colors.append("#10b981")
        else:
            colors.append("#64748b")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")

    y_pos = np.arange(len(diseases))
    bars = ax.barh(y_pos, deltas, color=colors, height=0.55, edgecolor="none")

    ax.axvline(0, color="#94a3b8", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diseases, color="#e2e8f0", fontsize=10.5)
    ax.tick_params(colors="#94a3b8")
    ax.set_xlabel("Disease Progression Rate Δ (% Change)", color="#94a3b8", fontsize=10.5)
    ax.set_title("Per-Disease Progression Rate (Follow-up vs Baseline)", color="#ffffff", fontsize=12.5, fontweight="bold", pad=12)

    for bar in bars:
        val = bar.get_width()
        sign = "+" if val > 0 else ""
        offset = 1.0 if val >= 0 else -1.0
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{sign}{val:.1f}%",
                va="center", ha=ha, color="#ffffff", fontsize=9.5, fontweight="bold")

    max_abs = max([abs(d) for d in deltas] + [15.0]) * 1.3
    ax.set_xlim(-max_abs, max_abs)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(axis="x", linestyle="--", alpha=0.2, color="#94a3b8")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Sidebar Configuration (User-Focused & Clean)
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "models/best_model.pt"

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 2.3rem;">👁️</span>
        <div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">RetinaSense</div>
            <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 600; letter-spacing: 0.04em;">AI RETINAL SCREENING</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("##### 🧭 Clinical Screening Mode")
    mode = st.radio(
        "Mode Selection",
        ["Single Image Screening & Severity Staging", "Longitudinal Progression Tracking"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### 🔬 Covered Eye Conditions")
    for code in DISEASE_COLS:
        st.markdown(f"**{DISEASE_ICONS[code]} {DISEASE_NAMES[code]}**")

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 10px 12px; border-radius: 6px; font-size: 0.78rem; color: #bae6fd; margin-bottom: 10px;">
        💡 <strong>Instant Screening:</strong> Upload any retinal fundus photograph to receive instant AI diagnostic triage and management plans.
    </div>
    <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; padding: 10px 12px; border-radius: 6px; font-size: 0.78rem; color: #fbbf24;">
        ⚠️ <strong>Clinical Notice:</strong> RetinaSense is an intelligent decision-support assistant. All findings require specialist clinical confirmation.
    </div>
    """, unsafe_allow_html=True)

# Auto-load model and calibrated thresholds quietly in background
checkpoint_path = DEFAULT_CHECKPOINT
calibrated_th = load_calibrated_thresholds(checkpoint_path)
active_thresholds = calibrated_th

# ---------------------------------------------------------------------------
# Main Application Header (User-Focused)
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div style="display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;">
        <span class="badge-chip badge-blue">🛡️ Multi-Disease Detection</span>
        <span class="badge-chip badge-purple">🔬 Clinical Severity Grading</span>
        <span class="badge-chip badge-green">🔍 Grad-CAM Visualizer</span>
        <span class="badge-chip badge-yellow">🔄 Longitudinal Tracker</span>
    </div>
    <div class="hero-title">RetinaSense Retinal Diagnostic Assistant</div>
    <div class="hero-subtitle">
        Automated multi-condition ophthalmic disease screening, graded clinical severity staging, visual neural attention heatmaps, and longitudinal progression tracking.
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.isfile(checkpoint_path):
    st.error(f"⚠️ Model files are being initialized. Please ensure `{checkpoint_path}` is available.")
    st.stop()

model, device = load_model(checkpoint_path)

# ===========================================================================
# MODE 1: SINGLE IMAGE SCREENING & ALL-DISEASE SEVERITY STAGING
# ===========================================================================

if mode == "Single Image Screening & Severity Staging":
    uploaded = st.file_uploader(
        "Upload Patient Retinal Fundus Image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload high-quality macula-centered or optic-disc centered retinal fundus photograph."
    )

    if uploaded:
        img_path = save_uploaded_file(uploaded)
        image_bgr = cv2.imread(img_path)
        h, w = image_bgr.shape[:2]

        with st.spinner("Analyzing retinal fundus architecture, segmenting lesions, and calculating multi-disease severities..."):
            results_sorted, raw_probs, severity_evals, elapsed = predict(image_bgr, model, device, active_thresholds)

        detected_codes = list(severity_evals.keys())
        is_normal = (len(detected_codes) == 0)

        # -------------------------------------------------------------------
        # TOP DUAL-COLUMN HERO SUMMARY
        # -------------------------------------------------------------------
        col_img, col_summary = st.columns([1, 1.25])

        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 12px 0;'>📷 Patient Fundus Photograph</h4>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                <div>Resolution: <strong style="color: #ffffff;">{w} × {h} px</strong></div>
                <div>Analysis Time: <strong style="color: #34d399;">{elapsed*1000:.1f} ms</strong></div>
                <div>Status: <span class="badge-chip badge-green" style="padding: 2px 8px; font-size: 0.72rem;">Screened</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_summary:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 12px 0;'>📋 AI Diagnostic & Triage Assessment</h4>", unsafe_allow_html=True)
            
            if is_normal:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 2.2rem;">🟢</span>
                    <div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #34d399;">Healthy / Normal Retina</div>
                        <div style="font-size: 0.88rem; color: #94a3b8;">No sight-threatening retinal pathology detected above calibrated cutoffs.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                top_prob = max(raw_probs)
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="stat-sub">Screening Confidence</div><div class="stat-value" style="color: #34d399;">{(1-top_prob)*100:.1f}%</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="stat-sub">Triage Recommendation</div><div class="stat-value" style="font-size: 1.3rem; color: #38bdf8;">{NORMAL_CLINICAL_PROTOCOL["urgency"]}</div>', unsafe_allow_html=True)
            else:
                primary_code = max(detected_codes, key=lambda c: raw_probs[DISEASE_COLS.index(c)])
                primary_sev = severity_evals[primary_code]
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 2.2rem;">{DISEASE_ICONS[primary_code]}</span>
                    <div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #f87171;">{primary_sev['disease_name']}</div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-top: 2px;">
                            <span class="badge-chip {primary_sev['pill']}">{primary_sev['grade_name']}</span>
                            <span style="font-size: 0.82rem; color: #94a3b8;">({primary_sev['scale_name']})</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if len(detected_codes) > 1:
                    additional_names = [severity_evals[c]['disease_name'] for c in detected_codes if c != primary_code]
                    st.markdown(f"<div style='font-size: 0.84rem; color: #fbbf24; margin-bottom: 12px;'>⚠️ <strong>Co-occurring findings:</strong> {', '.join(additional_names)}</div>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="stat-sub">Primary Confidence</div><div class="stat-value" style="color: #f87171;">{primary_sev["confidence"]*100:.1f}%</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="stat-sub">Triage Urgency</div><div class="stat-value" style="font-size: 1.25rem; color: #fbbf24;">{primary_sev["urgency"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # -------------------------------------------------------------------
        # INTERACTIVE 5-TAB WORKSPACE
        # -------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Disease Probability Matrix",
            "🩺 Clinical Care & Action Plan",
            "🔍 Grad-CAM Neural Attention",
            "📈 Probability Distribution Chart",
            "📚 Clinical Pathology Atlas"
        ])

        # TAB 1: DISEASE PROBABILITY MATRIX
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📊 Multi-Pathology Diagnostic Matrix</h4>", unsafe_allow_html=True)
            
            for name, prob in results_sorted.items():
                code = [c for c, n in DISEASE_NAMES.items() if n == name][0]
                icon = DISEASE_ICONS[code]
                th = active_thresholds.get(code, 0.30)
                is_detected = prob >= th
                
                status_badge = '<span class="badge-chip badge-red">🔴 Detected</span>' if is_detected else '<span class="badge-chip badge-green">🟢 Negative</span>'
                bar_color = "linear-gradient(90deg, #ef4444, #f87171)" if is_detected else "linear-gradient(90deg, #38bdf8, #818cf8)"
                
                sev_label = severity_evals[code]["grade_name"] if code in severity_evals else "Normal / Low Risk"
                sev_pill = severity_evals[code]["pill"] if code in severity_evals else "badge-blue"

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.5); padding: 14px 18px; border-radius: 12px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.3rem;">{icon}</span>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #ffffff;">{name}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="badge-chip {sev_pill}">{sev_label}</span>
                            {status_badge}
                            <span style="font-size: 1.1rem; font-weight: 800; color: #ffffff; min-width: 60px; text-align: right;">{prob*100:.1f}%</span>
                        </div>
                    </div>
                    <div class="prog-bar-container">
                        <div class="prog-bar-fill" style="width: {min(100.0, prob*100):.1f}%; background: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 2: CLINICAL CARE & ACTION PLAN
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>🩺 Evidence-Based Clinical Recommendations & Triage Directives</h4>", unsafe_allow_html=True)
            
            if is_normal:
                st.markdown(f"""
                <div class="rec-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="color: #34d399; margin: 0;">Routine Screening Protocol (Healthy Eye)</h4>
                        <span class="badge-chip badge-green">{NORMAL_CLINICAL_PROTOCOL['urgency']}</span>
                    </div>
                    <p style="color: #cbd5e1; margin-bottom: 12px;">{NORMAL_CLINICAL_PROTOCOL['description']}</p>
                    <p style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">🔬 Recommended Periodic Workup:</p>
                    <ul style="color: #e2e8f0; margin-bottom: 12px;">{''.join([f'<li>{w}</li>' for w in NORMAL_CLINICAL_PROTOCOL['workup']])}</ul>
                    <p style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">🥗 Preventative Eye Care Guidance:</p>
                    <ul style="color: #e2e8f0;">{''.join([f'<li>{m}</li>' for m in NORMAL_CLINICAL_PROTOCOL['management']])}</ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                for code, s_data in severity_evals.items():
                    st.markdown(f"""
                    <div class="rec-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <h4 style="color: #38bdf8; margin: 0;">{DISEASE_ICONS[code]} Clinical Protocol: {s_data['disease_name']} ({s_data['grade_name']})</h4>
                            <span class="badge-chip {s_data['pill']}">{s_data['urgency']}</span>
                        </div>
                        <p style="color: #cbd5e1; margin-bottom: 10px;"><strong>Pathology Profile:</strong> {s_data['description']}</p>
                        <p style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">🔬 Recommended Confirmatory Diagnostic Workup:</p>
                        <ul style="color: #e2e8f0; margin-bottom: 10px;">{''.join([f'<li>{w}</li>' for w in s_data['workup']])}</ul>
                        <p style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">💊 Medical & Surgical Management Plan:</p>
                        <ul style="color: #e2e8f0; margin-bottom: 12px;">{''.join([f'<li>{m}</li>' for m in s_data['management']])}</ul>
                        <div class="red-flag-box">
                            <strong style="color: #f87171;">⚠️ Patient Red Flags & Emergent Warning Signs:</strong>
                            <ul style="margin: 4px 0 0 0; color: #fca5a5;">{''.join([f'<li>{rf}</li>' for rf in s_data['red_flags']])}</ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 3: GRAD-CAM X-RAY SALIENCY
        with tab3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 6px 0;'>🔍 Grad-CAM Neural Attention Visualizer</h4>", unsafe_allow_html=True)
            st.caption("Inspect the exact retinal anatomical landmarks (macula, optic disc cup, vascular arcades) driving deep neural network predictions.")
            
            selected_disease_name = st.selectbox("Select Target Disease for Attention Heatmap", list(DISEASE_NAMES.values()))
            selected_code = [c for c, n in DISEASE_NAMES.items() if n == selected_disease_name][0]
            selected_idx = DISEASE_COLS.index(selected_code)

            cam_overlay = generate_gradcam(model, image_bgr, selected_idx)
            g1, g2 = st.columns(2)
            with g1:
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="1. Original Fundus Photograph", use_container_width=True)
            with g2:
                st.image(cam_overlay, caption=f"2. Grad-CAM Activation Heatmap ({selected_disease_name})", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 4: PROBABILITY DISTRIBUTION CHART
        with tab4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📈 Disease Probability Distribution</h4>", unsafe_allow_html=True)
            st.pyplot(plot_probability_bars(results_sorted, active_thresholds))
            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 5: CLINICAL PATHOLOGY ATLAS
        with tab5:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📚 Clinical Reference & Staging Atlas</h4>", unsafe_allow_html=True)
            for code, s_info in SEVERITY_KNOWLEDGE_BASE.items():
                with st.expander(f"{DISEASE_ICONS[code]} {s_info['disease_name']} ({s_info['scale_name']})"):
                    st.markdown(f"**Clinical Scale Standard:** `{s_info['scale_name']}`")
                    for g_idx, g_details in s_info["grades"].items():
                        st.markdown(f"**• {g_details['name']}** [{g_details['urgency']}]: {g_details['description']}")
            st.markdown('</div>', unsafe_allow_html=True)

        os.remove(img_path)

# ===========================================================================
# MODE 2: RETINAL PROGRESSION TRACKING & TEMPORAL COMPARISON
# ===========================================================================

else:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #38bdf8; margin-top: 0;">🔄 Longitudinal Retinal Progression Tracking</h3>
        <p style="color: #94a3b8; margin-bottom: 0;">Quantifies disease progression rates across all 6 pathologies and maps structural microvascular changes between visits using feature homography alignment and SSIM difference mapping.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        baseline_file = st.file_uploader("Upload Baseline (Earlier Visit) Fundus Photograph", type=["jpg", "jpeg", "png"], key="baseline")
    with c2:
        followup_file = st.file_uploader("Upload Follow-up (Later Visit) Fundus Photograph", type=["jpg", "jpeg", "png"], key="followup")

    if baseline_file and followup_file:
        baseline_path = save_uploaded_file(baseline_file)
        followup_path = save_uploaded_file(followup_file)

        baseline_raw_bgr = cv2.imread(baseline_path)
        followup_raw_bgr = cv2.imread(followup_path)

        baseline_img = load_and_preprocess(baseline_path)
        followup_img = load_and_preprocess(followup_path)

        with st.spinner("Aligning fundus landmarks (ORB + RANSAC) and computing multi-pathology progression rates..."):
            aligned, ok = align_images(baseline_img, followup_img)
            if not ok:
                st.warning("Feature alignment fell back to unwarped comparison due to insufficient matching keypoints.")
            change_score, heatmap, overlay = compute_difference(baseline_img, aligned)

            base_results, base_probs, base_sev_evals, _ = predict(baseline_raw_bgr, model, device, active_thresholds)
            fup_results, fup_probs, fup_sev_evals, _ = predict(followup_raw_bgr, model, device, active_thresholds)

            deltas_dict = {}
            for i, code in enumerate(DISEASE_COLS):
                d_name = DISEASE_NAMES[code]
                delta = float(fup_probs[i] - base_probs[i])
                deltas_dict[d_name] = delta

        # Top Summary Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB), caption="1. Baseline Visit Image", use_container_width=True)
        with colB:
            st.image(cv2.cvtColor(followup_img, cv2.COLOR_BGR2RGB), caption="2. Follow-up Visit Image", use_container_width=True)
        with colC:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="3. Structural Difference Heatmap (SSIM)", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        level_label, level_pill = risk_level(change_score)
        max_prog_disease = max(deltas_dict.keys(), key=lambda k: deltas_dict[k])
        max_prog_val = deltas_dict[max_prog_disease]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown(f'<div class="stat-sub">Structural Change Score</div><div class="stat-value">{change_score:.3f}</div>', unsafe_allow_html=True)
        pc2.markdown(f'<div class="stat-sub">Overall Trajectory</div><div class="stat-value"><span class="badge-chip {level_pill}" style="font-size: 1.15rem;">{level_label}</span></div>', unsafe_allow_html=True)
        
        prog_color = "#f87171" if max_prog_val > 0.05 else ("#fbbf24" if max_prog_val > 0 else "#34d399")
        sign_str = "+" if max_prog_val > 0 else ""
        pc3.markdown(f'<div class="stat-sub">Top Progressing Pathology</div><div class="stat-value" style="font-size: 1.15rem; color: {prog_color};">{max_prog_disease} ({sign_str}{max_prog_val*100:.1f}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
            "📊 Disease Progression Rate (Δ% Graph)",
            "📈 Risk Comparison (Baseline vs Follow-up)",
            "📋 Longitudinal Disease Trajectory Table",
            "🩺 Clinical Action Directives"
        ])

        # TAB 1: Progression Rate Bar Graph
        with p_tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📊 Per-Disease Progression Rate (Δ% Change)</h4>", unsafe_allow_html=True)
            st.pyplot(plot_disease_progression_rate_bars(deltas_dict))
            st.markdown("""
            <div style="display: flex; gap: 18px; margin-top: 10px; font-size: 0.85rem; color: #94a3b8; flex-wrap: wrap;">
                <div><span style="color: #ef4444; font-weight: bold;">■ Red:</span> Significant Progression (> +5.0%)</div>
                <div><span style="color: #f59e0b; font-weight: bold;">■ Yellow:</span> Mild Progression (0.0% to +5.0%)</div>
                <div><span style="color: #64748b; font-weight: bold;">■ Gray:</span> Stable (0.0%)</div>
                <div><span style="color: #10b981; font-weight: bold;">■ Green:</span> Regression / Improvement (< -5.0%)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 2: Grouped Comparison Chart
        with p_tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📈 Baseline vs. Follow-up Disease Probability Comparison</h4>", unsafe_allow_html=True)
            st.pyplot(plot_progression_comparison_bars(base_probs, fup_probs))
            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 3: Longitudinal Trajectory Table
        with p_tab3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>📋 Longitudinal Disease Trajectory & Severity Shift</h4>", unsafe_allow_html=True)
            
            for i, code in enumerate(DISEASE_COLS):
                d_name = DISEASE_NAMES[code]
                b_p = base_probs[i]
                f_p = fup_probs[i]
                d_val = deltas_dict[d_name] * 100
                
                if d_val > 8.0:
                    status_badge = f"<span class='badge-chip badge-red'>🔺 Rapid Progression (+{d_val:.1f}%)</span>"
                elif d_val > 2.0:
                    status_badge = f"<span class='badge-chip badge-yellow'>⚠️ Moderate Worsening (+{d_val:.1f}%)</span>"
                elif d_val < -2.0:
                    status_badge = f"<span class='badge-chip badge-green'>📉 Improvement ({d_val:.1f}%)</span>"
                else:
                    status_badge = f"<span class='badge-chip badge-blue'>🟢 Stable ({d_val:+.1f}%)</span>"

                b_sev = base_sev_evals[code]["grade_name"] if code in base_sev_evals else "None / Low"
                f_sev = fup_sev_evals[code]["grade_name"] if code in fup_sev_evals else "None / Low"

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.5); padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div style="font-weight: 700; color: #ffffff; width: 220px;">{DISEASE_ICONS[code]} {d_name}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Baseline: <strong style="color: #38bdf8;">{b_p*100:.1f}%</strong></div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Follow-up: <strong style="color: #f97316;">{f_p*100:.1f}%</strong></div>
                    <div>{status_badge}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; min-width: 180px; text-align: right;">{b_sev} ➔ <strong style="color: #ffffff;">{f_sev}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 4: Clinical Progression Recommendations
        with p_tab4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #38bdf8; margin: 0 0 16px 0;'>🩺 Longitudinal Clinical Assessment & Action Directives</h4>", unsafe_allow_html=True)
            
            progressing_diseases = [d for d, val in deltas_dict.items() if val > 0.03]
            
            if progressing_diseases:
                st.error(f"🔴 **Progressing Pathology Alert:** Clinically significant progression noted in: **{', '.join(progressing_diseases)}**.")
                for d_name in progressing_diseases:
                    code = [c for c, n in DISEASE_NAMES.items() if n == d_name][0]
                    delta_pct = deltas_dict[d_name] * 100
                    s_data = fup_sev_evals.get(code, classify_disease_severity(code, fup_probs[DISEASE_COLS.index(code)], active_thresholds.get(code, 0.30)))
                    
                    st.markdown(f"""
                    <div class="rec-box">
                        <h4 style="color: #f87171; margin: 0 0 6px 0;">{DISEASE_ICONS[code]} {d_name} — Progression Delta: +{delta_pct:.1f}%</h4>
                        <p><strong>Follow-up Staging:</strong> {s_data['grade_name']} | <strong>Urgency:</strong> <span class="badge-chip {s_data['pill']}">{s_data['urgency']}</span></p>
                        <p><strong>Priority Diagnostic Workup:</strong></p>
                        <ul>{''.join([f'<li>{w}</li>' for w in s_data['workup']])}</ul>
                        <p><strong>Action Plan:</strong></p>
                        <ul>{''.join([f'<li>{m}</li>' for m in s_data['management']])}</ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🟢 **Stable Longitudinal Status:** No significant disease progression detected across the 6 target retinal pathologies between visits.")
                st.markdown("""
                <div class="rec-box">
                    <h4 style="color: #34d399; margin: 0 0 6px 0;">Routine Follow-up Recommendation</h4>
                    <p style="margin: 0;">Continue standard annual or biennial screening. Maintain systemic metabolic and cardiovascular control.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        os.remove(baseline_path)
        os.remove(followup_path)
    else:
        st.info("Upload both a baseline (earlier) and a follow-up (later) retinal fundus image to track disease progression over time.")

# ===========================================================================
# Clean User-Centric Clinical Footer
# ===========================================================================

st.markdown("""
<div class="clinical-footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-weight: 700; color: #ffffff; font-size: 0.95rem;">🏥 RetinaSense Clinical Intelligence System</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">Automated Multi-Disease Retinal Screening & Decision Support System</div>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span class="badge-chip badge-blue" style="font-size: 0.75rem;">🛡️ Secure & Private Analysis</span>
            <span class="badge-chip badge-purple" style="font-size: 0.75rem;">⚡ Real-Time Neural Triage</span>
            <span class="badge-chip badge-green" style="font-size: 0.75rem;">🩺 Evidence-Based Guidance</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)