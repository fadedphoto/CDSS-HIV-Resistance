"""
===========================================================================
HIV CDSS DUAL-PIPELINE PURWARUPA — ENHANCED v2.0
Sistem Pendukung Keputusan Klinis untuk Deteksi Resistensi ARV
Arsitektur: XGBoost (Klasifikasi) + Bi-LSTM (Regresi Kuantitatif)
Kerangka: CRISP-MED-DM | Standar: IUPAC Amino Acid Notation
===========================================================================
"""

import os
# ✅ FIX: Nonaktifkan optimisasi oneDNN SEBELUM TensorFlow di-import.
# oneDNN dapat mengubah urutan komputasi floating-point tergantung hardware CPU,
# yang berpotensi menyebabkan instabilitas numerik (NaN) pada inferensi Bi-LSTM
# di lingkungan cloud yang berbeda dari lingkungan lokal tempat model dilatih.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import concurrent.futures
import time
import logging
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# KONFIGURASI LOGGING (Reproduksibilitas Komputasional)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HIV-CDSS")


# ===========================================================================
# KONFIGURASI HALAMAN & TEMA VISUAL
# ===========================================================================
st.set_page_config(
    page_title="CDSS Resistensi HIV | 21106050069",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "CDSS Prediksi Resistensi HIV — Purwarupa Eksperimental Arsitektur Dual-Pipeline (CRISP-DM)"
    }
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — Desain Klinis Profesional
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ===== FONT & BASE ===== */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ===== ROOT VARIABLES — CLINICAL LIGHT MODE ===== */
:root {
    --bg-primary: #F0F4F8;
    --bg-secondary: #E8EFF5;
    --bg-card: #FFFFFF;
    --bg-card-hover: #F7FAFC;
    --accent-blue: #1D6FA4;
    --accent-teal: #0E8C7A;
    --accent-red: #C0392B;
    --accent-green: #1A7A4A;
    --accent-amber: #B7770D;
    --text-primary: #1A2332;
    --text-secondary: #4A5568;
    --text-muted: #718096;
    --border-subtle: rgba(29, 111, 164, 0.15);
    --border-active: rgba(29, 111, 164, 0.45);
    --shadow-glow: 0 0 20px rgba(29, 111, 164, 0.10);
}

/* ===== MAIN APP BACKGROUND — CLINICAL WHITE ===== */
.stApp {
    background: linear-gradient(160deg, #EEF3F8 0%, #F5F8FB 50%, #EEF3F8 100%);
    color: var(--text-primary);
}

/* ===== HIDE DEFAULT STREAMLIT ELEMENTS ===== */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* ✅ FIX TERVERIFIKASI dari source bundle Streamlit 1.58.0 (frontend/app build):
   Tombol BUKA sidebar bernama data-testid="stExpandSidebarButton", dan ternyata
   dia adalah ANAK LANGSUNG dari stToolbar — bukan elemen terpisah di header.
   Patch sebelumnya menyembunyikan SELURUH stToolbar, sehingga tombol buka ikut
   hilang. Sekarang hanya stToolbarActions (ikon Deploy/menu/running-man) yang
   disembunyikan, sementara stExpandSidebarButton & stSidebarCollapseButton
   dipaksa selalu tampil & bisa diklik. */
header [data-testid="stToolbarActions"] {
    visibility: hidden !important;
}

/* ===== HIDE DEPLOY BUTTON — kompatibel Streamlit 1.38+ (class diganti stAppDeployButton) ===== */
.stAppDeployButton { display: none !important; }
.stDeployButton { display: none !important; }

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ===== SIDEBAR — CLEAN CLINICAL WHITE ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F8FC 100%) !important;
    border-right: 1px solid rgba(29, 111, 164, 0.18) !important;
    box-shadow: 2px 0 12px rgba(29,111,164,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-weight: 600;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ===== SELECTBOX & INPUTS — LIGHT ===== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1px solid rgba(29, 111, 164, 0.25) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--border-active) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ===== PRIMARY BUTTON — CLINICAL BLUE ===== */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #1D6FA4 0%, #2589C2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(29, 111, 164, 0.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29, 111, 164, 0.38) !important;
}

/* ===== METRIC CARDS — LIGHT ===== */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(29, 111, 164, 0.18) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 4px rgba(29,111,164,0.07) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* ===== ALERTS ===== */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
    font-size: 0.875rem !important;
}

/* ===== TABS — LIGHT ===== */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 0.6rem 1.2rem !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ===== EXPANDER — LIGHT ===== */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(29, 111, 164, 0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ===== DATAFRAME — LIGHT ===== */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(29, 111, 164, 0.15) !important;
    border-radius: 8px !important;
}

/* ===== SPINNER ===== */
[data-testid="stSpinner"] {
    color: var(--accent-blue) !important;
}

/* ===== DIVIDER ===== */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
}

/* ===== HEADER CUSTOM CLASSES — CLINICAL LIGHT ===== */
.cdss-header {
    background: linear-gradient(135deg, rgba(29,111,164,0.07) 0%, rgba(14,140,122,0.05) 100%);
    border: 1px solid rgba(29,111,164,0.2);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.cdss-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal), transparent);
}
.cdss-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.cdss-header p {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin: 0.5rem 0 0 0;
    line-height: 1.5;
}
.cdss-badge {
    display: inline-block;
    background: rgba(29,111,164,0.10);
    border: 1px solid rgba(29,111,164,0.28);
    color: #1D6FA4;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 0.4rem;
}

/* ===== RESULT CARDS — LIGHT ===== */
.result-card-resistant {
    background: linear-gradient(135deg, rgba(192,57,43,0.07) 0%, rgba(192,57,43,0.03) 100%);
    border: 1px solid rgba(192,57,43,0.25);
    border-left: 4px solid #C0392B;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
}
.result-card-susceptible {
    background: linear-gradient(135deg, rgba(26,122,74,0.07) 0%, rgba(26,122,74,0.03) 100%);
    border: 1px solid rgba(26,122,74,0.25);
    border-left: 4px solid #1A7A4A;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
}
.result-value-resistant {
    font-size: 2rem;
    font-weight: 800;
    color: #C0392B;
    line-height: 1.1;
    font-family: 'IBM Plex Mono', monospace;
}
.result-value-susceptible {
    font-size: 2rem;
    font-weight: 800;
    color: #1A7A4A;
    line-height: 1.1;
    font-family: 'IBM Plex Mono', monospace;
}
.result-fc-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1D6FA4;
    line-height: 1.1;
    font-family: 'IBM Plex Mono', monospace;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ===== PIPELINE STATUS — LIGHT ===== */
.pipeline-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #FFFFFF;
    border: 1px solid rgba(29,111,164,0.18);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.status-dot-active {
    width: 8px; height: 8px;
    background: #1A7A4A;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(26,122,74,0.5);
    animation: pulse-dot 2s infinite;
}
.status-dot-idle {
    width: 8px; height: 8px;
    background: var(--text-muted);
    border-radius: 50%;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ===== SECTION TITLE — LIGHT ===== */
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(29,111,164,0.15);
}

/* ===== MONOSPACE DISPLAY — LIGHT ===== */
.mono-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    background: #F0F4F8;
    border: 1px solid rgba(29,111,164,0.15);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    color: var(--text-secondary);
    word-break: break-all;
    line-height: 1.6;
}

/* ===== INFO ROW — LIGHT ===== */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(29,111,164,0.1);
    font-size: 0.82rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--text-muted); font-weight: 500; }
.info-val { color: var(--text-primary); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

/* ===== SHAP CONTAINER ===== */
.shap-container {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid var(--border-subtle);
}

/* ===== FOOTER — LIGHT ===== */
.cdss-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.72rem;
    padding: 1.5rem 0 0.5rem;
    letter-spacing: 0.04em;
    border-top: 1px solid rgba(29,111,164,0.15);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 1. VALIDASI INPUT — Ketahanan IUPAC (Robustness)
# ===========================================================================
VALID_AMINO_ACIDS = set("ARNDCEQGHILKMFPSTWYV-.*")

# Clinical cutoffs per drug — sesuai Stanford HIVDB PhenoSense Assay (Cell 3 notebook)
CLINICAL_CUTOFFS = {
    # NRTI
    '3TC': 3.5, 'ABC': 4.5, 'AZT': 1.9, 'D4T': 1.7, 'DDI': 1.3, 'TDF': 1.4,
    # NNRTI
    'EFV': 3.0, 'NVP': 4.5, 'ETR': 2.9, 'RPV': 2.0,
    # PI
    'FPV': 4.0, 'ATV': 5.2, 'IDV': 10.0, 'LPV': 9.0,
    'NFV': 3.6, 'SQV': 2.3, 'TPV': 2.0,  'DRV': 10.0,
    # INSTI
    'RAL': 1.5, 'EVG': 2.5, 'DTG': 4.0, 'BIC': 2.5,
}

def validate_iupac_sequence(sequence: str) -> tuple[bool, str]:
    if not sequence or not sequence.strip():
        return False, "Kolom input sekuens tidak boleh kosong."
    seq_upper = "".join(sequence.upper().split())

    # ✅ FIX: Deteksi sekuens yang tidak mengandung residu asam amino valid sama sekali
    # (mis. input murni numerik seperti "12345" harus ditolak)
    valid_aa_letters = set("ARNDCEQGHILKMFPSTWYV")
    if not set(seq_upper).intersection(valid_aa_letters):
        return False, (
            "⚠️ Sekuens tidak mengandung residu asam amino IUPAC yang valid. "
            "Input berupa angka murni atau simbol tanpa kode asam amino tidak dapat diproses."
        )

    invalid_chars = sorted(set(
        char for char in seq_upper
        if char not in VALID_AMINO_ACIDS and not char.isdigit()
    ))
    if invalid_chars:
        return False, f"Karakter tidak standar terdeteksi pada sekuens input: `{'`, `'.join(invalid_chars)}`. Harap periksa kembali."
    if len(seq_upper) < 3:
        return False, "Input terlalu pendek untuk diproses. Masukkan profil mutasi (contoh: L10F M41L) atau sekuens asam amino lengkap."
    return True, seq_upper


def parse_mutation_profile(sequence: str) -> list[str]:
    import re
    pattern = r'[A-Z]\d+[A-Z]'
    return re.findall(pattern, sequence.upper())


# ===========================================================================
# 2. PEMUATAN ARTEFAK — PATH DIPERBAIKI DENGAN os.path.join()
# ===========================================================================
SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))
BILSTM_DIR = SAVE_DIR


@st.cache_resource(show_spinner=False)
def load_global_artifacts() -> dict | None:
    # ✅ FIX: Semua path pakai os.path.join() — tidak ada lagi f-string dengan "/"
    required_files = {
        "xgb_models":     os.path.join(SAVE_DIR, "xgb_models_medical_grade.pkl"),
        "all_ready_data": os.path.join(SAVE_DIR, "all_ready_data_hybrid.pkl"),
        "threshold_map":  os.path.join(SAVE_DIR, "xgb_threshold_map.pkl"),
    }

    missing = [k for k, p in required_files.items() if not os.path.isfile(p)]
    if missing:
        logger.error(f"File artefak tidak ditemukan: {missing}")
        return None

    try:
        loaded = {key: joblib.load(path) for key, path in required_files.items()}

        # ✅ PATCH 5A: Muat lstm_confidence_map.pkl (opsional — tidak blokir sistem)
        # File ini dihasilkan oleh notebook Cell 15 dan menyimpan Confidence Tier
        # per-obat berdasarkan CV RMSE Bi-LSTM: HIGH (<0.35) | MODERATE (<0.55) | LOW (≥0.55)
        confidence_map_path = os.path.join(SAVE_DIR, "lstm_confidence_map.pkl")
        if os.path.isfile(confidence_map_path):
            loaded["lstm_confidence_map"] = joblib.load(confidence_map_path)
            logger.info(f"Confidence map Bi-LSTM dimuat: {loaded['lstm_confidence_map']}")
        else:
            loaded["lstm_confidence_map"] = {}
            logger.warning("lstm_confidence_map.pkl tidak ditemukan — tier akan ditampilkan sebagai N/A")

        logger.info(f"Artefak global berhasil dimuat. Obat tersedia: {list(loaded['xgb_models'].keys())}")
        return loaded
    except Exception as e:
        logger.error(f"Gagal memuat artefak global: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_bilstm_model(drug_name: str) -> tf.keras.Model | None:
    # ✅ FIX: Semua candidates pakai os.path.join()
    candidates = [
        os.path.join(BILSTM_DIR, f"bilstm_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"bilstm_model_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"model_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"bilstm_{drug_name}.h5"),
        os.path.join(BILSTM_DIR, f"bilstm_model_{drug_name}.h5"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                model = tf.keras.models.load_model(path)
                logger.info(f"Bi-LSTM untuk {drug_name} dimuat dari: {path}")
                return model
            except Exception as e:
                logger.warning(f"Gagal memuat dari {path}: {e}")
                continue
    logger.error(f"Tidak ada file Bi-LSTM yang valid ditemukan untuk obat: {drug_name}")
    return None


def get_available_drugs(artifacts: dict) -> list[str]:
    return sorted(list(artifacts["xgb_models"].keys()))


def check_bilstm_availability(drug_name: str) -> bool:
    # ✅ FIX: Semua candidates pakai os.path.join()
    candidates = [
        os.path.join(BILSTM_DIR, f"bilstm_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"bilstm_model_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"model_{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"{drug_name}.keras"),
        os.path.join(BILSTM_DIR, f"bilstm_{drug_name}.h5"),
        os.path.join(BILSTM_DIR, f"bilstm_model_{drug_name}.h5"),
    ]
    return any(os.path.isfile(p) for p in candidates)


# ===========================================================================
# 3. PIPELINE PREPROCESSING & EKSTRAKSI FITUR
# ===========================================================================
def extract_features(sequence, tokenizer, max_len, xgb_model):
    seq_numeric = tokenizer.texts_to_sequences([sequence])
    X_lstm = pad_sequences(seq_numeric, maxlen=max_len, padding='post', truncating='post')

    num_features = xgb_model.n_features_in_
    X_xgb = pd.DataFrame(np.zeros((1, num_features), dtype=int))

    if hasattr(xgb_model, 'feature_names_in_') and xgb_model.feature_names_in_ is not None:
        X_xgb.columns = xgb_model.feature_names_in_

    tokens = sequence.upper().split()
    feature_cols = list(X_xgb.columns)

    for i, token in enumerate(tokens):
        if i >= len(feature_cols):
            break
        if token not in ('-', '.', 'NAN', ''):
            X_xgb.iloc[0, i] = 1

    logger.info(f"X_xgb shape={X_xgb.shape}, non-zero={int(X_xgb.values.sum())}")
    return X_xgb, X_lstm


# ===========================================================================
# 4. WORKER THREAD — Alur Paralel
# ===========================================================================
def run_xgb_classification(model, X_xgb, threshold):
    t0 = time.time()
    prob = float(model.predict_proba(X_xgb)[:, 1][0])
    diagnosis = "RESISTEN" if prob >= threshold else "RENTAN"
    margin = abs(prob - threshold)
    confidence = "Tinggi" if margin > 0.2 else ("Sedang" if margin > 0.08 else "Rendah")
    elapsed = time.time() - t0
    logger.info(f"XGBoost: {diagnosis} ({prob:.4f}) dalam {elapsed:.3f}s")
    return {
        "probability": prob,
        "diagnosis": diagnosis,
        "threshold": threshold,
        "margin": margin,
        "confidence": confidence,
        "elapsed_ms": elapsed * 1000
    }


def run_bilstm_regression(model, X_lstm, drug_name=""):
    t0 = time.time()
    pred_log10 = float(model.predict(X_lstm, verbose=0)[0][0])
    fold_change = 10 ** pred_log10
    elapsed = time.time() - t0
    clinical_cutoff = CLINICAL_CUTOFFS.get(drug_name.upper(), 3.5)

    # ✅ FIX: Tangkap NaN/Inf di sumbernya. Tanpa ini, nilai non-finite akan
    # diam-diam gagal di SEMUA perbandingan "<" di bawah (NaN selalu False),
    # sehingga jatuh ke kategori "Resistensi Tinggi" secara keliru.
    if not np.isfinite(fold_change):
        logger.error(f"Bi-LSTM [{drug_name}] menghasilkan nilai non-finite (log10={pred_log10}) — prediksi ditolak")
        return {
            "fold_change": float("nan"),
            "log10_value": pred_log10,
            "severity": "Tidak Valid",
            "severity_color": "gray",
            "clinical_cutoff": clinical_cutoff,
            "elapsed_ms": elapsed * 1000,
            "is_valid": False
        }

    if fold_change < clinical_cutoff:
        severity = "Rentan"
        severity_color = "green"
    elif fold_change < clinical_cutoff * 3:
        severity = "Resistensi Rendah"
        severity_color = "amber"
    elif fold_change < clinical_cutoff * 9:
        severity = "Resistensi Sedang"
        severity_color = "orange"
    else:
        severity = "Resistensi Tinggi"
        severity_color = "red"

    logger.info(f"Bi-LSTM [{drug_name}] FC={fold_change:.2f} ({severity}) dalam {elapsed:.3f}s")
    return {
        "fold_change": fold_change,
        "log10_value": pred_log10,
        "severity": severity,
        "severity_color": severity_color,
        "clinical_cutoff": clinical_cutoff,
        "elapsed_ms": elapsed * 1000,
        "is_valid": True
    }


def render_shap_summary(model, X_train_xgb):
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_xgb)

    if isinstance(shap_values, list):
        shap_arr = np.abs(shap_values[1]) if len(shap_values) > 1 else np.abs(shap_values[0])
    else:
        shap_arr = np.abs(shap_values)
    mean_shap = shap_arr.mean(axis=0)
    feature_names = list(X_train_xgb.columns)
    top_indices = np.argsort(mean_shap)[::-1][:5]
    top_features = {feature_names[i]: float(mean_shap[i]) for i in top_indices if i < len(feature_names)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    shap.summary_plot(shap_values, X_train_xgb, show=False, max_display=15)
    plt.title(
        f"Faktor Mutasi Utama (SHAP Global) — {len(X_train_xgb)} Sampel Training",
        fontweight='bold', fontsize=10, color='#1E293B', pad=12
    )
    plt.xlabel("Nilai SHAP (Pengaruh terhadap Prediksi Resisten)", fontsize=8, color='#475569')
    plt.tick_params(labelsize=7.5, colors='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    elapsed = time.time() - t0
    logger.info(f"SHAP selesai dalam {elapsed:.3f}s. Top fitur: {top_features}")
    # ✅ FIX: Lepaskan figure dari registry global pyplot agar tidak menumpuk di
    # memori pada sesi yang panjang (setiap klik tombol membuat figure baru).
    # fig tetap valid untuk dipakai di st.pyplot() setelah di-close.
    plt.close(fig)
    return fig, top_features


def render_probability_gauge(probability, threshold):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#FFFFFF')

    r_outer, r_inner = 1.0, 0.62
    zones = [(0.0, 0.4, '#1A7A4A', 'Rentan'), (0.4, 0.65, '#B7770D', 'Ambang'), (0.65, 1.0, '#C0392B', 'Resisten')]
    for (v_start, v_end, color, _) in zones:
        ang_s = np.pi - v_start * np.pi
        ang_e = np.pi - v_end * np.pi
        theta = np.linspace(ang_s, ang_e, 60)
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        x_inner = r_inner * np.cos(theta[::-1])
        y_inner = r_inner * np.sin(theta[::-1])
        ax.fill(np.concatenate([x_outer, x_inner]), np.concatenate([y_outer, y_inner]), color=color, alpha=0.85)

    needle_angle = np.pi - probability * np.pi
    needle_len = 0.82
    ax.annotate("", xy=(needle_len * np.cos(needle_angle), needle_len * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#1A2332', lw=2.5))

    thresh_angle = np.pi - threshold * np.pi
    ax.plot([0.6 * np.cos(thresh_angle)], [0.6 * np.sin(thresh_angle)],
            marker='v', color='#B7770D', markersize=7, zorder=5)

    circle = plt.Circle((0, 0), 0.12, color='#FFFFFF', zorder=6, linewidth=1.5, edgecolor='#CBD5E0')
    ax.add_patch(circle)

    color_prob = '#C0392B' if probability >= threshold else '#1A7A4A'
    ax.text(0, -0.28, f"{probability:.3f}", ha='center', va='center',
            fontsize=16, fontweight='bold', color=color_prob)
    ax.text(0, -0.46, "PROBABILITAS", ha='center', va='center',
            fontsize=6, color='#718096', fontweight='600')

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.65, 1.2)
    ax.axis('off')
    ax.set_facecolor('#FFFFFF')
    plt.tight_layout()
    # ✅ FIX: sama seperti render_shap_summary — cegah penumpukan figure di memori.
    plt.close(fig)
    return fig


# ===========================================================================
# 5. UTILITAS TAMPILAN
# ===========================================================================
def format_sequence_display(sequence, max_chars=80):
    mutations = parse_mutation_profile(sequence)
    if mutations:
        return " · ".join(mutations)
    return sequence[:max_chars] + ("..." if len(sequence) > max_chars else "")


def get_drug_class(drug_name):
    classifications = {
        "3TC": "NRTI", "FTC": "NRTI", "TDF": "NRTI", "AZT": "NRTI",
        "D4T": "NRTI", "DDI": "NRTI", "ABC": "NRTI",
        "EFV": "NNRTI", "NVP": "NNRTI", "RPV": "NNRTI", "ETR": "NNRTI",
        "LPV": "PI", "ATV": "PI", "DRV": "PI", "SQV": "PI",
        "IDV": "PI", "NFV": "PI", "FPV": "PI", "TPV": "PI",
        "RAL": "INSTI", "DTG": "INSTI", "EVG": "INSTI", "BIC": "INSTI",
    }
    for code, cls in classifications.items():
        if code.upper() in drug_name.upper():
            return cls
    return "ARV"


def render_fold_change_interpretation(fc):
    if fc < 3.5:
        return "**Rentan (Susceptible)**: FC < 3.5× — Virus masih sensitif terhadap terapi ARV."
    elif fc < 10.0:
        return "**Resistensi Rendah**: 3.5× ≤ FC < 10× — Kemungkinan terjadi penurunan efektivitas klinis obat."
    elif fc < 30.0:
        return "**Resistensi Sedang**: 10× ≤ FC < 30× — Terdapat kompromi signifikan pada efikasi terapi."
    else:
        return "**Resistensi Tinggi**: FC ≥ 30× — Terapi ini kemungkinan besar sudah tidak efektif dan harus dihindari."


# ===========================================================================
# SESSION STATE
# ===========================================================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def add_to_history(drug, sequence, classification, regression):
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "drug": drug,
        "sequence_preview": format_sequence_display(sequence, 30),
        "diagnosis": classification["diagnosis"],
        "probability": classification["probability"],
        "fold_change": regression["fold_change"],
        "severity": regression["severity"],
    }
    st.session_state.analysis_history.insert(0, entry)
    if len(st.session_state.analysis_history) > 20:
        st.session_state.analysis_history = st.session_state.analysis_history[:20]


# ===========================================================================
# HEADER UTAMA
# ===========================================================================
st.markdown("""
<div class="cdss-header">
    <h1>🧬 Sistem Prediksi Resistensi Obat HIV (Clinical Decision Support System)</h1>
    <p>Purwarupa arsitektur Dual-Pipeline (Klasifikasi Diagnostik XGBoost &amp; Regresi Kuantitatif Bi-LSTM) untuk deteksi 
    resistensi obat antiretroviral berdasarkan profil mutasi sekuens asam amino. 
    Dilengkapi lapisan transparansi Explainable AI (SHAP) untuk mendukung keputusan klinis.</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PANEL SIDEBAR
# ===========================================================================
artifacts = load_global_artifacts()

with st.sidebar:
    st.markdown("## 🎛️ Panel Kontrol Klinis")
    st.markdown("---")

    if artifacts is None:
        st.error(
            "**Sistem Tidak Tersedia**\n\n"
            f"File artefak model tidak ditemukan di direktori `{SAVE_DIR}`.\n\n"
            "Pastikan file berikut tersedia:\n"
            "- `xgb_models_medical_grade.pkl`\n"
            "- `all_ready_data_hybrid.pkl`\n"
            "- `xgb_threshold_map.pkl`"
        )
        st.stop()

    available_drugs = get_available_drugs(artifacts)

    st.markdown('<div class="section-title">Seleksi Obat Antiretroviral (ARV)</div>', unsafe_allow_html=True)
    drug_choice = st.selectbox(
        "Obat ARV Target:",
        available_drugs,
        help="Pilih jenis obat spesifik untuk mengevaluasi profil resistensi sekuens input virus."
    )

    drug_class = get_drug_class(drug_choice)
    threshold_val = artifacts["threshold_map"].get(drug_choice, 0.5)
    bilstm_ready = check_bilstm_availability(drug_choice)

    st.markdown('<div class="section-title" style="margin-top:1rem">Status Pipeline</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            '<div class="pipeline-status"><div class="status-dot-active"></div><span>XGBoost</span></div>',
            unsafe_allow_html=True
        )
    with col_s2:
        dot_class = "status-dot-active" if bilstm_ready else "status-dot-idle"
        status_label = "Bi-LSTM"
        st.markdown(
            f'<div class="pipeline-status"><div class="{dot_class}"></div><span>{status_label}</span></div>',
            unsafe_allow_html=True
        )

    if not bilstm_ready:
        st.warning(
            f"⚠️ File model Bi-LSTM untuk **{drug_choice}** tidak tersedia. "
            "Alur regresi kuantitatif dinonaktifkan sementara."
        )

    st.markdown('<div class="section-title" style="margin-top:1rem">Informasi Obat</div>', unsafe_allow_html=True)

    # ✅ PATCH 5B: Ambil Confidence Tier per-obat dari lstm_confidence_map.pkl
    # Tier ini ditetapkan saat training berdasarkan CV RMSE Bi-LSTM (bukan per-prediksi)
    _conf_map = artifacts.get("lstm_confidence_map", {})
    _bilstm_tier = _conf_map.get(drug_choice, None)
    _tier_style = {
        "HIGH":     ("🟢", "#1A7A4A", "rgba(26,122,74,0.12)"),
        "MODERATE": ("🟡", "#B7770D", "rgba(183,119,13,0.12)"),
        "LOW":      ("🔴", "#C0392B", "rgba(192,57,43,0.12)"),
    }
    if _bilstm_tier and _bilstm_tier in _tier_style:
        _t_icon, _t_color, _t_bg = _tier_style[_bilstm_tier]
        _tier_html = (
            f'<span style="background:{_t_bg};color:{_t_color};'
            f'font-size:0.7rem;font-weight:700;padding:0.15rem 0.5rem;'
            f'border-radius:12px;font-family:\'IBM Plex Mono\',monospace;">'
            f'{_t_icon} {_bilstm_tier}</span>'
        )
    else:
        _tier_html = '<span style="color:var(--text-muted);font-size:0.75rem">— (model belum dimuat)</span>'

    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid rgba(29,111,164,0.18);border-radius:8px;padding:0.75rem 1rem;">'
        f'<div class="info-row"><span class="info-key">Kode Obat</span><span class="info-val">{drug_choice}</span></div>'
        f'<div class="info-row"><span class="info-key">Kelas Farmakologi</span><span class="info-val">{drug_class}</span></div>'
        f'<div class="info-row"><span class="info-key">Dynamic Threshold</span><span class="info-val">{threshold_val:.4f}</span></div>'
        f'<div class="info-row"><span class="info-key">Bi-LSTM Confidence Tier</span><span class="info-val">{_tier_html}</span></div>'
        f'<div class="info-row"><span class="info-key">Total Obat Tersedia</span><span class="info-val">{len(available_drugs)}</span></div>'
        f'</div>', unsafe_allow_html=True
    )

    st.markdown('<div class="section-title" style="margin-top:1rem">Input Sekuens / Mutasi</div>', unsafe_allow_html=True)

    default_seqs = {
        "3TC": "M184V K65R L74V", "FTC": "M184V K65R", "TDF": "K65R A62V",
        "AZT": "M41L D67N K70R L210W T215Y K219Q", "EFV": "K103N Y181C",
        "NVP": "K103N Y181C G190A", "LPV": "M46I I54V V82A",
        "DTG": "R263K", "RAL": "N155H Q148H",
    }
    default_val = next(
        (v for k, v in default_seqs.items() if k.upper() in drug_choice.upper()),
        "K65R M184V"
    )

    sequence_input = st.text_area(
        "Sekuens Asam Amino / Profil Mutasi:",
        value=default_val,
        height=110,
        help="Format input: Notasi mutasi standar (misal: K65R, M184V) atau sekuens residu asam amino murni."
    )

    with st.expander("⚙️ Opsi Analisis Lanjutan", expanded=False):
        show_gauge = st.checkbox("Tampilkan Probability Gauge", value=True)
        run_mode = st.radio(
            "Mode Eksekusi:",
            ["Simultan (Paralel)", "Sekuensial"],
            help="Mode simultan menggunakan ThreadPoolExecutor untuk efisiensi latensi komputasi yang optimal."
        )

    st.markdown("---")
    analyze_btn = st.button("⚙️ Jalankan Diagnosis Dual-Pipeline", use_container_width=True, type="primary")


# ===========================================================================
# AREA KONTEN UTAMA
# ===========================================================================
tab_analysis, tab_history, tab_info = st.tabs([
    "🔬 Analisis Dual-Pipeline",
    "📋 Riwayat Analisis",
    "ℹ️ Tentang Sistem"
])

# ---------------------------------------------------------------------------
# TAB 1: ANALISIS UTAMA
# ---------------------------------------------------------------------------
with tab_analysis:

    if not analyze_btn and st.session_state.last_result is None:
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid rgba(29,111,164,0.15);border-radius:10px;padding:1.5rem;text-align:center;box-shadow:0 1px 4px rgba(29,111,164,0.07)">
                <div style="font-size:2rem;margin-bottom:0.5rem">🧪</div>
                <div style="font-size:0.85rem;font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">Alur Klasifikasi Diagnostik (XGBoost)</div>
                <div style="font-size:0.78rem;color:var(--text-secondary)">Diagnosis biner klinis RESISTEN / RENTAN dengan optimalisasi metrik AUC-ROC dan pencarian Dynamic Threshold untuk meminimalkan False Negative.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_w2:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid rgba(29,111,164,0.15);border-radius:10px;padding:1.5rem;text-align:center;box-shadow:0 1px 4px rgba(29,111,164,0.07)">
                <div style="font-size:2rem;margin-bottom:0.5rem">📈</div>
                <div style="font-size:0.85rem;font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">Alur Regresi Kuantitatif (Bi-LSTM)</div>
                <div style="font-size:0.78rem;color:var(--text-secondary)">Estimasi nilai kelipatan resistensi (Fold Change) dalam skala Log₁₀ untuk mengukur derajat keparahan resistensi virus berdasarkan dependensi sekuensial.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_w3:
            st.markdown("""
            <div style="background:#FFFFFF;border:1px solid rgba(29,111,164,0.15);border-radius:10px;padding:1.5rem;text-align:center;box-shadow:0 1px 4px rgba(29,111,164,0.07)">
                <div style="font-size:2rem;margin-bottom:0.5rem">🧠</div>
                <div style="font-size:0.85rem;font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">Transparansi Medis (Explainable AI)</div>
                <div style="font-size:0.78rem;color:var(--text-secondary)">Membedah penalaran model kotak hitam (black-box) melalui atribusi kontribusi fitur untuk menunjukkan mutasi genetik mana yang paling memengaruhi keputusan prediksi.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;color:var(--text-muted);font-size:0.82rem;margin-top:2rem;padding:1rem">
            Konfigurasi target obat ARV dan masukkan sekuens pada panel kontrol di sebelah kiri, kemudian klik <strong>Jalankan Diagnosis Dual-Pipeline</strong>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn:
        is_valid, seq_result = validate_iupac_sequence(sequence_input)

        if not is_valid:
            st.error(f"⚠️ **Mitigasi Kesalahan Input**\n\n{seq_result}")
        else:
            seq_clean = seq_result
            mutations_found = parse_mutation_profile(seq_clean)

            st.info(
                f"🧬 Mengaktifkan Dual-Pipeline untuk obat **{drug_choice}** ({drug_class}) | "
                f"Sekuens: `{format_sequence_display(seq_clean)}` | "
                f"Mutasi terdeteksi: **{len(mutations_found)}** ({', '.join(mutations_found) if mutations_found else '—'})"
            )

            xgb_model    = artifacts["xgb_models"][drug_choice]
            tokenizer    = artifacts["all_ready_data"][drug_choice]['tokenizer']
            max_len      = artifacts["all_ready_data"][drug_choice]['max_len']
            threshold    = artifacts["threshold_map"][drug_choice]
            bilstm_model = load_bilstm_model(drug_choice) if bilstm_ready else None

            X_xgb, X_lstm = extract_features(seq_clean, tokenizer, max_len, xgb_model)
            t_total_start = time.time()

            with st.spinner("Memproses komputasi Klasifikasi, Regresi, dan atribusi SHAP..."):
                X_train_exp = artifacts['all_ready_data'][drug_choice]['X_train_xgb']

                # ✅ FIX (root cause segfault): render_shap_summary() memanggil API
                # pyplot global (plt.subplots/plt.title/plt.tight_layout) yang menurut
                # dokumentasi resmi Matplotlib TIDAK thread-safe. Menjalankannya di
                # worker thread ThreadPoolExecutor secara bersamaan dengan
                # tf.keras Model.predict() (juga bukan thread-safe secara resmi —
                # lih. tensorflow/tensorflow#61298) di thread lain memicu race
                # condition di level C/C++ yang berujung Segmentation Fault.
                # Segfault adalah crash di level OS (SIGSEGV), BUKAN exception Python,
                # sehingga tidak bisa dicegah dengan try/except atau guard NaN apa pun.
                # Solusi: SHAP/Matplotlib SELALU dijalankan sekuensial di main thread.
                # XGBoost & Bi-LSTM tetap boleh paralel karena keduanya operasi numerik
                # murni yang tidak menyentuh state global pyplot.
                if run_mode == "Simultan (Paralel)" and bilstm_model is not None:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_class = executor.submit(run_xgb_classification, xgb_model, X_xgb, threshold)
                        future_reg   = executor.submit(run_bilstm_regression, bilstm_model, X_lstm, drug_choice)
                        class_result = future_class.result()
                        reg_result   = future_reg.result()
                else:
                    class_result = run_xgb_classification(xgb_model, X_xgb, threshold)
                    reg_result   = run_bilstm_regression(bilstm_model, X_lstm, drug_choice) if bilstm_model else None

                # SHAP/Matplotlib: SELALU sekuensial di main thread, di luar executor.
                shap_result = render_shap_summary(xgb_model, X_train_exp)
                shap_fig, top_features = shap_result
                total_elapsed = (time.time() - t_total_start) * 1000

            st.session_state.last_result = {
                "drug": drug_choice,
                "seq_clean": seq_clean,
                "mutations": mutations_found,
                "classification": class_result,
                "regression": reg_result,
                "shap_fig": shap_fig,
                "top_features": top_features,
                "total_elapsed_ms": total_elapsed,
            }

            if reg_result:
                add_to_history(drug_choice, seq_clean, class_result, reg_result)

            st.success(f"✅ Komputasi selesai dalam **{total_elapsed:.0f} ms**")

    if st.session_state.last_result is not None:
        r = st.session_state.last_result
        class_res = r["classification"]
        reg_res   = r["regression"]

        st.markdown("---")

        col_class, col_reg = st.columns(2, gap="medium")

        with col_class:
            st.markdown("#### 📊 Hasil Klasifikasi Diagnostik — XGBoost")
            is_resistant = class_res["diagnosis"] == "RESISTEN"
            card_class = "result-card-resistant" if is_resistant else "result-card-susceptible"
            val_class  = "result-value-resistant" if is_resistant else "result-value-susceptible"
            icon = "⚠️" if is_resistant else "✅"

            # ✅ FIX: Confidence Tier Badge — visual berwarna (HIGH/MODERATE/LOW)
            _confidence_badge_map = {
                "Tinggi": {
                    "label": "HIGH",
                    "bg": "rgba(26,122,74,0.12)",
                    "border": "rgba(26,122,74,0.40)",
                    "color": "#1A7A4A",
                    "icon": "●"
                },
                "Sedang": {
                    "label": "MODERATE",
                    "bg": "rgba(183,119,13,0.12)",
                    "border": "rgba(183,119,13,0.40)",
                    "color": "#B7770D",
                    "icon": "●"
                },
                "Rendah": {
                    "label": "LOW",
                    "bg": "rgba(192,57,43,0.12)",
                    "border": "rgba(192,57,43,0.40)",
                    "color": "#C0392B",
                    "icon": "●"
                },
            }
            _cb = _confidence_badge_map.get(class_res['confidence'], _confidence_badge_map["Sedang"])
            confidence_badge_html = (
                f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
                f'background:{_cb["bg"]};border:1px solid {_cb["border"]};'
                f'color:{_cb["color"]};font-size:0.68rem;font-weight:700;'
                f'padding:0.18rem 0.65rem;border-radius:20px;'
                f'letter-spacing:0.1em;font-family:\'IBM Plex Mono\',monospace;'
                f'vertical-align:middle;">'
                f'<span style="font-size:0.55rem">{_cb["icon"]}</span>{_cb["label"]}'
                f'</span>'
            )

            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-label">Diagnosis Klinis</div>
                <div class="{val_class}">{icon} {class_res['diagnosis']}</div>
                <div class="result-sub" style="margin-top:0.6rem">
                    <strong>Probabilitas Kepercayaan:</strong> <code>{class_res['probability']:.4f}</code> &nbsp;|&nbsp;
                    <strong>Ambang Batas (Threshold):</strong> <code>{class_res['threshold']:.4f}</code>
                </div>
                <div class="result-sub" style="margin-top:0.5rem;display:flex;align-items:center;gap:0.6rem">
                    <span style="color:var(--text-secondary);font-size:0.8rem"><strong>Keyakinan Prediksi:</strong></span>
                    {confidence_badge_html}
                </div>
                <div class="result-sub" style="margin-top:0.3rem;font-size:0.75rem;color:var(--text-muted)">
                    Latensi inferensi: {class_res['elapsed_ms']:.1f} ms
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_reg:
            st.markdown("#### 📈 Hasil Estimasi Kuantitatif — Bi-LSTM")
            if reg_res is not None:
                fc_colors = {"green": "#22C55E", "amber": "#F59E0B", "orange": "#F97316", "red": "#EF4444"}
                fc_color_hex = fc_colors.get(reg_res["severity_color"], "#60A5FA")

                # ✅ PATCH 5C: Ambil Confidence Tier per-obat dari artifacts
                # (bukan per-prediksi — ini kualitas model Bi-LSTM berdasarkan CV RMSE training)
                _conf_map_res  = artifacts.get("lstm_confidence_map", {})
                _bilstm_tier_r = _conf_map_res.get(r["drug"], None)
                _tier_badge_styles = {
                    "HIGH":     ("#1A7A4A", "rgba(26,122,74,0.12)", "🟢"),
                    "MODERATE": ("#B7770D", "rgba(183,119,13,0.12)", "🟡"),
                    "LOW":      ("#C0392B", "rgba(192,57,43,0.12)", "🔴"),
                }
                if _bilstm_tier_r in _tier_badge_styles:
                    _tc, _tb, _ti = _tier_badge_styles[_bilstm_tier_r]
                    _tier_badge = (
                        f'<span style="background:{_tb};color:{_tc};font-size:0.68rem;'
                        f'font-weight:700;padding:0.18rem 0.6rem;border-radius:20px;'
                        f'font-family:\'IBM Plex Mono\',monospace;">{_ti} {_bilstm_tier_r}</span>'
                    )
                else:
                    _tier_badge = '<span style="color:var(--text-muted);font-size:0.75rem">N/A</span>'

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(59,130,246,0.1) 0%,rgba(20,184,166,0.06) 100%);border:1px solid rgba(59,130,246,0.25);border-left:4px solid {fc_color_hex};border-radius:10px;padding:1.25rem 1.5rem">
                    <div class="result-label">Prediksi Kelipatan Resistensi (Fold Change)</div>
                    <div class="result-fc-value" style="color:{fc_color_hex}">{reg_res['fold_change']:.2f}×</div>
                    <div class="result-sub">
                        <strong>Keparahan:</strong> {reg_res['severity']} &nbsp;|&nbsp;
                        <strong>Log₁₀(FC):</strong> <code>{reg_res['log10_value']:.3f}</code> &nbsp;|&nbsp;
                        <strong>Cutoff Klinis:</strong> <code>{reg_res['clinical_cutoff']}×</code>
                    </div>
                    <div class="result-sub" style="margin-top:0.5rem;display:flex;align-items:center;gap:0.6rem">
                        <span style="color:var(--text-secondary);font-size:0.8rem"><strong>Confidence Tier Model:</strong></span>
                        {_tier_badge}
                    </div>
                    <div class="result-sub" style="margin-top:0.3rem;font-size:0.75rem;color:var(--text-muted)">
                        Latensi inferensi: {reg_res['elapsed_ms']:.1f} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── DISCLAIMER: Confidence Tier LOW — dipicu dari pkl, bukan hardcoded set ──
                if _bilstm_tier_r == "LOW":
                    st.markdown(f"""
                    <div style="
                        background: rgba(183,119,13,0.08);
                        border: 1px solid rgba(183,119,13,0.35);
                        border-left: 4px solid #B7770D;
                        border-radius: 8px;
                        padding: 0.75rem 1.1rem;
                        margin-top: 0.65rem;
                        display: flex;
                        align-items: flex-start;
                        gap: 0.7rem;
                    ">
                        <span style="font-size:1.1rem;margin-top:0.05rem">⚠️</span>
                        <div>
                            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;
                                        text-transform:uppercase;color:#B7770D;margin-bottom:0.25rem;
                                        font-family:'IBM Plex Mono',monospace">
                                Peringatan · Confidence Tier LOW
                            </div>
                            <div style="font-size:0.78rem;color:#4A5568;line-height:1.55">
                                Model Bi-LSTM untuk obat <strong>{r["drug"]}</strong> dilatih
                                dengan jumlah data yang lebih sedikit dibanding obat lain di dataset Stanford HIVDB,
                                sehingga prediksi <em>Fold Change</em>-nya kurang akurat dan sebaiknya
                                tidak dijadikan acuan utama. <strong>Untuk obat ini,
                                gunakan hasil XGBoost sebagai dasar keputusan klinis.</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#FFFFFF;border:1px solid rgba(29,111,164,0.15);border-radius:10px;padding:1.25rem 1.5rem;opacity:0.6">
                    <div class="result-label">Model Bi-LSTM</div>
                    <div style="color:var(--text-muted);font-size:0.9rem">— Tidak Tersedia</div>
                    <div class="result-sub">File model Bi-LSTM untuk obat ini belum ditemukan di sistem.</div>
                </div>
                """, unsafe_allow_html=True)

        # ✅ KONGRUENSI DUAL-PIPELINE: Sintesis integratif antara XGBoost & Bi-LSTM
        if reg_res is not None:
            if not reg_res.get("is_valid", True):
                # ✅ FIX: Bi-LSTM gagal menghasilkan angka valid (NaN/Inf) — jangan
                # dipaksa dibandingkan ke XGBoost. "nan >= x" selalu False di Python,
                # yang tanpa guard ini akan keliru terbaca sebagai "konsisten sensitif".
                st.markdown("""
                <div style="background:rgba(113,128,150,0.08);border:1px solid rgba(113,128,150,0.3);border-radius:10px;
                            padding:0.9rem 1.5rem;margin-top:0.75rem;
                            display:flex;align-items:flex-start;gap:1rem">
                    <div style="font-size:1.5rem;margin-top:0.05rem">⚪</div>
                    <div>
                        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#718096;margin-bottom:0.25rem;
                                    font-family:'IBM Plex Mono',monospace">
                            Kesimpulan Kedua Model &nbsp;·&nbsp; TIDAK DAPAT DIBANDINGKAN
                        </div>
                        <div style="font-size:0.82rem;color:var(--text-secondary);line-height:1.5">
                            Bi-LSTM tidak menghasilkan prediksi numerik yang valid untuk input ini, sehingga evaluasi kongruensi dilewati. <strong>Gunakan hasil XGBoost sebagai dasar keputusan untuk kasus ini.</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                xgb_resistant   = class_res["diagnosis"] == "RESISTEN"
                bilstm_resistant = reg_res["fold_change"] >= reg_res["clinical_cutoff"]

                if xgb_resistant == bilstm_resistant:
                    _status = "KONSISTEN"
                    if xgb_resistant:
                        _c, _bg, _bd = "#C0392B", "rgba(192,57,43,0.07)", "rgba(192,57,43,0.28)"
                        _icon = "🔴"
                        _text = ("XGBoost dan Bi-LSTM <strong>sama-sama mendeteksi resistensi</strong>. "
                                 "Pertimbangkan evaluasi penggantian regimen ARV bersama klinisi.")
                    else:
                        _c, _bg, _bd = "#1A7A4A", "rgba(26,122,74,0.07)", "rgba(26,122,74,0.28)"
                        _icon = "🟢"
                        _text = ("XGBoost dan Bi-LSTM <strong>sama-sama menunjukkan virus masih sensitif</strong>. "
                                 "Terapi ARV yang berjalan saat ini dinilai masih efektif.")
                else:
                    _status = "TIDAK KONSISTEN"
                    _c, _bg, _bd = "#B7770D", "rgba(183,119,13,0.07)", "rgba(183,119,13,0.35)"
                    _icon = "⚠️"
                    _text = ("Hasil XGBoost dan Bi-LSTM <strong>tidak saling mendukung</strong>. "
                             "Disarankan pemeriksaan klinis lanjutan atau uji genotipe konfirmatori.")

                st.markdown(f"""
                <div style="background:{_bg};border:1px solid {_bd};border-radius:10px;
                            padding:0.9rem 1.5rem;margin-top:0.75rem;
                            display:flex;align-items:flex-start;gap:1rem">
                    <div style="font-size:1.5rem;margin-top:0.05rem">{_icon}</div>
                    <div>
                        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                                    text-transform:uppercase;color:{_c};margin-bottom:0.25rem;
                                    font-family:'IBM Plex Mono',monospace">
                            Kesimpulan Kedua Model &nbsp;·&nbsp; {_status}
                        </div>
                        <div style="font-size:0.82rem;color:var(--text-secondary);line-height:1.5">{_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if show_gauge or reg_res is not None:
            st.markdown("---")
            col_gauge, col_fc_interp = st.columns([1, 1], gap="medium")
            with col_gauge:
                if show_gauge:
                    st.markdown("##### 🎯 Indikator Probabilitas Klasifikasi")
                    gauge_fig = render_probability_gauge(class_res["probability"], class_res["threshold"])
                    st.pyplot(gauge_fig, use_container_width=True)
            with col_fc_interp:
                if reg_res is not None:
                    st.markdown("##### 📐 Interpretasi Klinis Fold Change")
                    st.markdown(render_fold_change_interpretation(reg_res["fold_change"]))
                    m1, m2 = st.columns(2)
                    m1.metric("Estimasi Fold Change", f"{reg_res['fold_change']:.2f}×", delta=reg_res["severity"])
                    m2.metric("Probabilitas Resistensi", f"{class_res['probability']:.3f}",
                              delta=f"Cutoff: {class_res['threshold']:.3f}")

        st.markdown("---")
        st.markdown("### 🧠 Transparansi Atribusi Fitur Medis — Explainable AI")
        st.markdown("""
        Sistem tidak beroperasi secara *black-box*. Visualisasi Beeswarm Plot di bawah menunjukkan transparansi penalaran matematis model, memetakan seberapa besar kontribusi dan tingkat pengaruh setiap mutasi genetik terhadap keputusan klasifikasi akhir.
        """)

        col_shap, col_top = st.columns([3, 1], gap="medium")
        with col_shap:
            st.markdown("##### Analisis Dampak Fitur Global (SHAP)")
            st.pyplot(r["shap_fig"], use_container_width=True)
        with col_top:
            st.markdown("##### Fitur Mutasi Paling Signifikan")
            if r["top_features"]:
                for feat, val in r["top_features"].items():
                    normalized = min(val / max(r["top_features"].values()), 1.0) if max(r["top_features"].values()) > 0 else 0
                    bar_width = int(normalized * 100)
                    st.markdown(f"""
                    <div style="margin-bottom:0.7rem">
                        <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:0.2rem">
                            <span style="color:var(--text-primary);font-family:'IBM Plex Mono',monospace;font-weight:600">{feat}</span>
                            <span style="color:var(--text-secondary)">{val:.4f}</span>
                        </div>
                        <div style="background:rgba(59,130,246,0.1);border-radius:3px;height:5px">
                            <div style="background:linear-gradient(90deg,#3B82F6,#14B8A6);width:{bar_width}%;height:5px;border-radius:3px"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("📊 Ringkasan Metadata Eksekusi", expanded=False):
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Latensi", f"{r['total_elapsed_ms']:.0f} ms")
            col_m2.metric("XGBoost Latency", f"{class_res['elapsed_ms']:.1f} ms")
            col_m3.metric("Bi-LSTM Latency", f"{reg_res['elapsed_ms']:.1f} ms" if reg_res else "N/A")
            col_m4.metric("Mutasi Terdeteksi", str(len(r["mutations"])))
            st.markdown(f"""
            **Sekuens Input:** `{r['seq_clean'][:120]}{'...' if len(r['seq_clean']) > 120 else ''}`

            **Mutasi Terparse:** {', '.join(f'`{m}`' for m in r['mutations']) if r['mutations'] else '*Tidak ada notasi mutasi standar yang valid terdeteksi.*'}

            **Obat Dianalisis:** {r['drug']} ({get_drug_class(r['drug'])})
            **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Mode Eksekusi:** {run_mode}
            """)


# ---------------------------------------------------------------------------
# TAB 2: RIWAYAT ANALISIS
# ---------------------------------------------------------------------------
with tab_history:
    st.markdown("### 📋 Riwayat Analisis Sesi Ini")

    if not st.session_state.analysis_history:
        st.markdown("""
        <div style="text-align:center;color:var(--text-muted);padding:3rem;font-size:0.85rem">
            Belum ada analisis yang dijalankan pada sesi ini.
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🗑️ Bersihkan Riwayat", type="secondary"):
            st.session_state.analysis_history = []
            st.rerun()

        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df.columns = ["Waktu", "Obat", "Sekuens", "Diagnosis", "Probabilitas", "Fold Change", "Severitas"]

        def style_diagnosis(val):
            if val == "RESISTEN":
                return "color: #EF4444; font-weight: bold;"
            return "color: #22C55E; font-weight: bold;"

        styled_df = history_df.style.map(style_diagnosis, subset=["Diagnosis"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📈 Statistik Agregat Sesi")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        n_resistant = sum(1 for h in st.session_state.analysis_history if h["diagnosis"] == "RESISTEN")
        n_total = len(st.session_state.analysis_history)
        col_s1.metric("Total Analisis", n_total)
        col_s2.metric("Kasus Resisten", n_resistant)
        col_s3.metric("Kasus Rentan", n_total - n_resistant)
        col_s4.metric("Rata-rata FC",
                      f"{sum(h['fold_change'] for h in st.session_state.analysis_history) / n_total:.2f}×"
                      if n_total else "—")


# ---------------------------------------------------------------------------
# TAB 3: INFORMASI SISTEM
# ---------------------------------------------------------------------------
with tab_info:
    col_i1, col_i2 = st.columns([1.2, 0.8], gap="large")

    with col_i1:
        st.markdown("### ℹ️ Tentang Sistem CDSS HIV")
        st.markdown("""
        Sistem CDSS (Clinical Decision Support System) ini merupakan **purwarupa penelitian** yang mengimplementasikan arsitektur 
        Dual-Pipeline untuk memprediksi status resistensi genotipe-fenotipe obat HIV.

        ---
        **🔬 Arsitektur Eksekusi Dual-Pipeline + SHAP:**

        **Pipeline A — Klasifikasi Diagnostik (XGBoost)**
        Mengekstraksi Matriks Biner Mutasi untuk memberikan diagnosis kebal/rentan dengan sensitivitas maksimum. Performa dioptimalkan melalui metrik AUC-ROC dan ambang batas dinamis (Dynamic Threshold) per obat.

        **Pipeline B — Regresi Kuantitatif (Bi-LSTM)**
        Mengolah tokenisasi sekuens temporal untuk memprediksi nilai Fold Change (FC) yang merepresentasikan tingkat perubahan sensitivitas virus.

        **Lapisan Transparansi — Explainable AI (SHAP)**
        Transparansi penuh melalui visualisasi atribusi fitur menggunakan SHAP untuk memperlihatkan bobot pengaruh tiap mutasi terhadap hasil prediksi dari arsitektur Dual-Pipeline.

        ---
        **📏 Interpretasi Tingkat Keparahan Resistensi (Fold Change):**
        - Rentan (Susceptible): FC < 3.5× — Virus masih sensitif terhadap terapi ARV.
        - Resistensi Rendah: 3.5× ≤ FC < 10× — Kemungkinan terjadi penurunan efektivitas klinis obat.
        - Resistensi Sedang: 10× ≤ FC < 30× — Terdapat kompromi signifikan pada efikasi terapi.
        - Resistensi Tinggi: FC ≥ 30× — Terapi ini kemungkinan besar sudah tidak efektif dan harus dihindari.

        ---
        **⚠️ Peringatan Penggunaan Klinis:**
        Purwarupa komputasional ini berstatus eksperimental (berbasis Machine Learning). Hasil prediksi **tidak boleh** dijadikan substitusi absolut atas uji laboratorium in-vitro dan diagnosis final dari tenaga kesehatan bersertifikat.
        """)

    with col_i2:
        st.markdown("### 📂 Standar Kualitas Purwarupa")
        st.markdown("""
        **Dimensi 1 — Kualitas Prediktif:**
        - ✅ Evaluasi berbasis sensitivitas klinis tinggi dan minimalisasi derau kesalahan
        - ✅ Optimalisasi AUC-ROC pada alur klasifikasi
        - ✅ Minimalisasi RMSE pada alur regresi
        - ✅ SHAP Beeswarm Plot

        **Dimensi 2 — Kualitas Fungsional:**
        - ✅ Purwarupa fungsional untuk pengolahan paralel dengan latensi komputasi yang terukur
        - ✅ Validasi IUPAC defensive programming
        - ✅ Eksekusi paralel ThreadPoolExecutor
        - ✅ Lazy loading Bi-LSTM per obat
        - ✅ Error handling informatif

        **Dimensi 3 — Kualitas Proses:**
        - ✅ Menggunakan siklus pengembangan data science terstruktur berbasis CRISP-DM
        - ✅ Caching O(1) dengan `@st.cache_resource`
        - ✅ Logging terstruktur
        - ✅ Reproduksibilitas komputasional
        """)




# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown(f"""
<div class="cdss-footer">
    Maulida Suryaning Aisha &nbsp;·&nbsp; 21106050069 &nbsp;·&nbsp; Informatika &nbsp;·&nbsp; {datetime.now().strftime('%Y')} &nbsp;·&nbsp; <em>Purwarupa Skripsi — Not for Clinical Use</em>
</div>
""", unsafe_allow_html=True)