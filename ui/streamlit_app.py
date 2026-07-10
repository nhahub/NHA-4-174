"""
ui/streamlit_app.py
─────────────────────
PERSON 5'S FILE

The web interface for Voxa — simple, clean, easy to use.

Run with:
    streamlit run ui/streamlit_app.py
"""

import streamlit as st
import os
import sys
import tempfile
import json
from datetime import datetime

# Make the project root importable (so `core` and `agents` packages resolve)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.orchestrator import VoxaOrchestrator


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Voxa — Accessible Speech-to-Text",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════
# STYLING — simple, calm, high-contrast, accessible design
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Overall page */
    .main {
        background-color: #FAFAF8;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    /* Title block */
    .voxa-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1A1A1A;
        margin-bottom: 0.1rem;
    }
    .voxa-subtitle {
        font-size: 1.05rem;
        color: #666660;
        margin-bottom: 1.8rem;
    }

    /* Section cards */
    .voxa-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E4DF;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .voxa-card-title {
        font-size: 1.0rem;
        font-weight: 600;
        color: #1A1A1A;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .voxa-card-text {
        font-size: 0.98rem;
        line-height: 1.65;
        color: #33332E;
        white-space: pre-wrap;
    }

    /* Metric badges */
    .voxa-badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.6rem;
    }
    .voxa-badge {
        background-color: #F1F1EC;
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.8rem;
        color: #444440;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1A1A1A;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #333330;
        color: #FFFFFF;
    }

    /* Download buttons */
    .stDownloadButton>button {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #D8D7D0;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD THE PIPELINE (only once — cached across interactions)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_orchestrator():
    """
    Loads all 3 agents ONE TIME and keeps them in memory.
    Without caching, Streamlit would reload the models on every click,
    which would be extremely slow.
    """
    return VoxaOrchestrator()


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="voxa-title">🎙️ Voxa</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="voxa-subtitle">Turn spoken audio into clear, accessible text — '
    'built for deaf and hard-of-hearing users.</div>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD AUDIO
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="voxa-card">', unsafe_allow_html=True)
st.markdown('<div class="voxa-card-title">1. Upload your audio</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Choose an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(
        f'<div class="voxa-badge-row">'
        f'<span class="voxa-badge">📄 {uploaded_file.name}</span>'
        f'<span class="voxa-badge">💾 {file_size_mb:.1f} MB</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# STEP 2 — PROCESS BUTTON
# ══════════════════════════════════════════════════════════════

process_clicked = st.button(
    "▶  Process Audio",
    disabled=(uploaded_file is None),
    use_container_width=True,
)


# ══════════════════════════════════════════════════════════════
# STEP 3 — RUN PIPELINE + SHOW RESULTS
# ══════════════════════════════════════════════════════════════

if process_clicked and uploaded_file is not None:

    # Save the uploaded file to a temporary path so agents can read it
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_audio_path = tmp_file.name

    with st.spinner("Loading models... this can take a minute on first run"):
        orchestrator = load_orchestrator()

    progress_placeholder = st.empty()
    progress_placeholder.info("🎧 Transcribing → simplifying → summarizing...")

    result = orchestrator.run(temp_audio_path)

    progress_placeholder.empty()

    # Clean up the temp file
    os.remove(temp_audio_path)

    if not result["success"]:
        st.error(f"Something went wrong: {result['error']}")
    else:
        st.session_state["voxa_result"] = result
        st.session_state["voxa_filename"] = uploaded_file.name


# ══════════════════════════════════════════════════════════════
# STEP 4 — DISPLAY RESULTS (persists across reruns via session_state)
# ══════════════════════════════════════════════════════════════

if "voxa_result" in st.session_state:
    result = st.session_state["voxa_result"]

    st.success(f"✓ Done in {result['elapsed_seconds']}s")

    # ── Transcript ────────────────────────────────────────────────
    st.markdown('<div class="voxa-card">', unsafe_allow_html=True)
    st.markdown('<div class="voxa-card-title">📝 Original Transcript</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voxa-card-text">{result["transcript"]}</div>', unsafe_allow_html=True)
    conf = result["transcription_meta"].get("confidence", 0)
    lang = result["transcription_meta"].get("language", "—")
    st.markdown(
        f'<div class="voxa-badge-row">'
        f'<span class="voxa-badge">🌐 Language: {lang}</span>'
        f'<span class="voxa-badge">🎯 Confidence: {conf:.0%}</span>'
        f'<span class="voxa-badge">📖 {result["transcription_meta"].get("word_count", 0)} words</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Simplified Text ──────────────────────────────────────────
    st.markdown('<div class="voxa-card">', unsafe_allow_html=True)
    st.markdown('<div class="voxa-card-title">✨ Simplified Text</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voxa-card-text">{result["simplified_text"]}</div>', unsafe_allow_html=True)
    sm = result["simplification_meta"]
    st.markdown(
        f'<div class="voxa-badge-row">'
        f'<span class="voxa-badge">{sm["original_words"]} → {sm["simplified_words"]} words</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary ──────────────────────────────────────────────────
    st.markdown('<div class="voxa-card">', unsafe_allow_html=True)
    st.markdown('<div class="voxa-card-title">📊 Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voxa-card-text">{result["summary"]}</div>', unsafe_allow_html=True)
    sum_meta = result["summarization_meta"]
    st.markdown(
        f'<div class="voxa-badge-row">'
        f'<span class="voxa-badge">{sum_meta["original_words"]} → {sum_meta["summary_words"]} words</span>'
        f'<span class="voxa-badge">Compression: {sum_meta["compression_ratio"]:.0%}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Download buttons ─────────────────────────────────────────
    st.markdown("#####")
    col1, col2, col3 = st.columns(3)

    base_name = os.path.splitext(st.session_state.get("voxa_filename", "voxa"))[0]

    with col1:
        st.download_button(
            "⬇ Transcript (.txt)",
            data=result["transcript"],
            file_name=f"{base_name}_transcript.txt",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ Simplified (.txt)",
            data=result["simplified_text"],
            file_name=f"{base_name}_simplified.txt",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇ Full Result (.json)",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"{base_name}_result.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("#####")
st.markdown(
    '<div style="text-align:center; color:#999990; font-size:0.82rem; padding-top:1rem;">'
    'Voxa — DEPI Generative AI Course Project'
    '</div>',
    unsafe_allow_html=True
)
