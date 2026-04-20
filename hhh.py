# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import time
import tempfile
import re
import hashlib
import pickle
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

from pypdf import PdfReader
from gtts import gTTS


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
CHUNK_SIZE        = 1200
CHUNK_OVERLAP     = 120
TOP_K             = 8
MAX_MAP_WORKERS   = 4
MAX_MAP_CHUNKS    = 40
FAISS_CACHE_DIR   = Path(".faiss_cache")
FAISS_CACHE_DIR.mkdir(exist_ok=True)
EMBED_BATCH_SIZE  = 64
EMBED_MAX_WORKERS = 1
LARGE_PDF_THRESHOLD = 150


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UnifiedRAG · Docs & Web",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  SESSION STATE — THEME
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True




# ─────────────────────────────────────────────
#  CUSTOM CSS — ADAPTIVE LIGHT / DARK THEMES
# ─────────────────────────────────────────────
def get_css(dark: bool) -> str:
    if dark:
        vars_block = """
    :root {
        --bg:              #07080c;
        --surface:         #0f1117;
        --surface2:        #161820;
        --surface3:        #1a1d28;
        --border:          #1e2130;
        --border2:         #262a3a;
        --doc-accent:      #e8704a;
        --doc-accent2:     #f08060;
        --web-accent:      #4ac8e8;
        --gold:            #f0c060;
        --success:         #50e89a;
        --text:            #dde2f0;
        --text2:           #a0a8c0;
        --muted:           #5a6080;
        --muted2:          #3a4060;
        --shadow:          rgba(0,0,0,0.55);
        --shadow2:         rgba(0,0,0,0.30);
        --input-bg:        #07080c;
        --tag-doc-bg:      rgba(232,112,74,0.12);
        --tag-doc-border:  rgba(232,112,74,0.35);
        --tag-web-bg:      rgba(74,200,232,0.10);
        --tag-web-border:  rgba(74,200,232,0.35);
        --cache-bg:        rgba(80,232,154,0.10);
        --cache-border:    rgba(80,232,154,0.35);
        --chunk-bg:        #07080c;
        --glow-doc:        rgba(232,112,74,0.13);
        --glow-web:        rgba(74,200,232,0.10);
        --radius:          14px;
        --radius-sm:       10px;
    }"""
    else:
        vars_block = """
    :root {
        --bg:              #f0f2f8;
        --surface:         #ffffff;
        --surface2:        #f7f8fc;
        --surface3:        #eceef6;
        --border:          #d8dcea;
        --border2:         #c2c8dc;
        --doc-accent:      #c04010;
        --doc-accent2:     #d85020;
        --web-accent:      #0878a8;
        --gold:            #986000;
        --success:         #166638;
        --text:            #12182e;
        --text2:           #3a4260;
        --muted:           #6878a0;
        --muted2:          #98a0bc;
        --shadow:          rgba(10,20,60,0.14);
        --shadow2:         rgba(10,20,60,0.07);
        --input-bg:        #ffffff;
        --tag-doc-bg:      rgba(192,64,16,0.08);
        --tag-doc-border:  rgba(192,64,16,0.28);
        --tag-web-bg:      rgba(8,120,168,0.08);
        --tag-web-border:  rgba(8,120,168,0.28);
        --cache-bg:        rgba(22,102,56,0.08);
        --cache-border:    rgba(22,102,56,0.28);
        --chunk-bg:        #f7f8fc;
        --glow-doc:        rgba(192,64,16,0.06);
        --glow-web:        rgba(8,120,168,0.06);
        --radius:          14px;
        --radius-sm:       10px;
    }"""

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

    {vars_block}

    /* ════ BASE RESET ════ */
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}

    #MainMenu, footer, header {{ visibility: hidden !important; }}

    /* ════ SIDEBAR — always visible, no toggle ════ */
    [data-testid="collapsedControl"],
    [data-testid="baseButton-headerNoPadding"],
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] > div:first-child > div:first-child > button {{
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] {{
        transform: none !important;
        width: 17rem !important;
        min-width: 17rem !important;
        max-width: 17rem !important;
        left: 0 !important;
        visibility: visible !important;
        display: block !important;
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 3px 0 20px var(--shadow2) !important;
        transition: background 0.3s ease, border-color 0.3s ease;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding: 1.4rem 1.1rem !important;
        max-width: 100% !important;
    }}

    /* ════ MAIN CONTENT — full width ════ */
    .block-container {{
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
        background: var(--bg) !important;
    }}

    section.main > div {{
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}

    [data-testid="stHorizontalBlock"] {{
        gap: 1rem !important;
        align-items: stretch !important;
    }}

    /* ════ HERO ════ */
    .hero-wrap {{
        position: relative;
        padding: 3rem 0 2rem;
        overflow: hidden;
        width: 100%;
    }}
    .hero-bg-glow {{
        position: absolute;
        top: -80px; left: -100px;
        width: 600px; height: 400px;
        background: radial-gradient(ellipse, var(--glow-doc) 0%, transparent 68%);
        pointer-events: none;
    }}
    .hero-bg-glow2 {{
        position: absolute;
        top: -60px; right: -80px;
        width: 500px; height: 350px;
        background: radial-gradient(ellipse, var(--glow-web) 0%, transparent 68%);
        pointer-events: none;
    }}
    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 400;
        letter-spacing: 0.24em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }}
    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: clamp(3.2rem, 6.5vw, 5.2rem);
        font-weight: 900;
        letter-spacing: -3px;
        line-height: 1.0;
        margin: 0;
    }}
    .hero-title .doc-c {{ color: var(--doc-accent); }}
    .hero-title .web-c {{ color: var(--web-accent); }}
    .hero-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.80rem;
        color: var(--muted);
        margin-top: 1rem;
        letter-spacing: 0.07em;
    }}

    /* ════ CARDS ════ */
    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px var(--shadow2);
        transition: background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        width: 100%;
        box-sizing: border-box;
    }}
    .card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--border2), transparent);
        border-radius: var(--radius) var(--radius) 0 0;
    }}
    .card.doc-card::before {{
        background: linear-gradient(90deg, transparent 0%, var(--doc-accent) 50%, transparent 100%);
        opacity: 0.6;
    }}
    .card.web-card::before {{
        background: linear-gradient(90deg, transparent 0%, var(--web-accent) 50%, transparent 100%);
        opacity: 0.6;
    }}
    .card.answer-card::before {{
        background: linear-gradient(90deg, var(--doc-accent) 0%, var(--web-accent) 100%);
        opacity: 0.75;
    }}

    .card-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        display: block;
    }}
    .card-label.doc-label {{ color: var(--doc-accent); }}
    .card-label.web-label {{ color: var(--web-accent); }}
    .card-label.ask-label {{ color: var(--gold); }}
    .card-label.ans-label {{ color: var(--success); }}

    /* ════ INPUTS ════ */
    .stTextInput > div > div > input,
    .stTextArea  > div > div > textarea {{
        background: var(--input-bg) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.92rem !important;
        padding: 0.85rem 1.1rem !important;
        min-height: 50px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 1px 4px var(--shadow2) !important;
    }}
    .stTextInput > div > div > input::placeholder,
    .stTextArea  > div > div > textarea::placeholder {{
        color: var(--muted) !important;
        opacity: 1 !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea  > div > div > textarea:focus {{
        border-color: var(--web-accent) !important;
        box-shadow: 0 0 0 3px rgba(74,200,232,0.14) !important;
        outline: none !important;
    }}

    /* ════ BUTTONS ════ */
    .stButton > button {{
        background: linear-gradient(135deg, var(--doc-accent) 0%, var(--doc-accent2) 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.07em !important;
        padding: 0.75rem 1.4rem !important;
        min-height: 48px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        box-shadow: 0 3px 12px var(--shadow2) !important;
        cursor: pointer !important;
    }}
    .stButton > button:hover {{
        opacity: 0.91 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px var(--shadow) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px var(--shadow2) !important;
    }}
    .stButton > button:disabled {{
        background: var(--surface2) !important;
        color: var(--muted) !important;
        opacity: 0.55 !important;
        transform: none !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }}

    /* ════ FILE UPLOADER ════ */
    [data-testid="stFileUploader"] {{
        background: var(--input-bg) !important;
        border: 2px dashed var(--border2) !important;
        border-radius: var(--radius-sm) !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--doc-accent) !important;
        background: var(--surface2) !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: var(--text2) !important;
    }}

    /* ════ SELECTBOX ════ */
    .stSelectbox > div > div {{
        background: var(--input-bg) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.83rem !important;
        min-height: 44px !important;
        transition: border-color 0.2s ease;
    }}
    .stSelectbox > div > div:focus-within {{
        border-color: var(--web-accent) !important;
    }}
    [data-baseweb="select"] [role="listbox"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius-sm) !important;
    }}
    [data-baseweb="select"] [role="option"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
    }}
    [data-baseweb="select"] [role="option"]:hover {{
        background: var(--surface2) !important;
    }}

    /* ════ RADIO ════ */
    .stRadio > div {{ gap: 0.6rem !important; }}
    .stRadio label {{
        font-size: 0.90rem !important;
        font-family: 'Outfit', sans-serif !important;
        color: var(--text) !important;
        cursor: pointer !important;
    }}
    .stRadio [data-testid="stMarkdownContainer"] p {{
        color: var(--text) !important;
    }}

    /* ════ EXPANDER ════ */
    .streamlit-expanderHeader {{
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
        color: var(--text2) !important;
        padding: 0.8rem 1rem !important;
    }}
    .streamlit-expanderContent {{
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
        padding: 1rem !important;
    }}

    /* ════ ANSWER CONTENT ════ */
    .answer-content {{
        font-size: 1.00rem;
        line-height: 1.90;
        color: var(--text);
        white-space: pre-wrap;
    }}

    /* ════ SOURCE BAR ════ */
    .source-bar {{
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1.3rem;
        margin-bottom: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: var(--text2);
        display: flex;
        align-items: center;
        gap: 0.9rem;
        box-shadow: 0 1px 6px var(--shadow2);
        flex-wrap: wrap;
    }}
    .source-dot {{
        width: 9px; height: 9px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .source-dot.doc {{
        background: var(--doc-accent);
        box-shadow: 0 0 8px var(--doc-accent);
    }}
    .source-dot.web {{
        background: var(--web-accent);
        box-shadow: 0 0 8px var(--web-accent);
    }}

    /* ════ DOC CHUNKS ════ */
    .doc-chunk {{
        background: var(--chunk-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.81rem;
        color: var(--text2);
        line-height: 1.75;
    }}
    .doc-chunk-num {{
        font-size: 0.63rem;
        color: var(--gold);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        font-weight: 700;
    }}

    /* ════ TAGS & BADGES ════ */
    .file-tag {{
        display: inline-block;
        background: var(--tag-doc-bg);
        border: 1px solid var(--tag-doc-border);
        border-radius: 6px;
        padding: 0.18rem 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        color: var(--doc-accent);
        margin: 0.25rem 0.3rem 0.25rem 0;
    }}
    .url-tag {{
        display: inline-block;
        background: var(--tag-web-bg);
        border: 1px solid var(--tag-web-border);
        border-radius: 6px;
        padding: 0.18rem 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        color: var(--web-accent);
        margin: 0.25rem 0.3rem 0.25rem 0;
        word-break: break-all;
    }}
    .cache-badge {{
        display: inline-block;
        background: var(--cache-bg);
        border: 1px solid var(--cache-border);
        border-radius: 6px;
        padding: 0.14rem 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        color: var(--success);
        margin-left: 0.6rem;
        vertical-align: middle;
        font-weight: 600;
    }}
    .badge-ready   {{ color: var(--success); font-weight: 600; }}
    .badge-pending {{ color: var(--gold);    font-weight: 600; }}

    /* ════ ALERTS ════ */
    .stAlert,
    div[data-testid="stAlert"] {{
        background: var(--surface2) !important;
        border-color: var(--border2) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
    }}
    .stAlert p,
    div[data-testid="stAlert"] p {{
        color: var(--text) !important;
    }}

    /* ════ CAPTIONS & MARKDOWN ════ */
    .stCaption, .stCaption p, small {{
        color: var(--muted) !important;
    }}
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {{
        color: var(--text) !important;
    }}

    /* ════ SIDEBAR TEXT ════ */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {{
        color: var(--text2) !important;
    }}

    /* ════ STATUS WIDGET ════ */
    [data-testid="stStatus"] {{
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text2) !important;
        border-radius: var(--radius-sm) !important;
    }}
    [data-testid="stStatus"] p {{
        color: var(--text2) !important;
    }}

    /* ════ FORM ════ */
    [data-testid="stForm"] {{
        border: 1px solid var(--border) !important;
        background: var(--surface2) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.6rem 0.8rem !important;
    }}

    /* ════ SPINNER ════ */
    .stSpinner > div {{
        border-top-color: var(--doc-accent) !important;
    }}

    /* ════ DIVIDER ════ */
    .divider {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }}

    /* ════ PROGRESS BAR ════ */
    [data-testid="stProgressBar"] > div > div {{
        background: linear-gradient(90deg, var(--doc-accent), var(--web-accent)) !important;
        border-radius: 4px;
    }}
    [data-testid="stProgressBar"] > div {{
        background: var(--surface3) !important;
        border-radius: 4px;
    }}

    /* ════ AUDIO PLAYER ════ */
    audio {{
        width: 100% !important;
        border-radius: var(--radius-sm) !important;
        background: var(--surface2) !important;
    }}

    </style>
    """

# Inject CSS based on current theme
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CACHED RESOURCE FACTORIES
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings(model: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ─────────────────────────────────────────────
#  FILE HASHING
# ─────────────────────────────────────────────
def get_files_hash(uploaded_files: list) -> str:
    h = hashlib.md5()
    for uf in sorted(uploaded_files, key=lambda f: f.name):
        h.update(uf.name.encode())
        h.update(uf.read())
        uf.seek(0)
    return h.hexdigest()


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


# ─────────────────────────────────────────────
#  SEQUENTIAL BATCHED EMBEDDING
# ─────────────────────────────────────────────
def embed_documents_fast(
    texts: list[str],
    embeddings: HuggingFaceEmbeddings,
    progress_cb=None,
) -> list:
    batches  = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    all_vecs: list = []
    for i, batch in enumerate(batches):
        all_vecs.extend(embeddings.embed_documents(batch))
        if progress_cb:
            progress_cb((i + 1) / len(batches))
    return all_vecs


# ─────────────────────────────────────────────
#  FAISS DISK CACHE
# ─────────────────────────────────────────────
def load_faiss_cache(cache_key: str, embeddings) -> tuple:
    index_path = FAISS_CACHE_DIR / cache_key
    docs_path  = FAISS_CACHE_DIR / f"{cache_key}.docs.pkl"
    if index_path.exists() and docs_path.exists():
        try:
            vectors    = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            split_docs = pickle.loads(docs_path.read_bytes())
            return vectors, split_docs
        except Exception:
            pass
    return None, None


def save_faiss_cache(cache_key: str, vectors: FAISS, split_docs: list):
    index_path = FAISS_CACHE_DIR / cache_key
    docs_path  = FAISS_CACHE_DIR / f"{cache_key}.docs.pkl"
    try:
        vectors.save_local(str(index_path))
        docs_path.write_bytes(pickle.dumps(split_docs))
    except Exception:
        pass


# ─────────────────────────────────────────────
#  TEXT CLEANUP FOR TTS
# ─────────────────────────────────────────────
def clean_for_tts(text: str) -> str:
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"^[\-\*\•\▶\▸\▪\✦\⚡\✅\❌\➡\➤\●\◆]+\s*", " ", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\.\?\!\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "and so on",
        "LLM":  "large language model",
        "RAG":  "retrieval augmented generation",
        "PDF":  "P D F",
        "vs.":  "versus",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise and reliable assistant. Your ONLY source of truth is the <context> block — do NOT use prior knowledge.

Rules:
1. Answer strictly from the context.
2. If the answer is absent, say: "I could not find a relevant answer in the provided content."
3. Never speculate or invent facts.
4. Use bullet points for lists. Cite page/source when quoting.
5. Be thorough — include all relevant details.

<context>
{context}
</context>

Question: {input}

Answer:"""
)

MAP_PROMPT = PromptTemplate(
    input_variables=["docs", "question"],
    template="""You are reading a section of content.
Extract ONLY information relevant to the question. If nothing is relevant, reply exactly: "IRRELEVANT"

Section:
{docs}

Question: {question}

Relevant information:"""
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["doc_summaries", "question"],
    template="""Synthesise the extracted information below into a single, comprehensive, well-structured answer.

Rules:
1. Answer ONLY from the extracts — no outside knowledge.
2. If none contain relevant info, say: "I could not find a relevant answer in the provided content."
3. Use bullet points for lists. Cite document/page/URL where available.
4. Be thorough and detailed.

Extracted information:
{doc_summaries}

Question: {question}

Final Answer:"""
)

HIERARCHICAL_REDUCE_PROMPT = PromptTemplate(
    input_variables=["partial_summaries", "question"],
    template="""Below are partial summaries from different sections of a large document.
Combine them into one coherent, comprehensive answer.

Rules:
1. Use ONLY the partial summaries — no outside knowledge.
2. Eliminate duplicates; keep all unique facts.
3. Structure with bullet points where appropriate.
4. If no relevant info exists, say so clearly.

Partial summaries:
{partial_summaries}

Question: {question}

Final synthesised answer:"""
)


# ─────────────────────────────────────────────
#  FAST PDF LOADER
# ─────────────────────────────────────────────
def load_pdfs_fast(uploaded_files: list) -> list[Document]:
    docs = []
    for uf in uploaded_files:
        uf.seek(0)
        try:
            reader = PdfReader(uf)
            for page_num, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uf.name, "page": page_num + 1},
                ))
        except Exception as e:
            st.warning(f"⚠️ Could not read {uf.name}: {e}")
    return docs


# ─────────────────────────────────────────────
#  BUILD FAISS FROM DOCUMENTS
# ─────────────────────────────────────────────
def build_faiss_from_docs(split_docs: list, embeddings, progress_bar=None) -> FAISS:
    texts     = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]

    def _progress(pct):
        if progress_bar:
            progress_bar.progress(pct, text=f"Embedding… {int(pct * 100)}%")

    all_embeddings = embed_documents_fast(texts, embeddings, progress_cb=_progress)
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_embeddings)),
        embedding=embeddings,
        metadatas=metadatas,
    )


# ─────────────────────────────────────────────
#  BUILD FROM PDFS
# ─────────────────────────────────────────────
def build_from_pdfs(uploaded_files: list, embedding_model: str) -> dict:
    t0         = time.time()
    embeddings = get_embeddings(embedding_model)
    cache_key  = get_files_hash(uploaded_files)

    cached_vectors, cached_docs = load_faiss_cache(cache_key, embeddings)
    if cached_vectors is not None:
        return {
            "vectors":    cached_vectors,
            "all_docs":   cached_docs,
            "num_docs":   len(cached_docs),
            "num_files":  len(uploaded_files),
            "elapsed":    round(time.time() - t0, 2),
            "from_cache": True,
        }

    docs = load_pdfs_fast(uploaded_files)
    if not docs:
        raise ValueError("No readable content found in the uploaded PDFs.")

    splitter     = get_splitter()
    split_docs   = splitter.split_documents(docs)
    progress_bar = st.progress(0.0, text="Embedding chunks…")
    vectors      = build_faiss_from_docs(split_docs, embeddings, progress_bar=progress_bar)
    progress_bar.empty()
    save_faiss_cache(cache_key, vectors, split_docs)

    return {
        "vectors":    vectors,
        "all_docs":   split_docs,
        "num_docs":   len(split_docs),
        "num_files":  len(uploaded_files),
        "elapsed":    round(time.time() - t0, 2),
        "from_cache": False,
    }


# ─────────────────────────────────────────────
#  BUILD FROM URL
# ─────────────────────────────────────────────
def build_from_url(url: str, embedding_model: str) -> dict:
    t0         = time.time()
    embeddings = get_embeddings(embedding_model)
    cache_key  = "url_" + get_url_hash(url)

    cached_vectors, cached_docs = load_faiss_cache(cache_key, embeddings)
    if cached_vectors is not None:
        return {
            "vectors":    cached_vectors,
            "all_docs":   cached_docs,
            "num_docs":   len(cached_docs),
            "elapsed":    round(time.time() - t0, 2),
            "from_cache": True,
        }

    loader     = WebBaseLoader(url)
    docs       = loader.load()
    if not docs:
        raise ValueError("No content could be loaded from the provided URL.")

    splitter     = get_splitter()
    split_docs   = splitter.split_documents(docs)
    progress_bar = st.progress(0.0, text="Embedding chunks…")
    vectors      = build_faiss_from_docs(split_docs, embeddings, progress_bar=progress_bar)
    progress_bar.empty()
    save_faiss_cache(cache_key, vectors, split_docs)

    return {
        "vectors":    vectors,
        "all_docs":   split_docs,
        "num_docs":   len(split_docs),
        "elapsed":    round(time.time() - t0, 2),
        "from_cache": False,
    }


# ─────────────────────────────────────────────
#  MAP-ONE (thread worker)
# ─────────────────────────────────────────────
def _map_one(args: tuple) -> tuple[int, str]:
    idx, doc, question, llm, map_chain = args
    try:
        result = map_chain.invoke({"docs": doc.page_content, "question": question})
        text   = result.get("text", "").strip()
        if not text or text.strip().upper() == "IRRELEVANT":
            return idx, ""
        src  = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        return idx, f"[{src} · p.{page}]\n{text}"
    except Exception:
        return idx, ""


# ─────────────────────────────────────────────
#  SMART SAMPLING
# ─────────────────────────────────────────────
def smart_sample_docs(all_docs: list, question: str, vectors, k: int) -> list[Document]:
    half = k // 2
    try:
        sim_docs    = vectors.similarity_search(question, k=half * 2)
        sim_indices: set[int] = set()
        for d in sim_docs:
            for i, ad in enumerate(all_docs):
                if ad.page_content == d.page_content:
                    sim_indices.add(i)
                    break
        sim_indices = set(list(sim_indices)[:half])
    except Exception:
        sim_indices = set()

    step            = max(1, len(all_docs) // half)
    uniform_indices = set(range(0, len(all_docs), step))
    combined        = sorted(sim_indices | uniform_indices)
    return [all_docs[i] for i in combined[:k]]


# ─────────────────────────────────────────────
#  MAPREDUCE WITH HIERARCHICAL REDUCE
# ─────────────────────────────────────────────
def ask_full_mapreduce(question: str, all_docs: list, llm, vectors=None) -> dict:
    t0 = time.time()

    if len(all_docs) > MAX_MAP_CHUNKS:
        docs_to_process = (
            smart_sample_docs(all_docs, question, vectors, MAX_MAP_CHUNKS)
            if vectors is not None
            else [all_docs[int(i * len(all_docs) / MAX_MAP_CHUNKS)] for i in range(MAX_MAP_CHUNKS)]
        )
    else:
        docs_to_process = all_docs

    map_chain   = LLMChain(llm=llm, prompt=MAP_PROMPT)
    tasks       = [(i, doc, question, llm, map_chain) for i, doc in enumerate(docs_to_process)]
    raw_results = {}
    done_count  = 0

    status = st.status(f"🔍 Scanning {len(docs_to_process)} sections…", expanded=False)

    with ThreadPoolExecutor(max_workers=MAX_MAP_WORKERS) as executor:
        futures = {executor.submit(_map_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, text = future.result()
            if text:
                raw_results[idx] = text
            done_count += 1
            status.update(
                label=f"🔍 Scanned {done_count}/{len(docs_to_process)} sections… ({len(raw_results)} relevant)"
            )

    status.update(label=f"✅ Scan complete — {len(raw_results)} relevant sections found", state="complete")

    map_results = [raw_results[k] for k in sorted(raw_results)]

    if not map_results:
        final = "I could not find a relevant answer in the provided content."
    else:
        REDUCE_BATCH = 12
        if len(map_results) > REDUCE_BATCH:
            partial_answers = []
            for i in range(0, len(map_results), REDUCE_BATCH):
                batch_text = "\n\n---\n\n".join(map_results[i:i + REDUCE_BATCH])
                r  = LLMChain(llm=llm, prompt=REDUCE_PROMPT).invoke(
                    {"doc_summaries": batch_text, "question": question}
                )
                pa = r.get("text", "").strip()
                if pa and "could not find" not in pa.lower():
                    partial_answers.append(pa)

            if partial_answers:
                result = LLMChain(llm=llm, prompt=HIERARCHICAL_REDUCE_PROMPT).invoke(
                    {"partial_summaries": "\n\n===\n\n".join(partial_answers), "question": question}
                )
                final = result.get("text", "").strip()
            else:
                final = "I could not find a relevant answer in the provided content."
        else:
            result = LLMChain(llm=llm, prompt=REDUCE_PROMPT).invoke(
                {"doc_summaries": "\n\n---\n\n".join(map_results), "question": question}
            )
            final = result.get("text", "").strip()

    return {
        "answer":    final,
        "context":   [],
        "map_count": len(map_results),
        "elapsed":   round(time.time() - t0, 3),
    }


# ─────────────────────────────────────────────
#  TARGETED RETRIEVAL
# ─────────────────────────────────────────────
def ask_targeted(question: str, vectors, llm) -> dict:
    t0              = time.time()
    document_chain  = create_stuff_documents_chain(llm, RAG_PROMPT)
    retriever       = vectors.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response        = retrieval_chain.invoke({"input": question})
    return {
        "answer":    response["answer"],
        "context":   response.get("context", []),
        "map_count": 0,
        "elapsed":   round(time.time() - t0, 3),
    }


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "source_type":       "PDF Documents",
    "num_docs":          0,
    "build_elapsed":     0.0,
    "last_q_elapsed":    0.0,
    "last_mode":         "full",
    "last_map_count":    0,
    "source_label":      "",
    "tts_audio_path":    None,
    "tts_source_text":   "",
    "sessions":          {},
    "active_session_id": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_active_session():
    sid = st.session_state.active_session_id
    if not sid:
        return None, None
    return sid, st.session_state.sessions.get(sid)


def set_active_session(session_id, data):
    st.session_state.sessions[session_id] = data
    st.session_state.active_session_id    = session_id


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.2rem;font-weight:900;'
        'letter-spacing:-0.5px;margin-bottom:0;">'
        '<span style="color:var(--doc-accent);">Info</span>'
        '<span style="color:var(--web-accent);">Wave</span>'
        '<span style="color:var(--gold);">AI</span>'
        '</p>'
        '<p style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
        'color:var(--muted);letter-spacing:0.15em;margin-top:0.1rem;">FULL-READ RAG</p>',
        unsafe_allow_html=True,
    )

    # ── Theme Toggle ──
    st.markdown('<hr style="border-color:var(--border);margin:0.8rem 0;"/>', unsafe_allow_html=True)
    current_icon = "☀️" if st.session_state.dark_mode else "🌙"
    current_label = "Switch to Light" if st.session_state.dark_mode else "Switch to Dark"
    if st.button(f"{current_icon}  {current_label}", use_container_width=True, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<hr style="border-color:var(--border);margin:1rem 0;"/>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
        index=0,
    )

    embedding_choice = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
        ],
        index=0,
        help="Downloaded from HuggingFace Hub and cached locally. No Ollama needed.",
    )

    st.markdown('<hr style="border-color:var(--border);margin:1rem 0;"/>', unsafe_allow_html=True)

    query_mode = st.radio(
        "Query Mode",
        ["📖 Full Read", "🎯 Targeted"],
        index=0,
    )
    is_full = "Full" in query_mode

    st.markdown('<hr style="border-color:var(--border);margin:1rem 0;"/>', unsafe_allow_html=True)

    cache_files = list(FAISS_CACHE_DIR.glob("*.docs.pkl"))
    st.markdown(
        f'<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:var(--muted);">'
        f'💾 Disk cache: <span style="color:var(--success);">{len(cache_files)} indexes saved</span>'
        f'</p>',
        unsafe_allow_html=True,
    )
    if cache_files and st.button("🗑 Clear Cache", use_container_width=True):
        for f in FAISS_CACHE_DIR.iterdir():
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    import shutil
                    shutil.rmtree(f)
            except Exception:
                pass
        st.rerun()

    st.markdown('<hr style="border-color:var(--border);margin:1rem 0;"/>', unsafe_allow_html=True)

    session_ids = list(st.session_state.sessions.keys())
    if session_ids:
        labels = [st.session_state.sessions[sid].get("label", sid) for sid in session_ids]
        idx    = 0
        if st.session_state.active_session_id in session_ids:
            idx = session_ids.index(st.session_state.active_session_id)
        chosen = st.selectbox(
            "Saved sources",
            options=list(range(len(session_ids))),
            format_func=lambda i: labels[i],
            index=idx,
        )
        st.session_state.active_session_id = session_ids[chosen]
    else:
        st.caption("No saved sources yet. Index a PDF or URL to create one.")


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-bg-glow"></div>'
    '<div class="hero-bg-glow2"></div>'
    '<p class="hero-eyebrow"></p>'
    '<h1 class="hero-title">'
    '<span class="doc-c">Info</span>'
    '<span class="web-c">Wave</span>'
    '<span style="color:var(--gold);"> AI</span>'
    '</h1>'
    '<p class="hero-sub">⚡ Retrieval-Augmented Generation</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STEP 1 — SOURCE SELECTION
# ─────────────────────────────────────────────
st.markdown(
    '<div class="card">'
    '<div class="card-label" style="color:var(--gold);">▸ Step 1 — Choose Your Source</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    doc_btn = st.button("📄  PDF Documents", use_container_width=True)
with col2:
    web_btn = st.button("🌐  Website URL", use_container_width=True)

if doc_btn:
    st.session_state.source_type = "PDF Documents"
if web_btn:
    st.session_state.source_type = "Website URL"

src_type = st.session_state.source_type
st.markdown(
    f'<p style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--muted);margin-top:0.5rem;">'
    f'Active source: <span style="color:{"var(--doc-accent)" if src_type == "PDF Documents" else "var(--web-accent)"};font-weight:600;">'
    f'{"📄 PDF Documents" if src_type == "PDF Documents" else "🌐 Website URL"}</span>'
    f'</p>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STEP 2 — LOAD & INDEX
# ─────────────────────────────────────────────
if src_type == "PDF Documents":
    st.markdown(
        '<div class="card doc-card">'
        '<div class="card-label doc-label">📥 Step 2 — Upload PDF Documents</div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        if uploaded_files:
            st.markdown(
                "".join(f'<span class="file-tag">📄 {f.name}</span>' for f in uploaded_files),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No files selected.")
    with col_btn:
        index_btn = st.button("⚡ Index", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if index_btn:
        if not GROQ_API_KEY:
            st.error("🔑 Groq API key missing — check your .env file.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one PDF.")
        else:
            names      = sorted([f.name for f in uploaded_files])
            session_id = "PDF: " + ", ".join(names)
            if session_id in st.session_state.sessions:
                st.info("✅ Already indexed — selected existing session.")
                st.session_state.active_session_id = session_id
            else:
                with st.spinner("🔄 Reading documents… (checking disk cache first)"):
                    try:
                        result = build_from_pdfs(uploaded_files, embedding_choice)
                        data = {
                            "label":         session_id,
                            "source_type":   "PDF Documents",
                            "vectors":       result["vectors"],
                            "all_docs":      result["all_docs"],
                            "num_docs":      result["num_docs"],
                            "build_elapsed": result["elapsed"],
                            "last_answer":   None,
                            "last_context":  [],
                            "from_cache":    result.get("from_cache", False),
                        }
                        set_active_session(session_id, data)
                        st.session_state.source_label = session_id
                        if result.get("from_cache"):
                            st.success(f"⚡ Loaded from disk cache in {result['elapsed']}s!")
                        else:
                            st.success(f"✅ Indexed {result['num_docs']} chunks in {result['elapsed']}s. Saved to disk cache.")
                    except Exception as e:
                        st.error(f"❌ {e}")
else:
    st.markdown(
        '<div class="card web-card">'
        '<div class="card-label web-label">🌐 Step 2 — Enter Website URL</div>',
        unsafe_allow_html=True,
    )
    col_url, col_btn = st.columns([4, 1])
    with col_url:
        url_input = st.text_input(
            label="URL",
            placeholder="https://example.com/docs/",
            label_visibility="collapsed",
        )
    with col_btn:
        process_btn = st.button("⚡ Load", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if process_btn:
        url = url_input.strip()
        if not GROQ_API_KEY:
            st.error("🔑 Groq API key missing — check your .env file.")
        elif not url:
            st.warning("⚠️ Please enter a URL.")
        elif not url.startswith(("http://", "https://")):
            st.error("❌ URL must start with http:// or https://")
        else:
            session_id = url
            if session_id in st.session_state.sessions:
                st.info("✅ Already indexed — selected existing session.")
                st.session_state.active_session_id = session_id
            else:
                with st.spinner("🔄 Fetching website content… (checking disk cache first)"):
                    try:
                        result = build_from_url(url, embedding_choice)
                        data = {
                            "label":         url,
                            "source_type":   "Website URL",
                            "vectors":       result["vectors"],
                            "all_docs":      result["all_docs"],
                            "num_docs":      result["num_docs"],
                            "build_elapsed": result["elapsed"],
                            "last_answer":   None,
                            "last_context":  [],
                            "from_cache":    result.get("from_cache", False),
                        }
                        set_active_session(session_id, data)
                        st.session_state.source_label = url
                        if result.get("from_cache"):
                            st.success(f"⚡ Loaded from disk cache in {result['elapsed']}s — zero re-embedding!")
                        else:
                            st.success(f"✅ Loaded {result['num_docs']} chunks in {result['elapsed']}s. Saved to disk cache.")
                    except Exception as e:
                        st.error(f"❌ {e}")


# ─────────────────────────────────────────────
#  STEP 3 — ASK
# ─────────────────────────────────────────────
mode_label = "Full Read" if is_full else f"MMR top-{TOP_K}"

st.markdown(
    f'<div class="card">'
    f'<div class="card-label ask-label">💬 Step 3 — Ask Anything '
    f'<span style="color:var(--muted2);font-size:0.6rem;text-transform:none;letter-spacing:0.05em;">'
    f'[ {mode_label} ]</span>'
    f'</div>',
    unsafe_allow_html=True,
)

active_id, active_session = get_active_session()
vectors_available = bool(active_session and active_session.get("vectors") is not None)

question_input = st.text_input(
    label="Question",
    placeholder="Summarise the main points… What does section X say about Y?",
    label_visibility="collapsed",
    disabled=not vectors_available,
)
ask_btn = st.button("🔍 Ask", disabled=not vectors_available)
st.markdown("</div>", unsafe_allow_html=True)

if not vectors_available:
    st.caption("⬆ Index a PDF or URL (Step 2) and/or select a saved source in the sidebar.")

if ask_btn and active_session:
    if not question_input.strip():
        st.warning("⚠️ Please type a question.")
    else:
        spinner_msg = (
            "📖 Smart-sampling + parallel MapReduce across document…"
            if is_full else
            "🎯 MMR retrieval & synthesis…"
        )
        with st.spinner(spinner_msg):
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=model_choice,
                    temperature=0.0,
                )
                if is_full:
                    result = ask_full_mapreduce(
                        question_input.strip(),
                        active_session["all_docs"],
                        llm,
                        vectors=active_session.get("vectors"),
                    )
                else:
                    result = ask_targeted(
                        question_input.strip(), active_session["vectors"], llm
                    )

                active_session["last_answer"]    = result["answer"]
                active_session["last_context"]   = result.get("context", [])
                st.session_state.last_q_elapsed  = result["elapsed"]
                st.session_state.last_mode       = "full" if is_full else "targeted"
                st.session_state.last_map_count  = result.get("map_count", 0)
                st.session_state.tts_audio_path  = None
                st.session_state.tts_source_text = ""
            except Exception as e:
                st.error(f"❌ {e}")


# ─────────────────────────────────────────────
#  ANSWER DISPLAY + LISTEN BUTTON
# ─────────────────────────────────────────────
active_id, active_session = get_active_session()

if active_session and active_session.get("last_answer"):
    last_answer  = active_session["last_answer"]
    last_context = active_session.get("last_context", [])

    src_is_url  = (active_session.get("source_type") == "Website URL")
    dot_class   = "web" if src_is_url else "doc"
    src_text    = active_session.get("label", "—")
    cache_badge = '<span class="cache-badge">⚡ cached</span>' if active_session.get("from_cache") else ""

    accent_color = "var(--web-accent)" if src_is_url else "var(--doc-accent)"

    st.markdown(
        f'<div class="source-bar">'
        f'<span class="source-dot {dot_class}"></span>'
        f'<span style="color:var(--muted);">Source:</span> '
        f'<span style="color:{accent_color}">'
        f'{src_text[:80]}{"…" if len(src_text) > 80 else ""}</span>'
        f'{cache_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card answer-card">'
        f'<div class="card-label ans-label">✦ Answer</div>'
        f'<div class="answer-content">{last_answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.form("tts_form"):
        tts_col1, tts_col2 = st.columns([1, 3])
        with tts_col1:
            speak_btn = st.form_submit_button("🔊 Listen")
        with tts_col2:
            st.caption("")

    if speak_btn:
        raw_lines     = last_answer.split("\n")
        cleaned_lines = []
        for line in raw_lines:
            ln = clean_for_tts(line).strip()
            if not ln:
                continue
            if ln[-1] not in [".", "?", "!"]:
                ln += "."
            cleaned_lines.append(ln)
        tts_text = " ".join(cleaned_lines)

        if tts_text:
            if (not st.session_state.tts_audio_path) or (st.session_state.tts_source_text != tts_text):
                try:
                    tts = gTTS(tts_text, lang="en")
                    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
                    tts.save(audio_path)
                    st.session_state.tts_audio_path  = audio_path
                    st.session_state.tts_source_text = tts_text
                except Exception as e:
                    st.error(f"❌ TTS error: {e}")

            if st.session_state.tts_audio_path:
                with open(st.session_state.tts_audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")

    if last_context:
        with st.expander(f"📚 View {len(last_context)} Retrieved Passages", expanded=False):
            for i, doc in enumerate(last_context, 1):
                meta   = doc.metadata
                source = Path(meta.get("source", "—")).name
                page   = meta.get("page", meta.get("title", "?"))
                st.markdown(
                    f'<div class="doc-chunk">'
                    f'<div class="doc-chunk-num">Passage {i} · {source} · {page}</div>'
                    f'{doc.page_content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
theme_indicator = "🌙 Dark" if st.session_state.dark_mode else "☀️ Light"
st.markdown(
    f'<p style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
    f'color:var(--muted);text-align:center;letter-spacing:0.1em;">'
    f'· InfoWave AI · Powered by Groq · {theme_indicator} Mode ·'
    f'</p>',
    unsafe_allow_html=True,
)
