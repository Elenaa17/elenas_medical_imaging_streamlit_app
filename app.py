import streamlit as st
import Chapter_1 as CH1
import Chapter_2 as CH2
import Chapter_3 as CH3
import Chapter_4 as CH4


st.set_page_config(
    page_title="Medical Image Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');


    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }


    #MainMenu { visibility: visible !important; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }


    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #0d1b3e 50%, #0a1628 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }


    .sidebar-logo {
        display: flex; align-items: center; gap: 12px;
        padding: 1.2rem 1rem 0.5rem; margin-bottom: 0.5rem;
    }
    .sidebar-logo-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }
    .sidebar-logo-text { font-size: 17px; font-weight: 600; color: #f0f9ff !important; letter-spacing: -0.3px; }
    .sidebar-logo-sub { font-size: 10px; font-weight: 400; color: #64748b !important; letter-spacing: 1.5px; text-transform: uppercase; }


    .nav-label {
        font-size: 10px; font-weight: 600; letter-spacing: 1.8px;
        text-transform: uppercase; color: #475569 !important;
        padding: 1.2rem 1rem 0.4rem;
    }


    .stButton > button {
        width: 100%; text-align: left !important;
        background: transparent !important; border: none !important;
        border-radius: 8px !important; padding: 0.55rem 1rem !important;
        font-size: 13.5px !important; font-weight: 400 !important;
        color: #94a3b8 !important; transition: all 0.15s ease !important;
        cursor: pointer !important; margin-bottom: 2px !important;
    }
    .stButton > button:hover {
        background: rgba(99, 179, 237, 0.08) !important;
        color: #e2e8f0 !important; transform: translateX(2px) !important;
    }
    .nav-active .stButton > button {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #93c5fd !important;
        border-left: 2px solid #3b82f6 !important;
        font-weight: 500 !important;
    }


    .sidebar-divider { height: 1px; background: rgba(99, 179, 237, 0.1); margin: 1rem; }
    .sidebar-footer { padding: 1rem; font-size: 11px; color: #334155 !important; text-align: center; }


    .main-header {
        display: flex; align-items: center; gap: 16px;
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid rgba(99, 179, 237, 0.1);
        margin-bottom: 2rem;
    }
    .main-header-badge {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(6,182,212,0.15));
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 8px; padding: 4px 10px;
        font-size: 11px; font-weight: 500; color: #60a5fa;
        letter-spacing: 0.5px; text-transform: uppercase;
    }


    .feature-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 1.5rem 0; }
    .feature-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(99, 179, 237, 0.12);
        border-radius: 12px; padding: 1.2rem 1.4rem; transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: rgba(59, 130, 246, 0.35); }
    .feature-card-icon { font-size: 22px; margin-bottom: 8px; }
    .feature-card-title { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
    .feature-card-desc { font-size: 12.5px; color: #64748b; line-height: 1.5; }


    .stats-row { display: flex; gap: 12px; margin: 1.5rem 0; flex-wrap: wrap; }
    .stat-pill {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 179, 237, 0.1);
        border-radius: 20px; padding: 6px 14px;
        font-size: 12px; color: #94a3b8;
        display: flex; align-items: center; gap: 6px;
    }
    .stat-pill span { color: #60a5fa; font-weight: 600; }


    .hero-title {
        font-size: 32px; font-weight: 700;
        background: linear-gradient(135deg, #e2e8f0 0%, #93c5fd 50%, #06b6d4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.2; margin-bottom: 8px;
    }
    .hero-subtitle { font-size: 15px; color: #64748b; font-weight: 400; margin-bottom: 1.5rem; }


    /* Upload zone styling — only cosmetic, no pointer-events override */
    .upload-section {
        background: rgba(15, 23, 42, 0.5);
        border: 1.5px dashed rgba(59, 130, 246, 0.35);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .upload-section-title {
        font-size: 13px; font-weight: 500;
        color: #64748b; margin-bottom: 0.75rem;
        text-transform: uppercase; letter-spacing: 1px;
    }


    .page-footer {
        margin-top: 3rem; padding-top: 1.5rem;
        border-top: 1px solid rgba(99, 179, 237, 0.08);
        display: flex; justify-content: space-between; align-items: center;
        font-size: 12px; color: #334155;
    }


    .stMarkdown p { color: #94a3b8; }
    .stApp { background: #070d1a; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──
if "chapter" not in st.session_state:
    st.session_state.chapter = "Home"


# ── Sidebar — navigation only, NO uploader here ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div>
            <div class="sidebar-logo-sub">Medical Image Analysis Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">🗂 Modules</div>', unsafe_allow_html=True)


    nav_items = [
        ("Home", "Dashboard"),
        ("Image display modification methods", "Display & Enhancement"),
        ("Spatial domain image filtering", "Spatial Filtering"),
        ("Frequency domain image filtering", "Frequency Filtering"),
        ("Tomographic image reconstruction methods", "Tomographic Reconstruction"),
    ]
    
    for chapter_key, label in nav_items:
        is_active = st.session_state.chapter == chapter_key
        st.markdown(f'<div class="{"nav-active" if is_active else ""}">', unsafe_allow_html=True)
        if st.button(f"{label}", key=f"nav_{chapter_key}"):
            st.session_state.chapter = chapter_key
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-footer">
        Medical Image Analysis Platform v2.0<br>© 2026 · All rights reserved
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──
chapter = st.session_state.chapter


if chapter == "Home":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="main-header">
            <div class="main-header-badge">Medical Imaging Platform</div>
        </div>
        <div class="hero-title">Advanced Medical<br>Image Analysis</div>
        <div class="hero-subtitle">
            Interactive tools for exploring medical image processing, including visualization, filtering, and reconstruction techniques.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:flex-end;padding-top:1rem;">
            <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);
            border-radius:8px;padding:6px 14px;font-size:12px;color:#60a5fa;">
                🟢 System ready
            </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <div class="stats-row">
        <div class="stat-pill"><span>4</span> analysis modules</div>
        <div class="stat-pill"><span>DICOM</span> compatible</div>
        <div class="stat-pill"><span>FFT</span> processing</div>
        <div class="stat-pill"><span>CT</span> reconstruction</div>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    
    
    
    
    
    
    
    st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-card-title">Display & Enhancement</div>
        <div class="feature-card-desc">Ιntensity transformations with window/level adjustment, histogram equalization, Cumulative Distribution Function Equalization (CDF) equalization and CLAHE.</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-title">Spatial Filtering</div>
        <div class="feature-card-desc">Applies convolution (smoothing, Laplacian, high-emphasis) and median filtering with low-pass, high-pass, band-pass and band-reject options.</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-title">Frequency Analysis</div>
        <div class="feature-card-desc">2D FFT-based image filtering using Ideal, Butterworth, Gaussian, Exponential, and Wiener filters with low-pass, high-pass, band-pass and band-reject options.</div>
    </div>
    <div class="feature-card">
        <div class="feature-card-title">Tomographic Reconstruction</div>
        <div class="feature-card-desc">CT image reconstruction from projection data using Filtered Back Projection (FBP) and Algebraic Reconstruction Technique (ART). Includes sinogram generation and reconstruction visualization.</div>
    </div>
</div>
""", unsafe_allow_html=True)
    
   


    # ── Upload lives here on the home page ──


    st.markdown('<div class="upload-section-title">📂 Upload Medical Image</div>', unsafe_allow_html=True)


    uploaded = st.file_uploader(
        "Supports BMP, JPG, PNG, DICOM",
        type=["bmp", "jpg", "png", "dcm"],
        label_visibility="visible"
    )
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
        border-radius:8px;padding:10px 14px;margin-top:10px;font-size:13px;color:#4ade80;
        display:flex;align-items:center;gap:8px;">
            ✓ <strong>{uploaded.name}</strong> &nbsp;·&nbsp;
            <span style="color:#64748b;">{round(uploaded.size/1024, 1)} KB — ready for analysis</span>
        </div>
        """, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)


    st.markdown("""
    <div class="page-footer">
        <div>Medical Image Analysis Platform</div>
        <div>© 2026 · All rights reserved</div>
    </div>
    """, unsafe_allow_html=True)


elif chapter == "Image display modification methods":
    st.markdown('<div class="main-header"><div class="main-header-badge">Module 01 · Display & Enhancement</div></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#e2e8f0;font-weight:600;margin-bottom:1.5rem;"> Display & Enhancement</h2>', unsafe_allow_html=True)
    CH1.run()


elif chapter == "Spatial domain image filtering":
    st.markdown('<div class="main-header"><div class="main-header-badge">Module 02 · Spatial Filtering</div></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#e2e8f0;font-weight:600;margin-bottom:1.5rem;"> Spatial Domain Filtering</h2>', unsafe_allow_html=True)
    CH2.run()


elif chapter == "Frequency domain image filtering":
    st.markdown('<div class="main-header"><div class="main-header-badge">Module 03 · Frequency Domain Filtering</div></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#e2e8f0;font-weight:600;margin-bottom:1.5rem;"> Frequency Domain Filtering</h2>', unsafe_allow_html=True)
    CH3.run()


elif chapter == "Tomographic image reconstruction methods":
    st.markdown('<div class="main-header"><div class="main-header-badge">Module 04 · Tomographic Reconstruction</div></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#e2e8f0;font-weight:600;margin-bottom:1.5rem;"> Tomographic image reconstruction methods</h2>', unsafe_allow_html=True)
    CH4.run()
    
    