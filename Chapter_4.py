import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
import pydicom as dicom
import streamlit as st
from PIL import Image
import io


def simpleWindow(im, wc, ww, image_depth, tones):
    im1 = np.zeros(im.shape, dtype=float)
    Vb = (2.0 * wc + ww) / 2.0
    if Vb > image_depth:
        Vb = image_depth
    Va = Vb - ww
    if Va < 0:
        Va = 0
    M = np.size(im, 0)
    N = np.size(im, 1)
    for i in range(M):
        for j in range(N):
            Vm = im[i][j]
            if Vm < Va:
                t = 0
            elif Vm > Vb:
                t = tones - 1
            else:
                t = (((tones - 1) * (Vm - Va) / (Vb - Va)))
            im1[i][j] = np.round(t)
    return im1


def prepare_image(x):
    x = x.astype(float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def sigNorm(x):
    xmax = np.max(x)
    xmin = np.min(x)
    return (x - xmin) / (xmax - xmin)


def loadImage(uploaded):
    uploaded.seek(0)
    if uploaded.name.endswith(".dcm"):
        file_bytes = uploaded.read()
        ds = dicom.dcmread(io.BytesIO(file_bytes))
        img = ds.pixel_array
    else:
        uploaded.seek(0)
        img = Image.open(uploaded)
        img = np.array(img)
    if len(img.shape) == 3:
        img = rgb2gray(img)
    return img.astype(float)


def run():

    st.markdown("""
    <style>
    .ch4-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #e2e8f0 0%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 2rem 0;
    }
    .ch4-divider {
        height: 1px;
        background: rgba(99,179,237,0.08);
        margin: 1.6rem 0;
    }
    .ch4-control-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .ch4-img-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .ch4-sino-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.8rem;
    }
    .ch4-warn {
        background: rgba(234,179,8,0.07);
        border: 1px solid rgba(234,179,8,0.2);
        border-radius: 10px;
        padding: 1rem 1.4rem;
        color: #ca8a04;
        font-size: 13px;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(20,184,166,0.12), rgba(6,182,212,0.12)) !important;
        border: 1px solid rgba(20,184,166,0.35) !important;
        color: #2dd4bf !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.6rem !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(20,184,166,0.22) !important;
        border-color: rgba(20,184,166,0.6) !important;
    }
    div[data-testid="stTabs"] button {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #64748b !important;
        padding: 0.6rem 1.6rem !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #2dd4bf !important;
        border-bottom-color: #2dd4bf !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # ── check upload ─────────────────────────────────────────────────────
    uploaded = st.session_state.get("uploaded_file", None)
    if uploaded is None:
        st.markdown("""
        <div class="ch4-warn">
            &#9888;&nbsp; No image loaded. Please upload an image from the Dashboard first.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    im = loadImage(uploaded)
    im = np.asanyarray(im, dtype=float)
    im = sigNorm(im)
    if len(im.shape) == 3:
        im = rgb2gray(im)

    tones = 256
    image_depth = 256

    # ── projections ──────────────────────────────────────────────────────
    st.markdown('<div class="ch4-control-label">Projections</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        N_proj = st.selectbox("", [180, 360, "Custom"], label_visibility="collapsed")
    with col_p2:
        if N_proj == "Custom":
            N_proj = st.slider("", 10, 720, 180, label_visibility="collapsed")

    theta = np.arange(0, N_proj)
    sinogram = radon(im, theta=theta, circle=False)
    st.markdown("""
    <style>
    div[data-testid="stTabs"] button {
        font-size: 22px !important;
        padding: 14px 28px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # ── method tabs ──────────────────────────────────────────────────────
    tab_fbp, tab_art = st.tabs(["FBP — Filtered Back Projection", "ART — Algebraic Reconstruction"])

    with tab_fbp:
        st.markdown('<div class="ch4-control-label" style="margin-top:1rem;">Filter</div>', unsafe_allow_html=True)
        filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'None']
        Choice = st.selectbox("", filters, label_visibility="collapsed")
        if Choice == 'None':
            I = iradon(sinogram, theta=theta, filter_name=None)
        else:
            I = iradon(sinogram, theta=theta, filter_name=Choice, output_size=im.shape[0])
        title = f"FBP — {Choice} filter"

    with tab_art:
        st.markdown('<div class="ch4-control-label" style="margin-top:1rem;">Iterations</div>', unsafe_allow_html=True)
        iterations = st.slider("", 1, 100, 1, label_visibility="collapsed")
        I = iradon_sart(sinogram, theta=theta)
        for _ in range(iterations):
            I = iradon_sart(sinogram, theta=theta, image=I)
        title = f"ART — {iterations} iteration{'s' if iterations > 1 else ''}"

    st.markdown('<div class="ch4-divider"></div>', unsafe_allow_html=True)

    # ── windowing ────────────────────────────────────────────────────────
    apply_window = st.checkbox("Apply windowing to reconstructed image")
    if apply_window:
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown('<div class="ch4-control-label">Window Center</div>', unsafe_allow_html=True)
            wc = st.slider("", 0, 255, 128, key="wc", label_visibility="collapsed")
        with col_w2:
            st.markdown('<div class="ch4-control-label">Window Width</div>', unsafe_allow_html=True)
            ww = st.slider("", 1, 255, 100, key="ww", label_visibility="collapsed")
        im_show = simpleWindow(I, wc, ww, image_depth, tones)
    else:
        im_show = I

    st.markdown('<div class="ch4-divider"></div>', unsafe_allow_html=True)

    # ── images side by side ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="ch4-img-label">Original Image</div>', unsafe_allow_html=True)
        st.image(prepare_image(im), use_container_width=True)

    with col2:
        st.markdown(f'<div class="ch4-img-label">{title}</div>', unsafe_allow_html=True)
        st.image(prepare_image(im_show), use_container_width=True)

    st.markdown('<div class="ch4-divider"></div>', unsafe_allow_html=True)

    # ── sinogram ─────────────────────────────────────────────────────────
    st.markdown('<div class="ch4-sino-label">Sinogram</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('#070d1a')
    ax.set_facecolor('#070d1a')
    ax.imshow(sinogram, cmap='gray', aspect='auto',
              extent=(0, N_proj, 0, sinogram.shape[0]))
    ax.set_xlabel("Angle of projection", color='#475569', fontsize=9)
    ax.set_ylabel("Tomo-projections", color='#475569', fontsize=9)
    ax.tick_params(colors='#334155', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="ch4-divider"></div>', unsafe_allow_html=True)

    # ── download ─────────────────────────────────────────────────────────
    Inorm = sigNorm(I)
    im1_uint8 = (Inorm * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(im1_uint8).save(buf, format="PNG")
    buf.seek(0)

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="Download Reconstructed Image",
            data=buf,
            file_name="reconstruction.png",
            mime="image/png"
        )
