import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
import pydicom as dicom
import streamlit as st
from PIL import Image
import io
import datetime


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
def imNormalize(w, tones=256):
    mx = np.max(w); mn = np.min(w)
    if mx == mn:
        return np.zeros_like(w, dtype=np.uint8)
    w = (tones - 1) * (w - mn) / (mx - mn)
    w = np.round(w)
    return w.astype(np.uint8)



def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    .ch1-badge-row {
        display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 1.1rem;
    }
    .ch1-badge {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 20px; padding: .22rem .75rem;
        font-size: .72rem; color: #94a3b8;
        font-family: 'Space Mono', monospace;
    }
    .ch1-badge b { color: #60a5fa; }

    .ch1-win-params { margin: .3rem 0 .7rem 0; }
    .ch1-wp {
        display: inline-block;
        background: rgba(0,255,179,0.07);
        border: 1px solid rgba(0,255,179,0.2);
        border-radius: 5px; padding: .18rem .6rem;
        font-family: 'Space Mono', monospace;
        font-size: .7rem; color: #00ffb3;
        margin-right: .35rem; margin-bottom: .3rem;
    }

    .ch1-divider {
        display: flex; align-items: center; gap: .7rem; margin: 1.1rem 0;
    }
    .ch1-divider span {
        font-size: .62rem; color: #334155; text-transform: uppercase;
        letter-spacing: 1.2px; white-space: nowrap;
        font-family: 'Space Mono', monospace;
    }
    .ch1-divider::before, .ch1-divider::after {
        content: ''; flex: 1; height: 1px;
        background: rgba(99,179,237,0.1);
    }

    .ch1-img-label {
        font-family: 'Space Mono', monospace; font-size: .62rem;
        text-transform: uppercase; letter-spacing: 1.1px;
        color: #475569; text-align: center;
        padding: .3rem 0 .2rem; margin-bottom: .25rem;
    }

    /* ── SECTION HEADERS ── */
    .ch4-section-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.2rem 0 .6rem 0;
        border-left: 3px solid #3b82f6;
        padding-left: .7rem;
    }

    /* ── TABS — κόκκινο χρώμα ── */
    div[data-testid="stTabs"] button {
        font-family: 'Space Mono', monospace !important;
        font-size: .78rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        letter-spacing: 1px !important;
        padding: .6rem 1.4rem !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #f87171 !important;
        border-bottom-color: #ef4444 !important;
    }
    div[data-testid="stTabs"] button:hover {
        color: #fca5a5 !important;
    }

    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
        color: #fff !important; border: none !important;
        border-radius: 7px !important;
        font-size: .72rem !important;
        font-family: 'Space Mono', monospace !important;
        letter-spacing: .6px !important;
        padding: .5rem 1rem !important;
        width: 100% !important;
        transition: opacity .2s !important;
    }
    [data-testid="stDownloadButton"] button:hover { opacity: .82 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _image_stats(im):
    return {
        "Min":  int(np.min(im)),
        "Max":  int(np.max(im)),
        "Mean": f"{np.mean(im):.1f}",
        "Std":  f"{np.std(im):.1f}",
        "Size": f"{im.shape[0]}×{im.shape[1]}",
    }


def _show_badges(im, filename):
    s   = _image_stats(im)
    now = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M")
    st.markdown(f"""
    <div class="ch1-badge-row">
      <div class="ch1-badge">📁 <b>{filename}</b></div>
      <div class="ch1-badge">📐 <b>{s['Size']}</b> px</div>
      <div class="ch1-badge">◐ range <b>{s['Min']}–{s['Max']}</b></div>
      <div class="ch1-badge">μ mean <b>{s['Mean']}</b></div>
      <div class="ch1-badge">σ std <b>{s['Std']}</b></div>
      <div class="ch1-badge">🕐 <b>{now}</b></div>
    </div>
    """, unsafe_allow_html=True)


def _section(label):
    st.markdown(
        f'<div class="ch4-section-title">{label}</div>',
        unsafe_allow_html=True
    )


def _divider(label=""):
    st.markdown(
        f'<div class="ch1-divider"><span>{label}</span></div>',
        unsafe_allow_html=True
    )


def _crop_circle(arr):
    """Κόβει τον κύκλο του CT reconstruction — μαυρίζει ό,τι είναι έξω."""
    arr = arr.astype(float)
    H, W = arr.shape
    cy, cx = H / 2, W / 2
    r = min(H, W) / 2 * 0.98          # ελαφρώς μέσα στα όρια
    Y, X = np.ogrid[:H, :W]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    out  = arr.copy()
    out[mask] = 0
    # crop στο τετράγωνο εγγεγραμμένο στον κύκλο
    margin = int((min(H, W) - min(H, W) * 0.98 * np.sqrt(2) / 2) / 2) + 2
    out = out[margin: H - margin, margin: W - margin]
    return out


def _to_uint8(arr):
    mn, mx = np.min(arr), np.max(arr)
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _build_report(im_orig, im_proc, method, params):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    for ax, arr, title in [
        (axes[0], im_orig, "ORIGINAL"),
        (axes[1], im_proc, f"RECONSTRUCTED · {method.upper()}"),
    ]:
        ax.imshow(_to_uint8(arr), cmap='gray', aspect='auto')
        ax.set_title(title, color='#60a5fa', fontsize=9, pad=5)
        ax.axis('off')

    axes[2].set_facecolor('#0a0f1e'); axes[2].axis('off')
    so = _image_stats(im_orig); sp = _image_stats(im_proc)
    txt = "\n".join([
        "MedVision · CT Report",
        "─" * 32,
        f"Date / Time : {now}",
        f"Method      : {method}",
        f"Parameters  : {params}",
        "",
        "ORIGINAL IMAGE",
        f"  Size  : {so['Size']}",
        f"  Range : [{so['Min']}, {so['Max']}]",
        f"  Mean  : {so['Mean']}   Std: {so['Std']}",
        "",
        "RECONSTRUCTED IMAGE",
        f"  Size  : {sp['Size']}",
        f"  Range : [{sp['Min']}, {sp['Max']}]",
        f"  Mean  : {sp['Mean']}   Std: {sp['Std']}",
    ])
    axes[2].text(0.05, 0.97, txt, transform=axes[2].transAxes,
                 color='#94a3b8', fontsize=7.8, va='top', linespacing=1.65,
                 fontfamily='monospace')
    fig.suptitle(f"CT Report  ·  {now}", color='#334155', fontsize=8, y=0.01)
    plt.tight_layout(pad=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#070d1a')
    buf.seek(0); plt.close(fig)
    return buf


def _download_section(im_orig, im_proc, method, params):
    _divider("Export")
    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        buf = io.BytesIO()
        Image.fromarray(_to_uint8(im_proc)).save(buf, format="PNG"); buf.seek(0)
        st.download_button(
            "⬇ Reconstructed Image", buf,
            file_name=f"{method.replace(' ','_')}_reconstructed.png",
            mime="image/png", use_container_width=True,
            key=f"dl_proc_{method}"
        )
    with d2:
        buf = io.BytesIO()
        Image.fromarray(_to_uint8(im_orig)).save(buf, format="PNG"); buf.seek(0)
        st.download_button(
            "⬇ Original Image", buf,
            file_name="original.png",
            mime="image/png", use_container_width=True,
            key=f"dl_orig_{method}"
        )
    with d3:
        report_buf = _build_report(im_orig, im_proc, method, params)
        st.download_button(
            "⬇ Full Report (PNG)", report_buf,
            file_name=f"Report_{method.replace(' ','_')}.png",
            mime="image/png", use_container_width=True,
            key=f"dl_rep_{method}"
        )


def _show_sinogram(sinogram, N_proj):
    fig, ax = plt.subplots(figsize=(9, 2.5), facecolor='#070d1a')
    ax.set_facecolor('#070d1a')
    ax.imshow(sinogram, cmap='inferno', aspect='auto',
              extent=(0, N_proj, 0, sinogram.shape[0]))
    ax.set_xlabel("Angle of projection", color='#475569', fontsize=8)
    ax.set_ylabel("Projection",          color='#475569', fontsize=8)
    ax.tick_params(colors='#334155', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────
#  ORIGINAL FUNCTIONS — αναλλοίωτα
# ─────────────────────────────────────────────
def simpleWindow(im, wc, ww, image_depth, tones):
    im1 = np.zeros(im.shape, dtype=float)
    Vb = (2.0 * wc + ww) / 2.0
    if Vb > image_depth: Vb = image_depth
    Va = Vb - ww
    if Va < 0: Va = 0
    M = np.size(im, 0); N = np.size(im, 1)
    for i in range(M):
        for j in range(N):
            Vm = im[i][j]
            if Vm < Va:      t = 0
            elif Vm > Vb:    t = tones - 1
            else:            t = (((tones - 1) * (Vm - Va) / (Vb - Va)))
            im1[i][j] = np.round(t)
    return im1

def prepare_image(x):
    x = x.astype(float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def sigNorm(x):
    xmax = np.max(x); xmin = np.min(x)
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

def _show_images_ct(im_orig, im_cropped, label):

    def safe_norm(x):
        x = np.nan_to_num(x, nan=0.0)
        mn, mx = np.min(x), np.max(x)
        if mx == mn:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    orig_disp = safe_norm(im_orig)
    crop_disp = safe_norm(im_cropped)

    H_orig, W_orig = orig_disp.shape
    H_crop, W_crop = crop_disp.shape

    # Canvas με μέγεθος της ΜΕΓΑΛΥΤΕΡΗΣ εικόνας
    H_canvas = max(H_orig, H_crop)
    W_canvas = max(W_orig, W_crop)

    # Κεντράρισε original
    canvas_orig = np.zeros((H_canvas, W_canvas), dtype=float)
    y1 = (H_canvas - H_orig) // 2
    x1 = (W_canvas - W_orig) // 2
    canvas_orig[y1:y1 + H_orig, x1:x1 + W_orig] = orig_disp

    # Κεντράρισε cropped
    canvas_crop = np.zeros((H_canvas, W_canvas), dtype=float)
    y2 = (H_canvas - H_crop) // 2
    x2 = (W_canvas - W_crop) // 2
    canvas_crop[y2:y2 + H_crop, x2:x2 + W_crop] = crop_disp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')

    ax1.imshow(canvas_orig, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_title('ORIGINAL', color='#475569', fontsize=8,
                  fontfamily='monospace', pad=8)
    ax1.axis('off')

    ax2.imshow(canvas_crop, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax2.set_title(f'RECONSTRUCTED · {label.upper()}', color='#475569',
                  fontsize=8, fontfamily='monospace', pad=8)
    ax2.axis('off')

    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
def _show_images_ct_art(im_orig, im_cropped, label):
    
    from PIL import Image as PILImage

    def safe_norm(x):
        x = np.nan_to_num(x, nan=0.0)
        mn, mx = np.min(x), np.max(x)
        if mx == mn:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    orig_disp = safe_norm(im_orig)
    crop_disp = safe_norm(im_cropped)

    H_orig, W_orig = orig_disp.shape

    # Resize την cropped στο μέγεθος της original
    crop_pil    = PILImage.fromarray((crop_disp * 255).astype(np.uint8))
    crop_pil    = crop_pil.resize((W_orig, H_orig), PILImage.LANCZOS)
    crop_resized = np.array(crop_pil) / 255.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')

    ax1.imshow(orig_disp,    cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_title('ORIGINAL', color='#475569', fontsize=8,
                  fontfamily='monospace', pad=8)
    ax1.axis('off')

    ax2.imshow(crop_resized, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax2.set_title(f'RECONSTRUCTED · {label.upper()}', color='#475569',
                  fontsize=8, fontfamily='monospace', pad=8)
    ax2.axis('off')

    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def _crop_circle(arr):
    arr = arr.astype(float)
    H, W = arr.shape
    cy, cx = H / 2, W / 2
    r = min(H, W) / 2 * 0.98
    Y, X = np.ogrid[:H, :W]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    out = arr.copy()
    out[mask] = 0              # ← 0 αντί για nan
    side = int(r * 1.6)
    cy_i, cx_i = int(cy), int(cx)
    half = side // 2
    out = out[cy_i - half: cy_i + half, cx_i - half: cx_i + half]
    return out

# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
def run():
    _inject_css()

    uploaded = st.session_state.get("uploaded_file", None)
    if uploaded is None:
        st.warning("Upload an image from the Dashboard.")
        st.stop()

    im = loadImage(uploaded)
    im = np.asarray(im, dtype=float)
    im = sigNorm(im)
    if len(im.shape) == 3:
        im = rgb2gray(im)

    tones       = 256
    image_depth = 256

    _show_badges(im, uploaded.name)

    # ── Projections ──────────────────────────────────────────────────────
    _section("Projections")
    c1, c2 = st.columns([1, 2])
    with c1:
        N_proj = st.selectbox("Number of projections", [180, 360, "Custom"], key="ch4_nproj")
    with c2:
        if N_proj == "Custom":
            N_proj = st.slider("Custom projections", 10, 720, 180, key="ch4_custom")

    theta    = np.arange(0, N_proj)
    sinogram = radon(im, theta=theta, circle=False)

    # ── Method tabs ──────────────────────────────────────────────────────
    tab_fbp, tab_art = st.tabs([
        "  FBP — Filtered Back Projection",
        "  ART — Algebraic Reconstruction",
    ])

    # ══════════════════════════════════════════
    with tab_fbp:
        _section("Filter")
        filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'None']
        Choice  = st.selectbox("Filter", filters, key="ch4_fbp_filter")

        if Choice == 'None':
            I = iradon(sinogram, theta=theta, filter_name=None)
        else:
            I = iradon(sinogram, theta=theta, filter_name=Choice, output_size=im.shape[0])

        method_name = f"FBP · {Choice} filter"
        params_text = f"filter={Choice}, projections={N_proj}"

        # windowing
        _section("Apply Window to Reconstructed Image")
        apply_window = st.checkbox("Enable windowing", key="ch4_fbp_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="ch4_fbp_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="ch4_fbp_ww")
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">WC = {wc}</span>'
                f'<span class="ch1-wp">WW = {ww}</span>'
                f'</div>', unsafe_allow_html=True)
            im_show      = simpleWindow(I, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = I

        # crop κύκλος
        # crop κύκλος
        im_cropped = _crop_circle(im_show)

        # ── Images ──
        _section("Images")
        _show_images_ct(im, im_cropped, Choice)   # FBP tab
        # ή για ART:
        # _show_images_ct(im, im_cropped, "ART")

        # ── Sinogram ──
        _section("Sinogram")
        _show_sinogram(sinogram, N_proj)

        _download_section(im, im_cropped, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_art:
        _section("Iterations")
        iterations = st.slider("Number of iterations", 1, 100, 1, key="ch4_art_iter")

        I = iradon_sart(sinogram, theta=theta)
        for _ in range(iterations):
            I = iradon_sart(sinogram, theta=theta, image=I)

        method_name = f"ART · {iterations} iteration{'s' if iterations > 1 else ''}"
        params_text = f"iterations={iterations}, projections={N_proj}"

        # windowing
        _section("Apply Window to Reconstructed Image")
        apply_window = st.checkbox("Enable windowing", key="ch4_art_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="ch4_art_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="ch4_art_ww")
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">WC = {wc}</span>'
                f'<span class="ch1-wp">WW = {ww}</span>'
                f'</div>', unsafe_allow_html=True)
            im_show      = simpleWindow(I, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = I

        # crop κύκλος

        # ── Images ──
        # crop κύκλος
        im_cropped = _crop_circle(im_show)

        # ── Images ──
        _section("Images")
        _show_images_ct_art(im, im_cropped, "ART — Algebraic Reconstruction")   # FBP tab
        # ή για ART:
        # _show_images_ct(im, im_cropped, "ART")

        # ── Sinogram ──
        _section("Sinogram")
        _show_sinogram(sinogram, N_proj)

        _download_section(im, im_cropped, method_name, params_text)