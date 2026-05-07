import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pydicom as dicom
import io
import datetime

def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn); 
    w=np.round(w)
    return w


# ─────────────────────────────────────────────
#  CH2 CSS  (ίδιο με ch1)
# ─────────────────────────────────────────────
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
    
    .ch3-section-title {
    font-size: 1.05rem;          /* ← μεγάλο και ορατό */
    font-weight: 700;
    color: #60a5fa;              /* ← φωτεινό μπλε */
    border-left: 3px solid #3b82f6;  /* ← μπλε γραμμή αριστερά */
    padding-left: .7rem;
    text-transform: uppercase;
    letter-spacing: 2px;}
    
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


def _divider(label=""):
    st.markdown(
        f'<div class="ch1-divider"><span>{label}</span></div>',
        unsafe_allow_html=True
    )


def _show_images_only(im_orig, im_proc, proc_label):
    c_orig, c_proc = st.columns(2, gap="medium")
    with c_orig:
        st.markdown('<div class="ch1-img-label">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_orig, 256).astype(np.uint8), use_container_width=True)
    with c_proc:
        st.markdown(f'<div class="ch1-img-label">PROCESSED · {proc_label.upper()}</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_proc, 256).astype(np.uint8), use_container_width=True)


def _build_report(im_orig, im_proc, method, params):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    for ax, arr, title in [
        (axes[0], im_orig, "ORIGINAL"),
        (axes[1], im_proc, f"PROCESSED · {method.upper()}"),
    ]:
        ax.imshow(imNormalize(arr, 256).astype(np.uint8), cmap='gray', aspect='auto')
        ax.set_title(title, color='#60a5fa', fontsize=9, pad=5)
        ax.axis('off')

    axes[2].set_facecolor('#0a0f1e'); axes[2].axis('off')
    so = _image_stats(im_orig); sp = _image_stats(im_proc)
    txt = "\n".join([
        "MedVision · Processing Report",
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
        "PROCESSED IMAGE",
        f"  Size  : {sp['Size']}",
        f"  Range : [{sp['Min']}, {sp['Max']}]",
        f"  Mean  : {sp['Mean']}   Std: {sp['Std']}",
    ])
    axes[2].text(0.05, 0.97, txt, transform=axes[2].transAxes,
                 color='#94a3b8', fontsize=7.8, va='top', linespacing=1.65,
                 fontfamily='monospace')

    fig.suptitle(f"Report  ·  {now}", color='#334155', fontsize=8, y=0.01)
    plt.tight_layout(pad=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#070d1a')
    buf.seek(0); plt.close(fig)
    return buf

def _section(label): st.markdown(f'<div class="ch3-section-title">{label}</div>', unsafe_allow_html=True)

def _download_section(im_orig, im_proc, method, params):
    _divider("Export")
    d1, d2, d3 = st.columns(3, gap="small")

    proc_u8 = imNormalize(im_proc, 256).astype(np.uint8)
    orig_u8 = imNormalize(im_orig, 256).astype(np.uint8)

    with d1:
        buf = io.BytesIO()
        Image.fromarray(proc_u8).save(buf, format="PNG"); buf.seek(0)
        st.download_button(
            "⬇ Processed Image", buf,
            file_name=f"{method.replace(' ','_')}_processed.png",
            mime="image/png", use_container_width=True,
            key=f"dl_proc_{method}"
        )
    with d2:
        buf = io.BytesIO()
        Image.fromarray(orig_u8).save(buf, format="PNG"); buf.seek(0)
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

def RGB2GRAY(im):
    Rim = im[:, :, 0]; Gim = im[:, :, 1]; Bim = im[:, :, 2]
    gray = 0.2989 * Rim + 0.5870 * Gim + 0.1140 * Bim
    return gray

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
        img = RGB2GRAY(img)
    return img.astype(float)


def sigNorm(x):
    xmax = np.max(x); xmin = np.min(x)
    x = (x - xmin) / (xmax - xmin)
    return x

def imNormalize(w, tones=256):
    mx = np.max(w); mn = np.min(w)
    if mx == mn:
        return np.zeros_like(w, dtype=np.uint8)
    w = (tones - 1) * (w - mn) / (mx - mn)
    w = np.round(w)
    return w.astype(np.uint8)

def conv2(im,mask):
    im1=signal.convolve2d(im,mask,mode='same')
    return im1
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
    

    tones    = 256
    im_depth = 256

    _show_badges(im, uploaded.name)

    tab_Conv, tab_Median = st.tabs([
        "  Convolution filters",
        "  Median filter",
    ])

    # ══════════════════════════════════════════
    with tab_Conv:
        _section("Choose Filter")
        Choice = st.selectbox("", ["Smoothing", "Laplacian", "High-emphasis"], label_visibility="collapsed", key="ch2_choice")
        kernel = np.zeros((3, 3), dtype=float)
    
        if Choice == "Smoothing":
            kernels = {
                "SM-1": np.array([[0,1,0],[1,1,1],[0,1,0]]),
                "SM-2": np.array([[1,1,1],[1,1,1],[1,1,1]]),
                "SM-3": np.array([[1,1,1],[1,2,1],[1,1,1]]),
                "SM-4": np.array([[1,2,1],[2,4,2],[1,2,1]]),
            }
            color = "#60a5fa"
    
        elif Choice == "Laplacian":
            kernels = {
                "LM-1": np.array([[0,1,0],[1,-4,1],[0,1,0]]),
                "LM-2": np.array([[1,1,1],[1,-8,1],[1,1,1]]),
                "LM-3": np.array([[1,2,1],[2,-12,2],[1,2,1]]),
                "LM-4": np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]]),
            }
            color = "#60a5fa"
    
        elif Choice == "High-emphasis":
            kernels = {
                "HEM-1": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
                "HEM-2": np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]),
                "HEM-3": np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]]),
                "HEM-4": np.array([[1,-2,1],[-2,5,-2],[1,-2,1]]),
            }
            color = "#60a5fa"
    
        _section(f"{Choice} Masks")
    
        # ── Εμφάνιση kernels ως matplotlib figures ──
        cols = st.columns(len(kernels))
        for i, (name, k) in enumerate(kernels.items()):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(2.0, 2.0), facecolor='#0d1117')
                ax.set_facecolor('#0d1117')
        
                for row in range(3):
                    for col in range(3):
                        val = k[row, col]
                        if val > 0:   c = '#60a5fa'
                        elif val < 0: c = '#f87171'
                        else:         c = '#334155'
        
                        ax.add_patch(plt.Rectangle(
                            (col - 0.5, row - 0.5), 1, 1,
                            facecolor='#0f172a', edgecolor='#1e3a5f', linewidth=1
                        ))
                        ax.text(col, row, f"{val:.3g}",
                                ha='center', va='center',
                                fontsize=11, fontweight='700',
                                color=c, fontfamily='monospace')
        
                ax.set_xlim(-0.5, 2.5)
                ax.set_ylim(-0.5, 2.5)
                ax.invert_yaxis()
                ax.set_title(name, color='#94a3b8', fontsize=9,
                             fontfamily='monospace', fontweight='700', pad=6)
                ax.axis('off')
                plt.tight_layout(pad=0.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
                selected    = st.session_state.get("kernel_name", "")
                is_selected = selected == name
                btn_label   = f"✓ {name}" if is_selected else f"Select {name}"
                if st.button(btn_label, key=f"{Choice.lower()}_{name}",
                             use_container_width=True):
                    st.session_state.kernel      = k
                    st.session_state.kernel_name = name
                    st.rerun()
    
    
        # Εφαρμογή kernel
        if "kernel" not in st.session_state:
            st.info("☝ Select a kernel above to apply it.")
            st.stop()
    
        kernel      = st.session_state.kernel
        kernel_name = st.session_state.get("kernel_name", Choice)
        sK = np.sum(kernel)
        if sK > 0:
            kernel = kernel / sK
        im1 = conv2(im, kernel)
    
        _section("Apply Window to Processed Image")
        apply_window = st.checkbox("Enable Windowing", key="ch2_conv_win")
        if apply_window:
            wc = st.slider("Window Center", 0, 255, 128, key="ch2_conv_wc")
            ww = st.slider("Window Width",  1, 255, 100, key="ch2_conv_ww")
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">WC = {wc}</span>'
                f'<span class="ch1-wp">WW = {ww}</span>'
                f'</div>', unsafe_allow_html=True)
            im_show     = simpleWindow(im1, wc, ww, im_depth, tones)
            params_text = f"kernel={kernel_name}, WC={wc}, WW={ww}"
        else:
            im_show     = im1
            params_text = f"kernel={kernel_name}"
    
        if Choice in ["Laplacian", "High-emphasis"]:
            M = np.size(im, 0); N = np.size(im, 1)
            for i in range(M):
                for j in range(N):
                    if   im_show[i][j] > im_depth: im_show[i][j] = im_depth
                    elif im_show[i][j] < 0:        im_show[i][j] = 0
    
        method_name = f"{Choice} · {kernel_name}"
    
        _divider("Output")
        _show_images_only(im, im_show, method_name)
        _download_section(im, im_show, method_name, params_text)


    # ══════════════════════════════════════════
    with tab_Median:
        im1 = signal.medfilt2d(im, (3, 3))
        _section("Apply Window to Processed Image")
        apply_window = st.checkbox("Enable Windowing", key="ch2_med_win")
        if apply_window:
            wc = st.slider("Window Center", 0, 255, 128, key="ch2_med_wc")
            ww = st.slider("Window Width",  1, 255, 100, key="ch2_med_ww")
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">WC = {wc}</span>'
                f'<span class="ch1-wp">WW = {ww}</span>'
                f'</div>', unsafe_allow_html=True)
            im_show     = simpleWindow(im1, wc, ww, im_depth, tones)
            params_text = f"Median 3×3, WC={wc}, WW={ww}"
        else:
            im_show     = im1
            params_text = "Median 3×3"

        method_name = "Median Filter"

        # ── OUTPUT ──
        _divider("Output")
        _show_images_only(im, im_show, method_name)
        _download_section(im, im_show, method_name, params_text)