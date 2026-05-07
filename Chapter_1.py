import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pydicom as dicom
from skimage import exposure
import datetime
from skimage import exposure

# ─────────────────────────────────────────────
#  CH1 CSS
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

    .ch1-help {
        display: flex; align-items: flex-start; gap: .55rem;
        background: rgba(123,97,255,0.07);
        border: 1px solid rgba(123,97,255,0.25);
        border-left: 3px solid #7b61ff;
        border-radius: 8px; padding: .6rem .9rem;
        font-size: .78rem; color: #cbd5e1; line-height: 1.55;
        margin-bottom: .85rem;
    }
    .ch1-help-icon { font-size: 1rem; flex-shrink: 0; margin-top:.05rem; }

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

    /* ── Stats bar ── */
    .ch1-stats-bar {
        display: flex; gap: .6rem; flex-wrap: wrap;
        margin-bottom: 1.1rem;
    }
    .ch1-stat {
        flex: 1; min-width: 100px;
        background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(13,27,46,0.9) 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px; padding: .7rem .9rem;
        position: relative; overflow: hidden;
        transition: border-color .2s;
    }
    .ch1-stat::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
    }
    .ch1-stat-lbl {
        font-size: .6rem; text-transform: uppercase;
        letter-spacing: 1.1px; color: #475569;
        font-family: 'Space Mono', monospace; margin-bottom: .25rem;
    }
    .ch1-stat-orig {
        font-family: 'Space Mono', monospace;
        font-size: .95rem; font-weight: 700; color: #60a5fa;
        line-height: 1.1;
    }
    .ch1-stat-proc {
        font-size: .65rem; color: #334155; margin-top: .18rem;
        font-family: 'Space Mono', monospace;
    }
    .ch1-stat-proc b { color: #06b6d4; }

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

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _section(label): 
    st.markdown(f'<div class="ch3-section-title">{label}</div>', unsafe_allow_html=True)


def _image_stats(im):
    return {
        "Min":   int(np.min(im)),
        "Max":   int(np.max(im)),
        "Mean":  f"{np.mean(im):.1f}",
        "Std":   f"{np.std(im):.1f}",
        "Size":  f"{im.shape[0]}×{im.shape[1]}",
    }

def _dark_plot(y_data, title, xlabel="Input level", ylabel="Output level"):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    ax.set_facecolor('#070d1a')
    ax.plot(y_data, color='#60a5fa', linewidth=1.8)
    ax.set_title(title, color='#94a3b8', fontsize=8.5, pad=5)
    ax.set_xlabel(xlabel, color='#475569', fontsize=7.5)
    ax.set_ylabel(ylabel, color='#475569', fontsize=7.5)
    ax.tick_params(colors='#475569', labelsize=6.5)
    ax.grid(True, color='#0d1b2e', linewidth=0.8, linestyle='--')
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    plt.tight_layout(pad=0.7)
    return fig

def _dark_histogram(h, title, color='#7b61ff'):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    ax.set_facecolor('#070d1a')
    ax.bar(range(len(h)), h, color=color, alpha=0.85, width=1.0)
    ax.set_title(title, color='#94a3b8', fontsize=8.5, pad=5)
    ax.set_xlabel("Gray level", color='#475569', fontsize=7.5)
    ax.set_ylabel("Frequency",  color='#475569', fontsize=7.5)
    ax.tick_params(colors='#475569', labelsize=6.5)
    ax.grid(True, color='#0d1b2e', linewidth=0.6, axis='y', linestyle='--')
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    plt.tight_layout(pad=0.7)
    return fig


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

def _show_images_and_curve(im_orig, im_proc, proc_label, fig_curve):
    """Images (left 2/3) + mapping curve (right 1/3) on the same row."""
    c_orig, c_proc, c_curve = st.columns([5, 5, 4], gap="medium")
    with c_orig:
        st.markdown('<div class="ch1-img-label">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_orig), use_container_width=True)
    with c_proc:
        st.markdown(f'<div class="ch1-img-label">PROCESSED · {proc_label.upper()}</div>',
                    unsafe_allow_html=True)
        st.image(imNormalize(im_proc), use_container_width=True)
    with c_curve:
        st.markdown('<div class="ch1-img-label">MAPPING CURVE</div>', unsafe_allow_html=True)
        if fig_curve is not None:
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)
            
def _show_images_and_curve_just_for_hystogram_eq(im_orig, im_proc, proc_label, fig_curve):
    """Images (left 2/3) + mapping curve (right 1/3) on the same row."""
    c_orig, c_proc, c_curve = st.columns([5, 5, 4], gap="medium")
    with c_orig:
        st.markdown('<div class="ch1-img-label">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_orig), use_container_width=True)
    with c_proc:
        st.markdown(f'<div class="ch1-img-label">PROCESSED · {proc_label.upper()}</div>',
                    unsafe_allow_html=True)
        st.image(im_proc, use_container_width=True)
    with c_curve:
        st.markdown('<div class="ch1-img-label">MAPPING CURVE</div>', unsafe_allow_html=True)
        if fig_curve is not None:
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)
            
def _show_images_and_curve_just_for_hystogram(im_orig, im_proc, proc_label, fig_curve):
    """Images (left 2/3) + mapping curve (right 1/3) on the same row."""
    c_orig, c_proc, c_curve = st.columns([5, 5, 4], gap="medium")
    with c_orig:
        st.markdown('<div class="ch1-img-label">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_orig), use_container_width=True)
    with c_proc:
        st.markdown(f'<div class="ch1-img-label">PROCESSED · {proc_label.upper()}</div>',
                    unsafe_allow_html=True)
        st.image(imNormalize(im_proc), use_container_width=True)
    with c_curve:
        st.markdown('<div class="ch1-img-label">MAPPING CURVE</div>', unsafe_allow_html=True)
        if fig_curve is not None:
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)

def _show_histograms(im_orig, im_proc, method_name):
    h1 = f_histogram(im_orig, 256, 256)
    h2 = f_histogram(im_proc, 256, 256)
    hc1, hc2 = st.columns(2, gap="medium")
    with hc1:
        fh1 = _dark_histogram(h1, "Histogram · Original", '#7b61ff')
        st.pyplot(fh1, use_container_width=True); plt.close(fh1)
    with hc2:
        fh2 = _dark_histogram(h2, f"Histogram · {method_name}", '#06b6d4')
        st.pyplot(fh2, use_container_width=True); plt.close(fh2)

def _build_report(im_orig, im_proc, method, params):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    def _ax(ax, title, arr):
        ax.imshow(arr, cmap='gray', aspect='auto')
        ax.set_title(title, color='#60a5fa', fontsize=9, pad=5)
        ax.axis('off')

    _ax(axes[0, 0], "ORIGINAL",             imNormalize(im_orig))
    _ax(axes[0, 1], f"PROCESSED · {method.upper()}", imNormalize(im_proc))

    h1 = f_histogram(im_orig, 256, 256)
    h2 = f_histogram(im_proc, 256, 256)
    for ax, h, lbl, col in [
        (axes[1, 0], h1, "Histogram: Original",  '#7b61ff'),
        (axes[1, 1], h2, f"Histogram: {method}", '#06b6d4'),
    ]:
        ax.set_facecolor('#0a0f1e')
        ax.bar(range(len(h)), h, color=col, alpha=0.85, width=1.0)
        ax.set_title(lbl, color='#60a5fa', fontsize=9, pad=5)
        ax.tick_params(colors='#475569', labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')

    axes[0, 2].set_facecolor('#0a0f1e'); axes[0, 2].axis('off')
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
    axes[0, 2].text(0.05, 0.97, txt, transform=axes[0, 2].transAxes,
                    color='#94a3b8', fontsize=7.8, va='top', linespacing=1.65,
                    fontfamily='monospace')

    axes[1, 2].set_facecolor('#0a0f1e'); axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, "MedVision\nImage Analysis Platform",
                    transform=axes[1, 2].transAxes, color='#1e3a5f',
                    fontsize=18, va='center', ha='center', fontfamily='monospace', alpha=0.6)

    fig.suptitle(f"Report  ·  {now}", color='#334155', fontsize=8, y=0.01)
    plt.tight_layout(pad=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#070d1a')
    buf.seek(0); plt.close(fig)
    return buf
def _download_section(im_orig, im_proc, method, params):
    _divider("Export")

    d1, d2, d3 = st.columns(3, gap="small")

    proc_u8 = imNormalize(im_proc)
    orig_u8 = imNormalize(im_orig)

    with d1:
        buf = io.BytesIO()
        Image.fromarray(proc_u8).save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            "⬇ Processed Image",
            buf,
            file_name=f"{method.replace(' ','_')}_processed.png",
            mime="image/png",
            use_container_width=True,
            key=f"processed_{method}"
        )

    with d2:
        buf = io.BytesIO()
        Image.fromarray(orig_u8).save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            "⬇ Original Image",
            buf,
            file_name="original.png",
            mime="image/png",
            use_container_width=True,
            key=f"original_{method}"
        )

    with d3:
        report_buf = _build_report(im_orig, im_proc, method, params)

        st.download_button(
            "⬇ Full Report (PNG)",
            report_buf,
            file_name=f"Report_{method.replace(' ','_')}.png",
            mime="image/png",
            use_container_width=True,
            key=f"report_{method}"
        )
# ─────────────────────────────────────────────
#  ORIGINAL FUNCTIONS — αναλλοίωτα
# ─────────────────────────────────────────────

def simpleDisplay(im, image_depth, tones):
    im1 = np.zeros(im.shape, dtype=float)
    im1 = np.round(((tones - 1) / (image_depth - 1)) * (im - 0))
    return im1

def optimalDisplay(im, tones):
    im1 = np.zeros(im.shape, dtype=float)
    vmn = int(np.min(im))
    vmx = int(np.max(im))
    im1 = (((tones - 1) * (im - vmn) / (vmx - vmn)))
    im1 = np.around(im1)
    return im1

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

def brokenWindow(im, image_depth, tones, gray_val, im_val):
    im = np.asarray(im, dtype=float)
    N = np.size(im, 0)
    M = np.size(im, 1)
    im1 = np.zeros(im.shape, dtype=float)
    for i in range(N):
        for j in range(M):
            if im[i][j] <= im_val:
                im1[i][j] = (gray_val / (im_val)) * im[i][j]
            else:
                im1[i][j] = (((tones - 1) - (gray_val + 1)) / (image_depth - (im_val + 1))) * (im[i][j] - (im_val + 1)) + (gray_val + 1)
    im1 = np.round(im1)
    return im1

def doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros(im.shape, dtype=float)
    N = np.size(im, 0)
    M = np.size(im, 1)
    half = (tones / 2) - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0)
    vs1 = ve1 - ww1
    ve2 = round((2.0 * wl2 + ww2) / 2.0)
    vs2 = ve2 - ww2
    if vs2 < ve1:
        new_point = round(((vs2 + ve1)) / 2.0)
        ve1 = new_point
        vs2 = ve1
    if vs1 < 0:
        vs1 = 0
    if ve2 > image_depth:
        ve2 = image_depth
    for i in range(N):
        for j in range(M):
            if im[i][j] < vs1:
                im1[i][j] = 0
            if im[i][j] >= vs1 and im[i][j] <= ve1:
                t = round(((half - 0) / (ve1 - vs1)) * (im[i][j] - vs1) + 0.0)
                im1[i][j] = round(t)
            if im[i][j] > ve1 and im[i][j] < vs2:
                im1[i][j] = half + 1
            if im[i][j] >= vs2 and im[i][j] <= ve2:
                t = round((((tones - 1) - (half + 1)) / (ve2 - vs2)) * (im[i][j] - vs2) + (half + 1))
                im1[i][j] = round(t)
            if im[i][j] > ve2:
                im1[i][j] = tones - 1
    return im1

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

def formPlotFunction(tones, Choice):
    import math
    w = np.zeros(tones)
    for i in range(0, tones):
        if Choice == 0:
            w[i] = tones - i - 1
            text = 'inverse'
        elif Choice == 1:
            r = 0.05
            w[i] = math.log(1 + r * i)
            text = 'logarithmic'
        elif Choice == 2:
            c = 128
            w[i] = np.exp(i) ** (1 / c) - 1
            text = 'inverse logarithmic'
        elif Choice == 3:
            gamma = 0.55
            w[i] = i ** gamma
            text = 'power'
        elif Choice == 4:
            w[i] = np.sin(2 * np.pi * i / (4 * (tones - 1)))
            text = 'sine-window'
        elif Choice == 5:
            w[i] = 1 - np.exp(-i / 90)
            text = 'exp-window'
        elif Choice == 6:
            w[i] = (1 / (1 + (np.exp(-i / 70))))
            text = 'sigmoid'
        elif Choice == 7:
            w[i] = np.cos(2 * np.pi * i / (4 * (tones - 1)))
            text = 'cos-window'
        elif Choice == 8:
            w[i] = 1 / np.sqrt(max(i, 1e-6))
            text = 'reciprocal-square-root-function'
    w = (tones - 1) * ((w - np.min(w)) / (np.max(w) - np.min(w)))
    return (w, text)

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

def f_histogram(A, image_depth, tones):
    minA = 0; maxA = image_depth
    if np.max(A) > (tones - 1):
        B = np.round((tones - 1) * ((A - minA) / (maxA - minA)))
    else:
        B = A
    M = np.size(B, 0)
    N = np.size(B, 1)
    Bval = np.reshape(B, M * N)
    h = np.zeros(tones, dtype=float)
    for i in range(np.size(Bval)):
        val = np.int16(Bval[i])
        h[val] = h[val] + 1
    return h

def f_hequalization (A,image_depth,tones):
    #implement HE method
    minA=0;maxA=image_depth;
    B=np.round((tones-1)*((A-minA)/(maxA-minA) ))
    M=np.size(B,0);N=np.size(B,1)
    Bval=np.reshape(B,M*N)#from 2d to 1d dimension
    p=np.argsort(Bval)#keep initial positions of sorted values
    neq=np.int32((M*N)/tones+0.5) #calculate number of pixels per gray-tone
    BL=len(Bval)
    az=np.int32(np.fix((N*M)/neq))#get integral number of pixels per gray level
    zRem=np.int32(np.remainder(BL,neq))#get  remainder of pixels if not intergal part of neq
    D=np.zeros(M*N)#define 1d array to hold the transformed image matrix
    k=-1;
    for i in range(0,(neq*az),neq):
        k=k+1
        for j in range(0,neq):
            D[i+j]=k
    if(zRem>0):
        for i in range( (neq*az),( (az*neq)+zRem) ):
            D[i]=tones-1
    #reassign equalized values into their proper position
    L=np.zeros(M*N)
    k=-1;
    for i in range (M):
        for j in range (N):
            k=k+1;
            L[p[k]]=D[k];
    #finalize equalized image
    Z=np.reshape(L,B.shape)
    Z=imNormalize(Z,tones)
    return (Z)

def CDF_equalization_formula(im, image_depth, tones):
    B = np.round((tones - 1) * ((im) / (np.max(im))))
    M = np.size(im, 0); N = np.size(im, 1)
    CDFh = np.zeros(tones, dtype=float)
    CDFq = np.zeros(tones, dtype=float)
    h = f_histogram(im, image_depth, tones)
    tone_values = ((M * N) / tones)
    q = (tone_values * np.ones(tones, dtype=float))
    for i in range(tones):
        for j in range(i + 1):
            CDFh[i] = CDFh[i] + h[j]
            CDFq[i] = CDFq[i] + q[j]
    B = CDFh[np.int32(B)] / tone_values - 1
    B = np.round(B)
    return B

def CLAHE_method(im, im_depth, clip_limit):
    im_uint8 = np.uint8(im)
    im_CLAHE = exposure.equalize_adapthist(im_uint8, clip_limit=clip_limit)
    im_CLAHE = (im_depth - 1) * im_CLAHE / np.max(im_CLAHE)
    if im_depth <= 8:
        im_CLAHE = np.uint8(np.round(im_CLAHE))
    elif im_depth <= 16:
        im_CLAHE = np.uint16(np.round(im_CLAHE))
    else:
        im_CLAHE = np.round(im_CLAHE)
    return im_CLAHE


# ─────────────────────────────────────────────
#  RUN
def run():
    _inject_css()

    uploaded = st.session_state.get("uploaded_file", None)
    if uploaded is None:
        st.warning("Upload an image from the Dashboard.")
        st.stop()

    im = loadImage(uploaded)
    

    tones       = 256
    image_depth = 256

    _show_badges(im, uploaded.name)

    tab_Display, tab_Window, tab_NonLinear, tab_Hyst = st.tabs([
        "  Display Methods",
        "  Window Methods",
        "  Non-Linear Mapping",
        "  Histogram Techniques",
    ])

    # ══════════════════════════════════════════
    with tab_Display:
        filters = ["Simple Display", "Optimal Display"]
        _section("Method")
        Choice = st.selectbox("", filters, key="ch1_disp")


        if Choice == "Simple Display":
            im2 = simpleDisplay(im, image_depth, tones)
            w_c = np.array([(tones - 1) / (image_depth - 1) * v for v in range(tones)])
        else:
            im2 = optimalDisplay(im, tones)
            mn_v, mx_v = np.min(im), np.max(im)
            w_c = np.round((tones - 1) * (np.arange(tones) - mn_v) / max(mx_v - mn_v, 1))
       
        _section("Apply Window to Processed Image")
        use_sw = st.checkbox("Enable Windowing", key="ch1_disp_sw")
        wc_d, ww_d = 128, 100
        if use_sw:
            cd1, cd2 = st.columns(2)
            with cd1: wc_d = st.slider("Window Center (WC)", 0, 255, 128, key="ch1_disp_wc")
            with cd2: ww_d = st.slider("Window Width (WW)",  1, 255, 100, key="ch1_disp_ww")

        if use_sw and im2 is not None:
            im2 = simpleWindow(im2, wc_d, ww_d, image_depth, tones)
            Va = max((2.0 * wc_d + ww_d) / 2.0 - ww_d, 0)
            Vb = min((2.0 * wc_d + ww_d) / 2.0, image_depth)
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Va = {Va:.0f}</span>'
                f'<span class="ch1-wp">Vb = {Vb:.0f}</span>'
                f'<span class="ch1-wp">WC = {wc_d}</span>'
                f'<span class="ch1-wp">WW = {ww_d}</span>'
                f'</div>', unsafe_allow_html=True)
            w_sw = np.clip((np.arange(tones) - Va) / max(Vb - Va, 1) * (tones - 1), 0, tones - 1)
            fig_curve   = _dark_plot(w_sw, f"{Choice} + Simple Window")
            params_text = f"method={Choice}, WC={wc_d}, WW={ww_d}"
        else:
            fig_curve   = _dark_plot(w_c, Choice)
            params_text = f"depth={image_depth}, tones={tones}"

        method_name = Choice if not use_sw else f"{Choice} + SW"

        # ── OUTPUT ──
    
    
        _divider("Output")
        _show_images_and_curve(im, im2, method_name, fig_curve)
        _download_section(im, im2, method_name, params_text)
        
    # ══════════════════════════════════════════
    with tab_Window:
        filters_win = ["Simple Window", "Broken Window", "Double Window"]
        _section("Method")
        Choice = st.selectbox("", filters_win, key="ch1_win")

        if Choice == "Simple Window":
            c1, c2 = st.columns(2)
            with c1: wc = st.slider("Window Center (WC)", 0, 255, 128, key="ch1_wc")
            with c2: ww = st.slider("Window Width (WW)",  1, 255, 100, key="ch1_ww")
            Va = max((2.0 * wc + ww) / 2.0 - ww, 0)
            Vb = min((2.0 * wc + ww) / 2.0, image_depth)
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Va = {Va:.0f}</span>'
                f'<span class="ch1-wp">Vb = {Vb:.0f}</span>'
                f'<span class="ch1-wp">WC = {wc}</span>'
                f'<span class="ch1-wp">WW = {ww}</span>'
                f'</div>', unsafe_allow_html=True)
            im2 = simpleWindow(im, wc, ww, image_depth, tones)
            params_text = f"WC={wc}, WW={ww}, Va={Va:.0f}, Vb={Vb:.0f}"
            w_line = np.clip((np.arange(tones) - Va) / max(Vb - Va, 1) * (tones - 1), 0, tones - 1)
            fig_curve = _dark_plot(w_line, f"Simple Window  WC={wc}  WW={ww}")

        elif Choice == "Broken Window":
            c1, c2 = st.columns(2)
            with c1: gray_val = st.slider("Gray split value", 0, 255, 128, key="ch1_gv")
            with c2: im_val   = st.slider("Intensity split",  0, 255, 128, key="ch1_iv")
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Gray split = {gray_val}</span>'
                f'<span class="ch1-wp">Intensity split = {im_val}</span>'
                f'</div>', unsafe_allow_html=True)
            im2 = brokenWindow(im, image_depth, tones, gray_val, im_val)
            params_text = f"gray_split={gray_val}, intensity_split={im_val}"
            x_arr = np.arange(tones, dtype=float)
            slope1 = gray_val / max(im_val, 1)
            slope2 = (tones - 1 - gray_val) / max(image_depth - im_val - 1, 1)
            w_bw = np.where(
               x_arr <= im_val,
               slope1 * x_arr,
               slope2 * (x_arr - im_val - 1) + (gray_val + 1))
            
            fig_curve = _dark_plot(np.clip(w_bw, 0, tones - 1), "Broken Window")

        elif Choice == "Double Window":
            c1, c2 = st.columns(2)
            with c1:
                
                wl1 = st.slider("Window Center 1 (WL1)", 0, 255,  80, key="ch1_wl1")
                ww1 = st.slider("Window Width 1 (WW1)", 1, 255,  60, key="ch1_ww1")
            with c2:
                wl2 = st.slider("Window Center 2 (WL2)",min_value=wl1,max_value=255,value=max(160, wl1),key="ch1_wl2")
                ww2 = st.slider("Window Width 2 (WW2)", 1, 255,  60, key="ch1_ww2")
            ve1 = round((2.0 * wl1 + ww1) / 2.0); vs1 = ve1 - ww1
            ve2 = round((2.0 * wl2 + ww2) / 2.0); vs2 = ve2 - ww2
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Window 1: [{max(vs1,0)} – {ve1}]</span>'
                f'<span class="ch1-wp">Window 2: [{vs2} – {min(ve2,image_depth)}]</span>'
                f'</div>', unsafe_allow_html=True)
            if vs2 < ve1:
               new_point = round((vs2 + ve1) / 2.0)
               ve1 = new_point
               vs2 = ve1
            if vs1 < 0:   vs1 = 0
            if ve2 > image_depth: ve2 = image_depth
        
            half = (tones / 2) - 1      # ← αυτό έλειπε
        
            im2 = doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones)
            params_text = f"WL1={wl1}, WW1={ww1}, WL2={wl2}, WW2={ww2}"
        
            x_arr = np.arange(tones, dtype=float)
            w_dw  = np.zeros(tones, dtype=float)
            for i in range(tones):
               v = x_arr[i]
               if v <= vs1:
                   w_dw[i] = 0
               elif v > vs1 and v <= ve1:
                   w_dw[i] = (half / max(ve1 - vs1, 1)) * (v - vs1)
               elif v > ve1 and v < vs2:
                   w_dw[i] = half + 1
               elif v >= vs2 and v <= ve2:
                   w_dw[i] = (half / max(ve2 - vs2, 1)) * (v - vs2) + (half + 1)
               elif v > ve2:
                   w_dw[i] = tones - 1
        
            fig_curve = _dark_plot(np.clip(w_dw, 0, tones - 1),
                                  f"Double Window  WL1={wl1} WW1={ww1}  WL2={wl2} WW2={ww2}")

        method_name = Choice

        # ── OUTPUT ──
        
        _divider("Output")
        _show_images_and_curve(im, im2, method_name, fig_curve)
        _download_section(im, im2, method_name, params_text)
    
    # ══════════════════════════════════════════
    with tab_NonLinear:
    
        filters_nl = [
            "Inverse", "Logarithmic", "Inverse Log", "Power",
            "Sine", "Exponential", "Sigmoid", "Cosine", "Reciprocal sqrt"
        ]
        _section("Function")
        Choice = st.selectbox("", filters_nl, key="ch1_nl")
        _section("Apply Window to Processed Image")
        use_sw_nl = st.checkbox("Enable Windowing", key="ch1_nl_sw")
        wc_nl, ww_nl = 128, 100
        if use_sw_nl:
            cn1, cn2 = st.columns(2)
            with cn1: wc_nl = st.slider("Window Center (WC)", 0, 255, 128, key="ch1_nl_wc")
            with cn2: ww_nl = st.slider("Window Width (WW)",  1, 255, 100, key="ch1_nl_ww")

        mapping_dict = {
            "Inverse": 0, "Logarithmic": 1, "Inverse Log": 2, "Power": 3,
            "Sine": 4, "Exponential": 5, "Sigmoid": 6, "Cosine": 7, "Reciprocal sqrt": 8
        }
        c_idx = mapping_dict[Choice]
        (w, stext) = formPlotFunction(tones, c_idx)

        N_im = np.size(im, 0); M_im = np.size(im, 1)
        mn_v = np.min(im);     mx_v = np.max(im)
        im2 = np.round((tones - 1) * (im - mn_v) / max(mx_v - mn_v, 1))
        for i in range(N_im):
            for j in range(M_im):
                v = int(im2[i, j])
                im2[i, j] = np.int32(w[v])

        if use_sw_nl:
            im2 = simpleWindow(im2, wc_nl, ww_nl, image_depth, tones)
            Va = max((2.0 * wc_nl + ww_nl) / 2.0 - ww_nl, 0)
            Vb = min((2.0 * wc_nl + ww_nl) / 2.0, image_depth)
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Va = {Va:.0f}</span>'
                f'<span class="ch1-wp">Vb = {Vb:.0f}</span>'
                f'<span class="ch1-wp">WC = {wc_nl}</span>'
                f'<span class="ch1-wp">WW = {ww_nl}</span>'
                f'</div>', unsafe_allow_html=True)
            w_sw = np.clip((np.arange(tones) - Va) / max(Vb - Va, 1) * (tones - 1), 0, tones - 1)
            fig_curve   = _dark_plot(w_sw, f"{Choice} + Simple Window")
            params_text = f"function={stext}, WC={wc_nl}, WW={ww_nl}"
        else:
            fig_curve   = _dark_plot(w, f"{Choice} Mapping")
            params_text = f"function={stext}"

        method_name = Choice if not use_sw_nl else f"{Choice} + SW"

        # ── OUTPUT ──

        _divider("Output")
        _show_images_and_curve(im, im2, method_name, fig_curve)
        _download_section(im, im2, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_Hyst:
        
        filters_hist = ["Histogram Equalization", "CDF Equalization", "CLAHE"]
        _section("Method")
        Choice = st.selectbox("", filters_hist, key="ch1_hist")
        _section("Apply Window to Processed Image")
        use_sw_h = st.checkbox("Enable Windowing", key="ch1_hist_sw")
        wc_h, ww_h = 128, 100
        if use_sw_h:
            ch1_c, ch2_c = st.columns(2)
            with ch1_c: wc_h = st.slider("Window Center (WC)", 0, 255, 128, key="ch1_hist_wc")
            with ch2_c: ww_h = st.slider("Window Width (WW)",  1, 255, 100, key="ch1_hist_ww")

        if Choice == "Histogram Equalization":
            im2 = f_hequalization(im, image_depth, tones)
            params_text = "global HE"
        elif Choice == "CDF Equalization":
            im2 = CDF_equalization_formula(im, image_depth, tones)
            params_text = "CDF-based HE"
        elif Choice == "CLAHE":

            im2 = exposure.equalize_adapthist(np.uint8(im), clip_limit=0.03)
            im2=255*im2/np.max(im2)
            clip=0.03
            params_text = f"clip={clip}"

        if use_sw_h and im2 is not None:
            im2 = simpleWindow(im2, wc_h, ww_h, image_depth, tones)
            Va = max((2.0 * wc_h + ww_h) / 2.0 - ww_h, 0)
            Vb = min((2.0 * wc_h + ww_h) / 2.0, image_depth)
            st.markdown(
                f'<div class="ch1-win-params">'
                f'<span class="ch1-wp">Va = {Va:.0f}</span>'
                f'<span class="ch1-wp">Vb = {Vb:.0f}</span>'
                f'<span class="ch1-wp">WC = {wc_h}</span>'
                f'<span class="ch1-wp">WW = {ww_h}</span>'
                f'</div>', unsafe_allow_html=True)
            params_text += f", WC={wc_h}, WW={ww_h}"

        method_name = Choice if not use_sw_h else f"{Choice} + SW"
        # ── OUTPUT (χωρίς mapping curve, με ιστογράμματα) ──
       
        # safety για HE/CLAHE
        _divider("Output")
        if Choice == "Histogram Equalization":
         _show_images_and_curve_just_for_hystogram_eq(im, im2, method_name, fig_curve=None)
        else:
          _show_images_and_curve_just_for_hystogram(im, im2, method_name, fig_curve=None)  
        _divider("Histograms")
        _show_histograms(im, im2, method_name)
        _download_section(im, im2, method_name, params_text)