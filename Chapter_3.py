
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import streamlit as st
from PIL import Image
import io
import pydicom as dicom


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def filterImage(im, FH, tones):
    Fim = np.fft.fftshift(np.fft.fft2(im))
    Fim1 = Fim * FH
    im1 = np.real(np.fft.ifft2(np.fft.ifftshift(Fim1)))
    return im1  


def prepare_image(x):
    x = x.astype(float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)



def simpleWindow(im,wc,ww,image_depth, tones):
    im1=np.asarray(im, dtype=float)
    Vb=(2.0*wc+ww)/2.0;
    Va=Vb-ww; 
    if(Vb>image_depth):
        Vb=image_depth;
    if(Va<0): 
        Va=0;
    im1=(((tones-1)*((im1-Va)/(Vb-Va))))
    M=np.size(im1,0)
    N=np.size(im1,1)
    for i in range (0,M):
        for j in range(N):
            if(  (im1[i][j]>=Va) and (im1[i][j]<=Vb)  ):    
                im1[i][j]=(((tones-1)*(im1[i][j]-Va)/(Vb-Va)))
            elif (im1[i][j]<Va):
                im1[i][j]=0;
            elif (im1[i][j]>Vb):
                im1[i][j]=tones-1;
    return im1 




def DeconvolveImage(im,FH):
    Fim=np.fft.fft2(im)
    Fim1=Fim*np.fft.fftshift(FH)
    im1=np.real(np.fft.ifft2(Fim1))
    return im1
  
def design2dFilter(im,fh):
    y=np.size(im,0);x=np.size(im,1);
    FH=np.zeros(np.shape(im),dtype=float);
    
    for k in range (y):
        for m in range(x):
            K=y/2-k+1;
            M=x/2-m+1;
            ir = np.sqrt(K*K + M*M)
            ir = int(ir)
            if(ir>len(fh)):
                ir=len(fh)-1;
            FH[k][m]=fh[ir];
    
    FH=np.fft.fftshift(FH)
    return(FH)


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

def _show_images_only(im_orig, im_proc, proc_label):
    c_orig, c_proc = st.columns(2, gap="medium")
    with c_orig:
        st.markdown('<div class="ch1-img-label">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_orig,256).astype(np.uint8), use_container_width=True)
    with c_proc:
        st.markdown(f'<div class="ch1-img-label">PROCESSED · {proc_label.upper()}</div>', unsafe_allow_html=True)
        st.image(imNormalize(im_proc,256).astype(np.uint8), use_container_width=True)


def ampl_fft2( im):
 im1=np.fft.fft2(im)
 im1=np.fft.fftshift(im1)
 im1=np.round(10.0*np.log(np.abs(im1)+1))
 return(im1)



def _show_freq_domain(im, fh, FH, sText, filter_subtype, family_name):
    """FFT αρχικής + 1D response (όπως το έκανε ο χρήστης) + 2D mask."""
    col_fft, col_1d, col_2d = st.columns(3, gap="medium")

    # FFT spectrum αρχικής
    with col_fft:
        st.markdown('<div class="ch1-img-label">FFT SPECTRUM · ORIGINAL</div>', unsafe_allow_html=True)
        F         = np.fft.fftshift(np.fft.fft2(im))
        magnitude = np.log1p(np.abs(F))
        fig, ax   = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
        ax.set_facecolor('#070d1a')
        ax.imshow(magnitude, cmap='inferno', aspect='auto')
        ax.set_title("FFT Spectrum", color='#94a3b8', fontsize=8.5, pad=5)
        ax.axis('off')
        plt.tight_layout(pad=0.7)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # 1D filter — ακριβώς όπως το είχε ο χρήστης, dark theme
    with col_1d:
        st.markdown('<div class="ch1-img-label">1-D FILTER RESPONSE</div>', unsafe_allow_html=True)
        N    = len(fh)
        N2   = [N // 2]
        fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
        ax.set_facecolor('#070d1a')
        ax.plot([N2, N2], [0, 1],          color='#06b6d4', linewidth=2,   linestyle='--', label='axis of symmetry')
        ax.plot(np.fft.fftshift(fh), '--', color='#f59e0b', linewidth=1.4,                label='filter shifted')
        ax.plot(fh,                        color='#60a5fa', linewidth=1.8,                label='filter')
        ax.set_title(f"1-D filter: {sText}", color='#94a3b8', fontsize=8.5, pad=5)
        ax.set_xlabel("Spatial frequencies", color='#475569', fontsize=7.5)
        ax.set_ylabel("Amplitude",           color='#475569', fontsize=7.5)
        ax.tick_params(colors='#475569', labelsize=6.5)
        ax.legend(fontsize=6, labelcolor='#94a3b8', facecolor='#0a0f1e', edgecolor='#1e3a5f')
        ax.grid(True, color='#0d1b2e', linewidth=0.8, linestyle='--')
        # ── label για LP/HP/BP/BR ──
        ax.set_title(f"1-D · {family_name} {filter_subtype}", color='#94a3b8', fontsize=8.5, pad=5)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
        plt.tight_layout(pad=0.7)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # 2D filter mask — gray cmap όπως το είχε ο χρήστης, dark theme
    with col_2d:
        st.markdown(
            '<div class="ch1-img-label">FFT SPECTRUM · FILTERED</div>',
            unsafe_allow_html=True
        )
    
        fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    
        ax.set_facecolor('#070d1a')
    
        mask_vis = np.log1p(np.abs(np.fft.fftshift(FH)))
    
        ax.imshow(mask_vis, cmap='inferno', aspect='auto')
    
        ax.set_title(
            f"Filtered Spectrum · {family_name} {filter_subtype}",
            color='#94a3b8',
            fontsize=8.5,
            pad=5
        )
    
        ax.axis('off')
    
        plt.tight_layout(pad=0.7)
    
        st.pyplot(fig, use_container_width=True)
    
        plt.close(fig)


def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn);    
    w=np.round(w)
    return w




def Ideal(N,fco,TYPE,enh,trans,w):
    N = int(np.int32(N))
    fh=np.zeros(np.int32(N),dtype=float)
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)


    if(TYPE==1):#LP
        fh = np.ones(N)
        sText = "Ideal LP filter"
        center = N // 2
        for k in range(N):
          if abs(k - center) > fco:
            fh[k] = 0 + enh


    elif(TYPE==2):     
        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(fco),np.int32(L)):
            fh[k]=1;#Ideal HP
            sText=" Ideal HP filter"
    elif(TYPE==3):     
        d=trans;
        fh=np.ones(np.int32(N),dtype=float)
        for k in range(np.int32(d-w/2+0.5),np.int32(d+w/2+0.5)):
            fh[k]=0+enh;#Ideal BR
        sText=" Ideal BR filter"


    elif(TYPE==4):     
        d=trans;
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(d-(w/2) + 0.5),np.int32(d+(w/2)+0.5)):
            fh[k]=1;#Ideal BP
        sText=" Ideal BP filter"    
    else: 
         print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )


    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
    return(fh/np.max(fh),sText) 


def Butterworth(N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)


    
    if((N % 2)==0):
        L=np.round(N/2+1) 
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)


    if (TYPE==1):#BLP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (k/fco), (2*ndegree)) );#LP
        sText='Butterworth LP'
    elif(TYPE==2):     
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (fco/(k+0.001)), (2*ndegree)) );#HP
        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Butterworth HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (fco/(k-d+0.001)), (2*ndegree)) );#BR
        sText='Butterworth BR'
    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( ((k-d)/fco), (2*ndegree)) );#BP
        sText='Butterworth BP'    
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)
#--------------------------------------------------------------
def Exponential( N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)
    print("--EXPONENTIAL FILTERS----")
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)


    if (TYPE==1):#ELP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=np.exp( (-np.log(2))*(k/fco)**ndegree)
        sText='Exponential LP'
    elif(TYPE==2): #EHP    
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*(fco/(k+0.0001))**ndegree)


        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Exponential HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*(fco/(k-d+0.00001))**ndegree)
        sText='Exponential BR'


    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*((k-d)/fco)**ndegree)
            
        sText='Exponential BP'
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)




#--------------------------------------------------------------
def Gaussian( N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)
    print("--GAUSSIAN FILTERS----")
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)


    if (TYPE==1):#gLP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=np.exp( -(k**2/(2*fco**2))**ndegree)
        sText='Gaussian LP'
    elif(TYPE==2): #GHP    
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=np.exp(-(2*fco**2/(k+0.0001)**2)**ndegree)


        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Gaussian HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=np.exp(-(2*fco**2/(k-d+0.00001)**2)**ndegree)
        sText='Gaussian BR'


    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=np.exp(-((k-d)**2/(2*fco**2))**ndegree)
            
        sText='Gaussian BP'
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)


def GaussianMTF(N):
    
    fh=np.zeros(np.int32(N),dtype=float)
    
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)


    sigma=L/2-1
    for k in range(np.int32(L)):
        fh[k]=np.exp(-k**2/(2*sigma**2))
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
    return(fh)




def generalizedWienerFilter(im1,fh,filtType):
    N=len(fh)
    SIGMA=np.std(im1)
    C=0.1/SIGMA
    
    if (filtType==1):
        a=1;b=0
        sText=" Inverse Filter";
    elif (filtType==2):
        a=0;b=1;
        sText=" Wiener Filter";
    elif(filtType==3):
        a=0.5;b=1;
        sText=" Power Filter";
    fhh=np.zeros(np.int32(N),dtype=float)
    
    
    for k in range(np.int32(N)):
        fhh[k]=(fh[k]**2/ ( (fh[k]**2+b*C)) )**(1-a)/fh[k]
        if(fhh[k]<C):
            fhh[k]=(fh[k]**2/ ( (fh[k]**2+b*C)) )**(1-a)/C
    fhh=fhh/np.max(fhh)
    return (fhh, sText)  

import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import streamlit as st
from PIL import Image
import io
import datetime


# ─────────────────────────────────────────────
#  CSS
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

    /* ── HEADERS — αλλάξτε εδώ font-size / color / border-left ── */
    .ch3-section-title {
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
    st.markdown(f'<div class="ch3-section-title">{label}</div>', unsafe_allow_html=True)


def _divider(label=""):
    st.markdown(
        f'<div class="ch1-divider"><span>{label}</span></div>',
        unsafe_allow_html=True
    )


def _dark_plot_1d(fh, sText):
    N = len(fh)
    N2_nor = [N // 2]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    ax.set_facecolor('#070d1a')
    ax.plot(fh,                        color='#60a5fa', linewidth=1.6, label='filter')
    ax.plot(np.fft.fftshift(fh), '--', color='#f59e0b', linewidth=1.2, label='shifted')
    ax.plot([N2_nor, N2_nor], [0, 1],  color='#06b6d4', linewidth=1.5, linestyle='--', label='axis')
    ax.set_title("1-D filter: " + sText, color='#94a3b8', fontsize=8.5, pad=5)
    ax.set_xlabel("Spatial frequency", color='#475569', fontsize=7.5)
    ax.set_ylabel("Amplitude",         color='#475569', fontsize=7.5)
    ax.tick_params(colors='#475569', labelsize=6.5)
    ax.legend(fontsize=6, labelcolor='#94a3b8', facecolor='#0a0f1e', edgecolor='#1e3a5f')
    ax.grid(True, color='#0d1b2e', linewidth=0.8, linestyle='--')
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    plt.tight_layout(pad=0.7)
    return fig


def _dark_plot_2d(FH, title):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    ax.set_facecolor('#070d1a')
    ax.imshow(np.fft.fftshift(FH), cmap='magma', aspect='auto')
    ax.set_title(title, color='#94a3b8', fontsize=8.5, pad=5)
    ax.axis('off')
    plt.tight_layout(pad=0.7)
    return fig


def _dark_fft(im):
    """FFT magnitude spectrum της αρχικής εικόνας."""
    F = np.fft.fftshift(np.fft.fft2(im))
    magnitude = np.log1p(np.abs(F))
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0f1e')
    ax.set_facecolor('#070d1a')
    ax.imshow(magnitude, cmap='inferno', aspect='auto')
    ax.set_title("FFT Spectrum · Original", color='#94a3b8', fontsize=8.5, pad=5)
    ax.axis('off')
    plt.tight_layout(pad=0.7)
    return fig


def _build_report(im_orig, im_proc, method, params):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#070d1a')
    fig.patch.set_facecolor('#070d1a')
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    for ax, arr, title in [
        (axes[0], im_orig, "ORIGINAL"),
        (axes[1], im_proc, f"PROCESSED · {method.upper()}"),
    ]:
        ax.imshow(arr.astype(np.uint8), cmap='gray', aspect='auto')
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


def _download_section(im_orig, im_proc, method, params):
    _divider("Export")
    d1, d2, d3 = st.columns(3, gap="small")

    def _to_uint8(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx == mn:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

    with d1:
        buf = io.BytesIO()
        Image.fromarray(_to_uint8(im_proc)).save(buf, format="PNG"); buf.seek(0)
        st.download_button(
            "⬇ Processed Image", buf,
            file_name=f"{method.replace(' ','_')}_processed.png",
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
        report_buf = _build_report(_to_uint8(im_orig), _to_uint8(im_proc), method, params)
        st.download_button(
            "⬇ Full Report (PNG)", report_buf,
            file_name=f"Report_{method.replace(' ','_')}.png",
            mime="image/png", use_container_width=True,
            key=f"dl_rep_{method}"
        )


# ─────────────────────────────────────────────
#  RUN  (τα functions αναλλοίωτα)
# ─────────────────────────────────────────────
def run():
    _inject_css()

    uploaded = st.session_state.get("uploaded_file", None)
    if uploaded is None:
        st.warning("Upload an image from the Dashboard.")
        st.stop()

    im = loadImage(uploaded)
    
    im = imNormalize(im, 256)

    tones       = 256
    image_depth = 256

    _show_badges(im, uploaded.name)

    M, N    = im.shape
    Flength = int(np.sqrt(M * M + N * N))

    # ── Tabs για Filter Family ──
    tab_BW, tab_Exp, tab_Gauss, tab_Ideal, tab_Wien = st.tabs([
        "  Butterworth",
        "  Exponential",
        "  Gaussian",
        "  Ideal",
        "  Generalized Wiener",
    ])

    # ══════════════════════════════════════════
    with tab_BW:
        _section("Filter Parameters")
        Filter = st.selectbox("Sub-type", ["Low Pass", "High Pass", "Band Pass", "Band Reject"], key="bw_sub")
        TYPE   = {"Low Pass": 1, "High Pass": 2, "Band Pass": 4, "Band Reject": 3}[Filter]

        c1, c2 = st.columns(2)
        with c1:
            fco     = st.slider("Cutoff frequency", 1, Flength // 2, Flength // 4, key="bw_fco")
            ndegree = st.slider("Order",             1, 5,            2,             key="bw_deg")
        with c2:
            trans   = st.slider("Transition",        1, Flength // 2, Flength // 3, key="bw_trans")

        fh, sText   = Butterworth(Flength, ndegree, fco, TYPE, trans)
        params_text = f"Butterworth {Filter}, fco={fco}, order={ndegree}, trans={trans}"
        method_name = f"Butterworth · {Filter}"

        FH  = design2dFilter(im, fh)
        im1 = filterImage(im, FH, tones)
        im1 = imNormalize(im1, tones)

        apply_window = st.checkbox("Apply Windowing", key="bw_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="bw_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="bw_ww")
            st.markdown(f'<div class="ch1-win-params"><span class="ch1-wp">WC = {wc}</span><span class="ch1-wp">WW = {ww}</span></div>', unsafe_allow_html=True)
            im_show      = simpleWindow(im1, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = im1

        _section("Images")
        _show_images_only(im, im_show, method_name)

        _section("Frequency Domain")
        _show_freq_domain(im, fh, FH, sText, Filter, "Butterworth")

        _download_section(im, im_show, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_Exp:
        _section("Filter Parameters")
        Filter = st.selectbox("Sub-type", ["Low Pass", "High Pass", "Band Pass", "Band Reject"], key="exp_sub")
        TYPE   = {"Low Pass": 1, "High Pass": 2, "Band Pass": 4, "Band Reject": 3}[Filter]

        c1, c2 = st.columns(2)
        with c1:
            fco     = st.slider("Cutoff frequency", 1, Flength // 2, Flength // 4, key="exp_fco")
            ndegree = st.slider("Order",             1, 5,            2,             key="exp_deg")
        with c2:
            trans   = st.slider("Transition",        1, Flength // 2, Flength // 3, key="exp_trans")

        fh, sText   = Exponential(Flength, ndegree, fco, TYPE, trans)
        params_text = f"Exponential {Filter}, fco={fco}, order={ndegree}, trans={trans}"
        method_name = f"Exponential · {Filter}"

        FH  = design2dFilter(im, fh)
        im1 = filterImage(im, FH, tones)
        im1 = imNormalize(im1, tones)

        apply_window = st.checkbox("Apply Windowing", key="exp_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="exp_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="exp_ww")
            st.markdown(f'<div class="ch1-win-params"><span class="ch1-wp">WC = {wc}</span><span class="ch1-wp">WW = {ww}</span></div>', unsafe_allow_html=True)
            im_show      = simpleWindow(im1, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = im1

        _section("Images")
        _show_images_only(im, im_show, method_name)

        _section("Frequency Domain")
        _show_freq_domain(im, fh, FH, sText, Filter, "Exponential")

        _download_section(im, im_show, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_Gauss:
        _section("Filter Parameters")
        Filter = st.selectbox("Sub-type", ["Low Pass", "High Pass", "Band Pass", "Band Reject"], key="gauss_sub")
        TYPE   = {"Low Pass": 1, "High Pass": 2, "Band Pass": 4, "Band Reject": 3}[Filter]

        c1, c2 = st.columns(2)
        with c1:
            fco     = st.slider("Cutoff frequency", 1, Flength // 2, Flength // 4, key="gauss_fco")
            ndegree = st.slider("Order",             1, 5,            2,             key="gauss_deg")
        with c2:
            trans   = st.slider("Transition",        1, Flength // 2, Flength // 3, key="gauss_trans")

        fh, sText   = Gaussian(Flength, ndegree, fco, TYPE, trans)
        params_text = f"Gaussian {Filter}, fco={fco}, order={ndegree}, trans={trans}"
        method_name = f"Gaussian · {Filter}"

        FH  = design2dFilter(im, fh)
        im1 = filterImage(im, FH, tones)
        im1 = imNormalize(im1, tones)

        apply_window = st.checkbox("Apply Windowing", key="gauss_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="gauss_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="gauss_ww")
            st.markdown(f'<div class="ch1-win-params"><span class="ch1-wp">WC = {wc}</span><span class="ch1-wp">WW = {ww}</span></div>', unsafe_allow_html=True)
            im_show      = simpleWindow(im1, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = im1

        _section("Images")
        _show_images_only(im, im_show, method_name)

        _section("Frequency Domain")
        _show_freq_domain(im, fh, FH, sText, Filter, "Gaussian")

        _download_section(im, im_show, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_Ideal:
        _section("Filter Parameters")
        Filter = st.selectbox("Sub-type", ["Low Pass", "High Pass", "Band Pass", "Band Reject"], key="ideal_sub")
        TYPE   = {"Low Pass": 1, "High Pass": 2, "Band Pass": 4, "Band Reject": 3}[Filter]

        c1, c2 = st.columns(2)
        with c1:
            fco     = st.slider("Cutoff frequency", 1, Flength // 2, Flength // 4, key="ideal_fco")
            ndegree = st.slider("Order",             1, 5,            2,             key="ideal_deg")
        with c2:
            trans   = st.slider("Transition",        1, Flength // 2, Flength // 3, key="ideal_trans")
            w_bw    = st.slider("Bandwidth (w)",     1, Flength // 2, Flength // 4, key="ideal_w")

        fh, sText   = Ideal(Flength, fco, TYPE, 0.0, trans, w_bw)
        params_text = f"Ideal {Filter}, fco={fco}, trans={trans}, w={w_bw}"
        method_name = f"Ideal · {Filter}"

        FH  = design2dFilter(im, fh)
        im1 = filterImage(im, FH, tones)
        im1 = imNormalize(im1, tones)

        apply_window = st.checkbox("Apply Windowing", key="ideal_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="ideal_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="ideal_ww")
            st.markdown(f'<div class="ch1-win-params"><span class="ch1-wp">WC = {wc}</span><span class="ch1-wp">WW = {ww}</span></div>', unsafe_allow_html=True)
            im_show      = simpleWindow(im1, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = im1

        _section("Images")
        _show_images_only(im, im_show, method_name)

        _section("Frequency Domain")
        _show_freq_domain(im, fh, FH, sText, Filter, "Ideal")

        _download_section(im, im_show, method_name, params_text)

    # ══════════════════════════════════════════
    with tab_Wien:
        _section("Filter Parameters")
        Filter   = st.selectbox("Wiener variant", ["Inverse Filter", "Wiener Filter", "Power Filter"], key="wien_sub")
        TYPE_map = {"Inverse Filter": 1, "Wiener Filter": 2, "Power Filter": 3}
        TYPE     = TYPE_map[Filter]
        f_mtf    = GaussianMTF(Flength)
        fh, sText = generalizedWienerFilter(im, f_mtf, TYPE)
        params_text = f"Wiener · {Filter}"
        method_name = f"Wiener · {Filter}"

        FH  = design2dFilter(im, fh)
        im1 = DeconvolveImage(im, FH)
        im1 = imNormalize(im1, tones)

        apply_window = st.checkbox("Apply Windowing", key="wien_win")
        if apply_window:
            cc1, cc2 = st.columns(2)
            with cc1: wc = st.slider("Window Center", 0, 255, 128, key="wien_wc")
            with cc2: ww = st.slider("Window Width",  1, 255, 100, key="wien_ww")
            st.markdown(f'<div class="ch1-win-params"><span class="ch1-wp">WC = {wc}</span><span class="ch1-wp">WW = {ww}</span></div>', unsafe_allow_html=True)
            im_show      = simpleWindow(im1, wc, ww, image_depth, tones)
            params_text += f", WC={wc}, WW={ww}"
        else:
            im_show = im1

        _section("Images")
        _show_images_only(im, im_show, method_name)

        _section("Frequency Domain")
        _show_freq_domain(im, fh, FH, sText, Filter, "Wiener")

        _download_section(im, im_show, method_name, params_text)