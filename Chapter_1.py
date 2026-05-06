import streamlit as st
import  numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pydicom as dicom

def simpleDisplay(im,image_depth,tones):
    im1=np.zeros(im.shape, dtype=float)
    im1=np.round(( (tones-1)/(image_depth-1) ) * (im-0))
    return im1


def optimalDisplay(im,tones):
    im1=np.zeros(im.shape, dtype=float) 
    vmn=int(np.min(im))
    vmx=int(np.max(im))
    im1=(((tones-1)*(im-vmn)/(vmx-vmn)))
    im1=np.around(im1)
    return im1


def simpleWindow(im,wc,ww,image_depth, tones):
    im1=np.zeros(im.shape, dtype=float)


    Vb=(2.0*wc+ww)/2.0;
    if(Vb>image_depth):
        Vb=image_depth;
    Va=Vb-ww;
    if(Va<0): 
        Va=0;
    M=np.size(im,0)#image rows
    N=np.size(im,1)#image columns        
    
    for i in range(M):
        for j in range(N):
            Vm=im[i][j]
            if(Vm<Va): 
                t=0
            elif(Vm>Vb):
                t=tones-1
            else: 
                t=(((tones-1)*(Vm-Va)/(Vb-Va)))
            im1[i][j]=np.round(t)
    return im1




#--------------------------------------------------------


def brokenWindow(im,image_depth,tones,gray_val,im_val):
    im=np.asarray(im,dtype=float)
    N=np.size(im,0)#image rows
    M=np.size(im,1)#image columns
    im1=np.zeros(im.shape, dtype=float)
    for i in range(N):
        for j in range(M):
            if(im[i][j]<=im_val):
                im1[i][j]=(gray_val/(im_val))*im[i][j]
            else:
                im1[i][j]= (  ((tones-1)-(gray_val+1))/(image_depth-(im_val+1)) )*(im[i][j]-(im_val+1)) + (gray_val+1);
    im1=np.round(im1)
    return im1

def doubleWindow(im,ww1,wl1,ww2,wl2,image_depth,tones):
    im=np.asarray(im,dtype=float)
    im1=np.zeros(im.shape, dtype=float)
    N=np.size(im,0)#image rows
    M=np.size(im,1)#image columns
    
    half= (tones/2)-1;
    ve1=round ( (2.0*wl1+ww1)/2.0 );
    vs1=ve1-ww1;
    ve2=round ( (2.0*wl2+ww2)/2.0 );
    vs2=ve2-ww2;
    if(vs2<ve1):
        new_point=round ( ((vs2+ve1))/2.0);
        ve1=new_point;
        vs2=ve1;
    
    if(vs1<0):
        vs1=0;
    if(ve2>image_depth):
        ve2=image_depth;
    
    for i in range (N):
        for j in range (M):
            if (im[i][j]<vs1) :
                im1[i][j]=0; 
            if ( im[i][j] >= vs1 and im[i][j]<=ve1 ):
                t= round ( (  (half-0)/(ve1-vs1) )*(im[i][j]-vs1) + 0.0);
                im1[i][j]=round(t);
            
            if ( im[i][j]>ve1 and im[i][j]<vs2 ):
                im1[i][j]=half+1;
            
            if ( im[i][j]>=vs2 and im[i][j]<=ve2 ):
                t= round ( (  ( (tones-1)-(half+1) )/(ve2-vs2) )*(im[i][j]-vs2) + (half+1));
                im1[i][j]=round(t);
            
            if (im[i][j]>ve2):
                im1[i][j]=tones-1;
    return im1


def RGB2GRAY(im):
    # return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    Rim=im[:,:,0];Gim=im[:,:,1];Bim=im[:,:,2];
    gray = 0.2989*Rim + 0.5870*Gim + 0.1140*Bim
    return (gray)

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

def formPlotFunction(tones,Choice):
    import math
    w=np.zeros(tones)
    # Choice=4
    for i in range (0,tones):
        if (Choice==0):#inverse
            w[i]=tones-i-1
            text='inverse'
        elif(Choice==1):#logarithmic
            r=0.05
            w[i]=math.log(1+r*i)
            text='logarithmic'
        elif(Choice==2):#inverse logarithmic
            c=128
            w[i]=np.exp(i)**(1/c)-1
            text='inverse logarithmic'
        elif(Choice==3):#power
            gamma=0.55    
            w[i]=i**gamma
            text='power'
        elif(Choice==4):
            w[i]=np.sin(2*np.pi*i/(4*(tones-1)))
            text='sine-window'
        elif(Choice==5):#exponential
            w[i]=1-np.exp(-i/90)
            text='exp-window'
        elif (Choice==6):
            w[i]=(1/(1+(np.exp(-i/70)))) 
            text='sigmoid'
        elif (Choice==7):
            w[i]=np.cos(2*np.pi*i/(4*(tones-1)))
            text='cos-window'
        elif (Choice==8):
            w[i] = 1 / np.sqrt(max(i, 1e-6))
            text='reciprocal-square-root-function'
                
    w=(tones-1)*((w-np.min(w))/(np.max(w)-np.min(w)))        
            
    # print('wMin: %d, wMax: %d: ' %(np.min(w),np.max(w)))    
    # print('display function '+text+': \n',w)
    # U.RETURN()
    return(w,text)


def sigNorm(x):
    xmax=np.max(x);xmin=np.min(x);
    x=(x-xmin)/(xmax-xmin)
    return(x)

import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv




def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn);    
    w=np.round(w)
    return w


def f_histogram(A,image_depth,tones):
    minA=0;maxA=image_depth;
    if(np.max(A)>(tones-1)):
        B=np.round((tones-1)*((A-minA)/(maxA-minA)))
    else:
        B=A
    #calculate histogram
    M=np.size(B,0)
    N=np.size(B,1)
    Bval=np.reshape(B,M*N)
    h=np.zeros(tones,dtype=float)
    for i in range(np.size(Bval)):
        val=np.int16(Bval[i])
        h[val]=h[val]+1;
    return (h)


def f_hequalization (A,image_depth,tones):
    #implement HE method
    minA=0;maxA=image_depth;
    B=np.round((tones-1)*((A-minA)/(maxA-minA) ))
    M=np.size(B,0);N=np.size(B,1)
    Bval=np.reshape(B,M*N)#from 2d to 1d dimension
    p=np.argsort(Bval)#keep initial positions of sorted values 
    neq=np.int32((M*N)/tones+0.5) #calculate number of pixels per gray-tone
    BL=len(Bval)
    az=np.int32(np.fix((N*M)/neq))#get integral number of pixels per gray l
    zRem=np.int32(np.remainder(BL,neq))#get  remainder of pixels if not int
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


def CDF_equalization_formula (im,image_depth,tones):
    B=np.round((tones-1)*((im)/(np.max(im)) ))
    M=np.size(im,0);N=np.size(im,1)
    CDFh=np.zeros(tones,dtype=float)
    CDFq=np.zeros(tones,dtype=float)
    h=f_histogram(im,image_depth,tones)
    tone_values=((M*N)/tones)
    q=(tone_values*np.ones(tones,dtype=float));
    for i in range (tones):
        for j in range (i+1):
            CDFh[i]=CDFh[i]+h[j];
            CDFq[i]=CDFq[i]+q[j];
    B=CDFh[np.int32(B)]/tone_values-1
    B=np.round(B)
    return B


def CLAHE_method(im,im_depth,clip_limit):
    im_uint8 = np.uint8(im)
    im_CLAHE = exposure.equalize_adapthist(im_uint8, clip_limit=clip_limit)
    
    
    im_CLAHE = (im_depth-1) * im_CLAHE / np.max(im_CLAHE)
    if im_depth <= 8:
        im_CLAHE = np.uint8(np.round(im_CLAHE))
    elif im_depth <= 16:
        im_CLAHE = np.uint16(np.round(im_CLAHE))
    else:
        im_CLAHE = np.round(im_CLAHE)
    return im_CLAHE







def prepare_image(x):
    x = x.astype(float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def run():


    uploaded = st.session_state.get("uploaded_file", None)

    if uploaded is None:
        st.warning("Upload an image from the Dashboard.")
        st.stop()

    im = loadImage(uploaded)
    im = np.asarray(im, dtype=float)

    tones = 256
    image_depth = 256

    # ---------------- SINGLE METHOD SELECTOR ----------------
    method = st.selectbox(
        "Choose processing method",
        [
            "Simple Display",
            "Optimal Display",
            "Simple Window",
            "Broken Window",
            "Double Window",
            "Non-linear Mapping",
            "Histogram Equalization",
            "CDF Equalization",
            "CLAHE"
        ]
    )

    # ---------------- PROCESSING ----------------
    if method == "Simple Display":
        im2 = simpleDisplay(im, image_depth, tones)

    elif method == "Optimal Display":
        im2 = optimalDisplay(im, tones)

    elif method == "Simple Window":
        wc = st.slider("Window Center", 0, 255, 128)
        ww = st.slider("Window Width", 1, 255, 100)
        im2 = simpleWindow(im, wc, ww, image_depth, tones)

    elif method == "Broken Window":
        gray_val = st.slider("Gray value split", 0, 255, 128)
        im_val = st.slider("Intensity split", 0, 255, 128)
        im2 = brokenWindow(im, image_depth, tones, gray_val, im_val)

    elif method == "Double Window":
        wl1 = st.slider("WL1", 0, 255, 80)
        ww1 = st.slider("WW1", 1, 255, 60)
        wl2 = st.slider("WL2", 0, 255, 160)
        ww2 = st.slider("WW2", 1, 255, 60)
        im2 = doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones)

    elif method == "Non-linear Mapping":
        option = st.selectbox(
            "Function",
            [
                "Inverse","Logarithmic","Inverse Log","Power",
                "Sine","Exponential","Sigmoid","Cosine","Reciprocal sqrt"
            ]
        )

        mapping_dict = {
            "Inverse": 0, "Logarithmic": 1, "Inverse Log": 2,
            "Power": 3, "Sine": 4, "Exponential": 5,
            "Sigmoid": 6, "Cosine": 7, "Reciprocal sqrt": 8
        }

        c = mapping_dict[option]
        (w,stext)=formPlotFunction(tones,c)
        N=np.size(im,0)
        M=np.size(im,1)
        mn=np.min(im)
        mx=np.max(im)
        im2=np.round((tones-1)*(im-mn)/(mx-mn))
        for i in range(N):
            for j in range(M):
                v=int(im2[i,j])
                im2[i,j]=np.int32(w[v])
        

    elif method == "Histogram Equalization":
        import cv2 as cv
        im2=f_hequalization(im,image_depth,tones)

    elif method == "CDF Equalization":
        im2=CDF_equalization_formula(im,image_depth,tones)

    elif method == "CLAHE":
        clip = st.slider("Clip limit", 1.0, 5.0, 2.0)
        im2=CLAHE_method(im,image_depth,clip)

    else:
        st.error("Invalid method")
        st.stop()

    # ---------------- DISPLAY ----------------
    if method in [
        "Simple Display",
        "Optimal Display"
        "Simple Window",
        "Broken Window",
        "Double Window",]:
    
        col1, col2 = st.columns(2)
        with col1:
            st.image(prepare_image(im), caption="Original", width="stretch")
        with col2:
            st.image(prepare_image(im2), caption=method, width="stretch")
        
    
    
    
    elif method== "Non-linear Mapping":

        col1, col2 = st.columns(2)
    
        with col1:
            st.image(prepare_image(im), caption="Original", width="stretch")
    
        with col2:
            st.image(prepare_image(im2), caption=method, width="stretch")
        
       
        
    
        fig, ax = plt.subplots(figsize=(3,2))
              
        ax.plot(w)
              
        
        ax.set_title('display method: '+ option)
        ax.set_xlabel('image-matrix values')
        ax.set_ylabel('gray-tone values')
        ax.axis("off")
              
        
        st.pyplot(fig, use_container_width=False)

# --- HISTOGRAM METHODS ---
    elif method in [
        "Histogram Equalization",
        "CDF Equalization",
        "CLAHE"
    ]:
        h1=f_histogram(im,image_depth,tones)
        h2=f_histogram(im2,image_depth,tones)
        col1, col2 = st.columns(2)
    
        with col1:
            st.image(prepare_image(im), caption="Original", width="stretch")
    
        with col2:
            st.image(prepare_image(im2), caption=method, width="stretch")
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig1 = plt.figure()
            plt.stem(h1)
            plt.title("Initial histogram")
            st.pyplot(fig1)
    
        with col4:
            fig2 = plt.figure()
            plt.stem(h2)
            plt.title(f"Histogram ({method})")
            st.pyplot(fig2)
    Inorm=sigNorm(im2)
    im1_uint8 = (Inorm*255).astype(np.uint8)
    buf = io.BytesIO()
    img = Image.fromarray(im1_uint8)
    img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
label="Download Processed Image",
data=buf,
file_name="processed.png",
mime="image/png")
     





