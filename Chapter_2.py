import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pydicom as dicom
import io

# -----------------------------
def prepare_image(x):
    x = x.astype(float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def conv2(im,mask):
    im1=signal.convolve2d(im,mask,mode='same')
    return im1


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def simpleWindow(im,wc,ww,image_depth, tones):
    im1=np.asarray(im, dtype=float)
    Vb=(2.0*wc+ww)/2.0
    Va=Vb-ww 
    if(Vb>image_depth):
        Vb=image_depth
    if(Va<0): 
        Va=0
    im1=(((tones-1)*((im1-Va)/(Vb-Va))))
    M=np.size(im1,0)
    N=np.size(im1,1)
    for i in range (0,M):
        for j in range(N):
            if( (im1[i][j]>=Va) and (im1[i][j]<=Vb) ):    
                im1[i][j]=(((tones-1)*(im1[i][j]-Va)/(Vb-Va)))
            elif (im1[i][j]<Va):
                im1[i][j]=0
            elif (im1[i][j]>Vb):
                im1[i][j]=tones-1
    return im1


def imPlot(im,title,fz):
    im = Image.fromarray(np.asarray(im, dtype="uint8"), "L")
    plt.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=255)
    plt.title(title)
    plt.axis("off")
    
def sigNorm(x):
    xmax=np.max(x);xmin=np.min(x);
    x=(x-xmin)/(xmax-xmin)
    return(x)

def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn); 
    w=np.round(w)
    return w


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

# -----------------------------
def run():
   
     
     
    uploaded = st.session_state.get("uploaded_file", None)
    
    if uploaded is None:
      st.warning("Upload an image from the Dashboard.")
      st.stop()
       

    if uploaded is not None:
     im=loadImage(uploaded)
     im=np.asanyarray(im,dtype=float)
     im=sigNorm(im)
     if (len(im.shape))==3:
        im=rgb2gray(im)
     
     
     
     
     im_depth=255
     tones=256
     
     
     if len(np.shape(im))==3:
         im=rgb2gray(im)
     
     
     im=np.asarray(im,dtype=float)
     
     
     method = st.radio(
    "Select filtering type",
    ["Convolution filters", "Median filter"]
)
     
     
     
     if method=="Convolution filters":
         Choice = st.selectbox( 
         "Choose filter",
    ["Smoothing", "Laplacian", "High-emphasis"])
         kernel=np.zeros((3,3),dtype=float)
     
     
         if Choice=="Smoothing":
             kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
             title='shoothing mask'
             sK=np.sum(kernel)
             if(sK>0):
                 kernel=kernel/sK
             im1=conv2(im,kernel)
     
     
         elif Choice=="Laplacian":
             kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
             title='laplacian mask'
             sK=np.sum(kernel)
             if(sK>0):
                 kernel=kernel/sK
             im1=conv2(im,kernel)
     
     
         elif Choice=="High-emphasis":
             kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
             title='high-emphasis'
             sK=np.sum(kernel)
             if(sK>0):
                 kernel=kernel/sK
             im1=conv2(im,kernel)
     
     
         else:
             st.write("Invalid choice")
             st.stop()
     
     
         
     
     
     elif method=="Median filter":
         im1=signal.medfilt2d(im,(3,3))
         title='median filter'
     
     
     else:
         st.write("Invalid choice")
         st.stop()
     im1=imNormalize(im1,tones)
     apply_window = st.checkbox("Apply Windowing to processed image") 
     if apply_window:
          wc = st.slider("Window Center", 0, 255, 128)
          ww = st.slider("Window Width", 1, 255, 100)
          im_show = simpleWindow(im1, wc, ww, im_depth, tones)
     else:
          im_show = im1     
    
     col1, col2 = st.columns(2)

     with col1:
         st.image(prepare_image(im), caption="Original Image", width="stretch")
     
     with col2:
        st.image(prepare_image(im_show), caption=f"Convolved Image with {title}", width="stretch")
     
     
     
     
     
     
     
     

     
     
     
    
     
      

     Inorm=sigNorm(im_show)
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
     


