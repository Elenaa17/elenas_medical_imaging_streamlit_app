
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



























def run():
    st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #00bcd4;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #00bcd4;
        color: #121212;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0097a7;
        color: white;
    }
    .stSelectbox > div > div {
        background-color: #263238;
        border-radius: 6px;
        color: white;
    }
    .stSlider > div > div {
        color: #00bcd4;
    }
    .stCheckbox > label {
        color: #e0e0e0;
        font-weight: 500;
    }
    img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    
    
    uploaded = st.session_state.get("uploaded_file", None)
    
    if uploaded is None:
      st.warning("Upload an image from the Dashboard.")
      st.stop()
    
    if uploaded is not None:
        im = loadImage(uploaded)
        im = np.asarray(im, dtype=float)
        im = imNormalize(im, 256)
        tones=256
        image_depth=256
    
        Choice = st.selectbox("Filter type", ["Butterworth","Exponential","Gaussian","Ideal","Wiener"])
    
    
        M,N = im.shape
        Flength = int(np.sqrt(M*M + N*N))
    
    
        if Choice == "Butterworth":
            Filter = st.selectbox("Filter type", ["Low Pass","High Pass","Band Pass","Band Reject"])
            TYPE_map = {
              "Low Pass": 1,
              "High Pass": 2,
              "Band Pass": 4,
              "Band Reject": 3
               }
     
            TYPE = TYPE_map[Filter]
            fco = st.slider("Cutoff", 1, Flength//2, Flength//4)
            ndegree = st.slider("Order", 1, 5, 2)
            trans = st.slider("Transition", 1, Flength//2, Flength//3)
            fh, sText = Butterworth(Flength,ndegree, fco, TYPE, trans)
    
    
        elif Choice == "Exponential":
            Filter = st.selectbox("Filter type", ["Low Pass","High Pass","Band Pass","Band Reject"])
            TYPE_map = {
              "Low Pass": 1,
              "High Pass": 2,
              "Band Pass": 4,
              "Band Reject": 3
               }
     
            TYPE = TYPE_map[Filter]
            fco = st.slider("Cutoff", 1, Flength//2, Flength//4)
            ndegree = st.slider("Order", 1, 5, 2)
            trans = st.slider("Transition", 1, Flength//2, Flength//3)
            
            fh, sText = Exponential(Flength,ndegree, fco, TYPE, trans)
    
    
        elif Choice == "Gaussian":
            Filter = st.selectbox("Filter type", ["Low Pass","High Pass","Band Pass","Band Reject"])
            TYPE_map = {
              "Low Pass": 1,
              "High Pass": 2,
              "Band Pass": 4,
              "Band Reject": 3
               }
     
            TYPE = TYPE_map[Filter]
            fco = st.slider("Cutoff", 1, Flength//2, Flength//4)
            ndegree = st.slider("Order", 1, 5, 2)
            trans = st.slider("Transition", 1, Flength//2, Flength//3)
            
            fh, sText = Gaussian(Flength,ndegree, fco, TYPE, trans)
    
    
        elif Choice == "Ideal":
           Filter = st.selectbox("Filter type", ["Low Pass","High Pass","Band Pass","Band Reject"])
           TYPE_map = {
              "Low Pass": 1,
              "High Pass": 2,
              "Band Pass": 4,
              "Band Reject": 3
               }
     
           TYPE = TYPE_map[Filter]
           fco = st.slider("Cutoff", 1, Flength//2, Flength//4)
           ndegree = st.slider("Order", 1, 5, 2)
           trans = st.slider("Transition", 1, Flength//2, Flength//3)
           w = st.slider("Bandwidth (w)", 1, Flength//2, Flength//4)
    
    
        
           fh, sText = Ideal(Flength,fco,TYPE,0.0,trans,w)
    
    
        elif Choice=="Wiener":
            
            Filter  = st.selectbox("Choose filter type",["Inverse Filter", "Wiener Filter", "Power Filter"])
            TYPE_map = {
              "Inverse Filter": 1,
              "Wiener Filter": 2,
              "Power Filter": 3,
               }
            TYPE = TYPE_map[Filter]
            f = GaussianMTF(Flength)
            fh, sText = generalizedWienerFilter(im, f,TYPE)
    
    
        if Choice in ["Butterworth","Exponential","Gaussian","Ideal"]:
          FH = design2dFilter(im, fh)
          im1 = filterImage(im, FH, tones)
          im1 = imNormalize(im1, tones)
    
    
        elif Choice == "Wiener" :
           FH = design2dFilter(im, fh)
           im1 = DeconvolveImage(im, FH)
           im1 = imNormalize(im1, tones)
           
        apply_window = st.checkbox("Apply Windowing to processed image") 
        if apply_window:
             wc = st.slider("Window Center", 0, 255, 128)
             ww = st.slider("Window Width", 1, 255, 100)
             im_show = simpleWindow(im1, wc, ww, image_depth, tones)
        else:
             im_show = im1     
           
           
        
        # ---------------- TOP ROW ----------------
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(prepare_image(im), caption="Original Image", width="stretch")
        
        with col2:
            st.image(prepare_image(im_show), caption="Filtered Image", width="stretch")
        
        
        # ---------------- BOTTOM ROW ----------------
        col3, col4 = st.columns(2)
        
        with col3:
            fig1 = plt.figure()
            N=len(fh)
            N2_nor=[int(N/2)]
            plt.plot([N2_nor,N2_nor],[0,1],'b--',lw=3);plt.plot(np.fft.fftshift(fh),'r--')
            plt.legend(('filter','axis of symmetry','filter shifted to N/2'), loc='best')
            plt.plot(fh)
            plt.xlabel ('spatial frequencies');plt.ylabel('amplitude')
            plt.title("1-d filter: "+sText)
            plt.grid()
            st.pyplot(fig1)
        
        with col4:
            fig2 = plt.figure()
            plt.imshow(np.fft.fftshift(FH), cmap='gray')
            plt.title("2D "+ Choice+" Filter")
            plt.axis("off")
            st.pyplot(fig2)
            
        
        im1_uint8 = im_show.astype(np.uint8)
        
        buf = io.BytesIO()
        img = Image.fromarray(im1_uint8)
        img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
        label="Download Processed Image",
        data=buf,
        file_name="processed.png",
        mime="image/png"
    )
      