import streamlit as st
import torch
import pandas as pd
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

## CFG
#cfg_model_path = "yolov5m6-lite.pt" 
cfg_model_path = "best.pt" 


def imageInput(device, src):
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
            pred = model(imgpath)
            df = pred.pandas().xyxy[0]
            jenis_roti = []
            availability = []
            dc = False
            lb = False
            rg = False
            kj = False
            tk = False
            rt = False
            sc = False
            sk = False
            ck = False
            cs = False
            #looping for print available type of breead
            for name in df['name']:
                if name == 'dorayaki cokelat':
                    if dc == False:
                        jenis_roti.append('Dorayaki Coklat')
                        availability.append('Available')
                        st.write('Dorayaki coklat available')
                        dc = True
                elif name == 'lapis bamkuhen':
                    if lb == False:
                        jenis_roti.append('Lapis Bamkuhen')
                        availability.append('Available')
                        st.write('lapis bamkuhen available')
                        lb = True
                elif name == 'roti gandum':
                    if rg == False:
                        jenis_roti.append('Roti Gandum')
                        availability.append('Available')
                        st.write('roti gandum available')
                        rg = True
                elif name == 'roti krim keju':
                    if kj == False:
                        jenis_roti.append('Roti Krim Keju')
                        availability.append('Available')
                        st.write('roti krim keju available')
                        kj = True
                elif name == 'roti tawar kupas':
                    if tk == False:
                        jenis_roti.append('Roti Tawar Kupas')
                        availability.append('Available')
                        st.write('roti tawar kupas available')
                        tk = True
                elif name == 'roti tawar':
                    if rt == False:
                        jenis_roti.append('Roti tawar')
                        availability.append('Available')
                        st.write('roti tawar available')
                        rt = True
                elif name == 'sandwich cokelat':
                    if sc == False:
                        jenis_roti.append('Sandwich Coklat')
                        availability.append('Available')
                        st.write('sandwich cokelat available')
                        sc = True
                elif name == 'sandwich keju':
                    if sk == False:
                        jenis_roti.append('Sandwich Keju')
                        availability.append('Available')
                        st.write('sandwich keju available')
                        sk = True
                elif name == 'sobek cokelat keju':
                    if ck == False:
                        jenis_roti.append('Sobek Coklat Keju')
                        availability.append('Available')
                        st.write('sobek cokelat keju available')
                        ck = True
                else :
                    if cs == False:
                        jenis_roti.append('Sobek Coklat Sarikaya')
                        availability.append('Available')
                        st.write('sobek cokelat sarikaya available')
                        cs = True

            #Not Available check
            if dc == False:
                jenis_roti.append('Dorayaki Coklat')
                availability.append('Not Available')
            if lb == False:
                jenis_roti.append('Lapis Bamkuhen')
                availability.append('Not Available')
            if rg == False:
                jenis_roti.append('Roti Gandum')
                availability.append('Not Available')
            if kj == False:
                jenis_roti.append('Roti Krim Keju')
                availability.append('Not Available')
            if tk == False:
                jenis_roti.append('Roti Tawar Kupas')
                availability.append('Not Available')
            if rt == False:
                jenis_roti.append('Roti tawar')
                availability.append('Not Available')
            if sc == False:
                jenis_roti.append('Sandwich Coklat')
                availability.append('Not Available')
            if sk == False:
                jenis_roti.append('Sandwich Keju')
                availability.append('Not Available')
            if ck == False:
                jenis_roti.append('Sobek Coklat Keju')
                availability.append('Not Available')
            if cs == False:
                jenis_roti.append('Sobek Coklat Sarikaya')
                availability.append('Not Available')

            # create dataframe from your lists and make it to csv
            daframe = pd.DataFrame(list(zip(jenis_roti , availability)), 
                columns =['Type of Bread', 'Availability'])
            daframe.to_csv(os.path.basename(imgpath)+('.csv'))
            
            #download the csv files
            with open(os.path.basename(imgpath)+'.csv') as f:
                st.download_button('Download CSV', f, os.path.basename(imgpath)+'.csv') 
            
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)
            
            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data.'])

    option = st.sidebar.radio("Select input type.", ['Image'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('🍞Various Bread Types Recognition🍞')
    st.subheader('👈🏽 Select options left-haned menu bar.')
    st.sidebar.markdown("@THE F4NTASTIC BOB")
    if option == "Image":    
        imageInput(deviceoption, datasrc)

    
if __name__ == '__main__':
  
    main()