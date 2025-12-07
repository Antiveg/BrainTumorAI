import streamlit as st
from PIL import Image
import keras
from src.utils.predict import predict
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt

model_type = 'resnet50v2'
model_path = f'./saved/{model_type}/model.keras'
model = keras.models.load_model(model_path)
class_names = ['Glioma', 'Meningioma', 'Sehat', 'Kanker Pituitari']

details = {
    "Glioma": "Glioma adalah tumor yang terbentuk ketika sel glia tubuh Anda tumbuh tak terkendali. Sel glia menopang saraf dan membantu sistem saraf pusat bekerja. Jika tumbuh tak terkendali, sel-sel ini dapat membentuk tumor di otak atau sumsum tulang belakang. Glioma adalah tumor primer. Artinya, tumor ini terbentuk langsung di otak atau sumsum tulang belakang.",
    "Meningioma": "Meningioma adalah tumor yang berkembang dari selaput meninges, yaitu lapisan pelindung otak dan sumsum tulang belakang. Meskipun bukan tumor otak, ia sering dikategorikan demikian karena dapat menekan jaringan otak, saraf, atau pembuluh darah di sekitarnya. Sebagian besar meningioma bersifat jinak (non-kanker) dan tumbuh lambat, tetapi ukuran dan lokasinya bisa menyebabkan gejala yang parah dan mengancam jiwa jika tidak ditangani.",
    "Sehat": "Tidak ada indikasi otak mengidap tipe kanker otak glioma, meningioma atau pituitari. Hasil ini tidak menyatakan kalau otak sehat sepenuhnya karena itu tetap perlu klarifikasi lebih lanjut dengan dokter yang relevan.",
    "Kanker Pituitari": "Kanker otak pituitari disebut juga tumor hipofisis atau adenoma pituitari, adalah pertumbuhan sel abnormal di kelenjar pituitari yang menekan area sekitarnya dan mengganggu produksi hormon. Tumor ini sebagian besar bersifat jinak (bukan kanker), namun dapat menyebabkan gejala seperti sakit kepala, gangguan penglihatan, dan masalah hormonal, karena ukurannya yang semakin membesar atau aktivitasnya dalam menghasilkan hormon."
}

st.set_page_config(
    page_title="Brain Tumor Identifier",
    layout="wide"
)

st.title("Brain Tumor X-ray Image Classifier")

with st.container():

    image_uploader, prob_displayer = st.columns([2, 4])

    image = None
    with image_uploader:

        uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image")
        else:
            st.markdown(
                """
                <div style="
                    width: 100%;
                    aspect-ratio: 2;
                    border: 2px dashed #aaa;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #aaa;
                    font-weight: bold;
                    text-align: center;
                ">
                    No Image yet...
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <style>
                .stButton button {{
                    width: 160px;
                    margin-top: 5px;
                    padding: 12px 20px;
                    font-size: 18px;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                    cursor: pointer;
                    display: block;
                    background-color: {'#4CAF50' if image else '#808080'};
                    color: white;
                }}
                .stButton button:hover {{
                    background-color: {'#45a049' if image else '#666666'};
                }}
                .stButton button:disabled {{
                    background-color: #808080 !important;
                    cursor: not-allowed;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        predict_button = st.button("Predict", disabled=image is None)

    with prob_displayer:

        if uploaded_file is not None and predict_button:

            probs = predict(model, image, class_names)
            data = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
            data["Probability"] = data["Probability"] * 100
            best_class = max(probs, key=probs.get)

            st.markdown(
                f"""
                <div style="
                    background-color: #4e4e4e;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    color: #ffffff;
                ">
                    <h3 style="margin-bottom: 10px;"><b>X-ray Otak diprediksi paling memungkinkan berstatus {best_class}</b></h3>
                    <p style="font-size: 16px;">
                        {details[best_class]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.bar_chart(data.set_index("Class")["Probability"], horizontal=True)

            cols = st.columns(4)
        
            for idx, row in data.iterrows():
                color = '#66cc66' if row['Class'] == best_class else '#ff6666'
                
                cols[idx].markdown(
                    f"""
                    <div style='
                        display: inline-block;
                        width: 100%;
                        padding-left: 20px;
                        margin: 0;
                        border-left: 0.5px solid #f5f5f5;
                        font-size: 18px;
                    '>
                        <span style='font-weight: bold; color: {color};'>{row['Class']}</span><br>
                        <span style='color:{color}; font-size: 22px;'>{row['Probability']:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                """
                <div style="
                    background-color: #4e4e4e;
                    color: #ffffff;
                    padding: 30px; 
                    border-radius: 8px;
                    font-size: 16px;
                    line-height: 1.5;
                ">
                    <h3>Brain Tumor CT-Scan Classifier</h3>
                    <p>Ini adalah pengklasifikasi CT-scan tumor otak. Untuk setiap CT-scan atau X-ray otak yang diberikan, Anda dapat mengunggah gambar dan model akan mengklasifikasikan apakah kondisi otak tersebut termasuk dalam salah satu kategori berikut:</p>
                    <ul>
                        <li><strong>Glioma</strong></li>
                        <li><strong>Meningioma</strong></li>
                        <li><strong>Pituitary</strong></li>
                        <li><strong>Sehat (Tanpa Tumor)</strong></li>
                    </ul>
                    <p><strong>Cara menggunakan:</strong></p>
                    <ul>
                        <li>Unggah file <strong>.jpg</strong> atau <strong>.png</strong> melalui tombol "Browse Files" di sisi kiri.</li>
                        <li>Klik tombol "Predict" setelah berubah menjadi hijau (ini menandakan bahwa gambar sudah diunggah).</li>
                        <li>Tunggu hingga model AI memproses dan mengklasifikasikan gambar, dan hasilnya akan ditampilkan di sisi kanan.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )