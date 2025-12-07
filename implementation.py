import streamlit as st
from PIL import Image
import keras
from src.utils.predict import predict
import pandas as pd

class_details = {
    "Glioma": 
        "Glioma adalah tumor yang terbentuk ketika sel glia tubuh Anda tumbuh tak terkendali. Sel glia menopang saraf dan membantu sistem saraf pusat bekerja. Jika tumbuh tak terkendali, sel-sel ini dapat membentuk tumor di otak atau sumsum tulang belakang. Glioma adalah tumor primer. Artinya, tumor ini terbentuk langsung di otak atau sumsum tulang belakang.",

    "Meningioma": 
        "Meningioma adalah tumor yang berkembang dari selaput meninges, yaitu lapisan pelindung otak dan sumsum tulang belakang. Meskipun bukan tumor otak, ia sering dikategorikan demikian karena dapat menekan jaringan otak, saraf, atau pembuluh darah di sekitarnya. Sebagian besar meningioma bersifat jinak (non-kanker) dan tumbuh lambat, tetapi ukuran dan lokasinya bisa menyebabkan gejala yang parah dan mengancam jiwa jika tidak ditangani.",

    "Sehat": 
        "Tidak ada indikasi otak mengidap tipe kanker otak glioma, meningioma atau pituitari. Hasil ini tidak sepenuhnya menyatakan kalau otak sehat, karena masih ada probabilitas otak mengidap tipe kanker otak lain yang tidak termasuk dalam tipe yang dilatih dan ada probabilitas mis-klasifikasi. Diharapkan untuk klarifikasi lebih lanjut dengan dokter yang relevan.",

    "Kanker Pituitari": 
        "Kanker otak pituitari disebut juga tumor hipofisis atau adenoma pituitari, adalah pertumbuhan sel abnormal di kelenjar pituitari yang menekan area sekitarnya dan mengganggu produksi hormon. Tumor ini sebagian besar bersifat jinak (bukan kanker), namun dapat menyebabkan gejala seperti sakit kepala, gangguan penglihatan, dan masalah hormonal, karena ukurannya yang semakin membesar atau aktivitasnya dalam menghasilkan hormon."
}

model_types = {
    'resnet50v2': './saved/resnet50v2/model.keras',
    'vgg16': './saved/vgg16/model.keras',
    'mobilenetv2': './saved/mobilenetv2/model.keras'
}

@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

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
                    cursor: pointer;
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

        model_name = st.selectbox("Pilih Model", options=["vgg16", "resnet50v2", "mobilenetv2"], index=0)
        model = load_model(model_types[model_name])

        col1, col2 = st.columns(2)
        with col1:
            predict_button = st.button("Predict", disabled=image is None)
        with col2:
            sample_download_button = st.download_button(
                label="Download Sample",
                data=open('data/predict_sample.zip', "rb"),
                file_name="sample_images.zip",
                mime="application/zip"
            )

    with prob_displayer:

        if uploaded_file is not None and predict_button:

            probs = predict(model, image, list(class_details.keys()))
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
                        {class_details[best_class]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Distribusi Probabilitas setiap tipe kanker otak")
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