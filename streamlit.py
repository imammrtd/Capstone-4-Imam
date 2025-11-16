import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# ------------------------------
# Load YOLO model
# ------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # pastikan best.pt berada di folder yang sama

model = load_model()

# ------------------------------
# UI / Layout
# ------------------------------
st.title("Construction Safety Equipment Detection")
st.write("Aplikasi ini mendeteksi peralatan keselamatan kerja seperti helm, rompi, dan pekerja pada sebuah gambar menggunakan model YOLOv8.")

uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])


# ------------------------------
# Jika user upload gambar
# ------------------------------
if uploaded_file is not None:

    # Baca gambar menggunakan PIL 
    image = Image.open(uploaded_file).convert("RGB") # Pastikan selalu RGB

    # Convert ke format numpy 
    img_array = np.array(image)

    # Predict dengan Feedback Spinner
    with st.spinner("⏳ Sedang memproses deteksi..."):
        results = model.predict(img_array)

    # Visualisasi bounding box
    result_img = results[0].plot()

    st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    # ------------------------------
    # Hitung jumlah objek
    # ------------------------------
    boxes = results[0].boxes  
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    counts = {name: 0 for name in class_names.values()}

    for cid in class_ids:
        counts[class_names[cid]] += 1

    # Tampilkan hasil hitung
    st.subheader("Jumlah Deteksi:")
    
    # Konversi ke DataFrame
    counts_df = pd.DataFrame(
        list(counts.items()), 
        columns=['Objek', 'Jumlah']
    )
    
    # Tampilkan sebagai tabel
    st.table(counts_df)

    # ------------------------------
    # Kesimpulan Safety
    # ------------------------------
    total_person = counts.get("person", 0)
    no_helmet = counts.get("no-helmet", 0)
    no_vest = counts.get("no-vest", 0)

    st.subheader("Analisis Keselamatan:")
    if total_person == 0:
        st.warning("Tidak ada pekerja yang terdeteksi.")
    else:
        st.write(f"Total pekerja terdeteksi: **{total_person}**")
        st.write(f"Pekerja tanpa helm: **{no_helmet}**")
        st.write(f"Pekerja tanpa rompi: **{no_vest}**")

        if no_helmet > 0 or no_vest > 0:
            st.error("⚠ Terdapat pekerja yang tidak menggunakan alat keselamatan!")
        else:
            st.success("✔ Semua pekerja terdeteksi aman.")
