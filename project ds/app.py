import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL
st.set_page_config(page_title="Deteksi Obesitas", page_icon="")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_obesitas_rf.pkl')
        scaler = joblib.load('scaler_obesitas.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File .pkl tidak ditemukan! Pastikan 'model_obesitas_rf.pkl' dan 'scaler_obesitas.pkl' ada di folder yang sama.")
        return None, None

model, scaler = load_assets()

# Kita harus mengubah input teks user menjadi angka, persis seperti saat training.
map_gender = {'Perempuan': 0, 'Laki-laki': 1}
map_yes_no = {'Tidak': 0, 'Ya': 1}
map_caec = {'Selalu (Always)': 0, 'Sering (Frequently)': 1, 'Kadang-kadang (Sometimes)': 2, 'Tidak Pernah (no)': 3}
map_calc = {'Selalu (Always)': 0, 'Sering (Frequently)': 1, 'Kadang-kadang (Sometimes)': 2, 'Tidak Pernah (no)': 3}
map_mtrans = {'Mobil (Automobile)': 0, 'Sepeda (Bike)': 1, 'Motor (Motorbike)': 2, 'Transportasi Umum': 3, 'Jalan Kaki': 4}

label_hasil = {
    0: "Berat Badan Kurang (Insufficient Weight)",
    1: "Berat Badan Normal (Normal Weight)",
    2: "Obesitas Tipe I",
    3: "Obesitas Tipe II",
    4: "Obesitas Tipe III",
    5: "Kelebihan Berat Badan Level I",
    6: "Kelebihan Berat Badan Level II"
}

st.title("Sistem Deteksi Dini Tingkat Obesitas")
st.write("Aplikasi ini menggunakan **Random Forest (Akurasi 99%)** untuk memprediksi tingkat obesitas berdasarkan kebiasaan hidup.")

st.sidebar.header("PROJEK DATA SCIENCE")

with st.form("obesity_form"):
    st.subheader("1. Data Fisik")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Umur (Tahun)", min_value=10, max_value=80, value=25)
    with col2:
        height = st.number_input("Tinggi Badan (Meter)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, help="Gunakan titik untuk desimal, misal 1.70")
    with col3:
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
    
    gender = st.selectbox("Jenis Kelamin", options=['Laki-laki', 'Perempuan'])
    family_hist = st.selectbox("Ada riwayat keluarga obesitas?", options=['Tidak', 'Ya'])

    st.subheader("2. Pola Makan")
    col4, col5 = st.columns(2)
    with col4:
        fcvc = st.slider("Konsumsi Sayuran (FCVC)", 1.0, 3.0, 2.0, help="1: Jarang, 3: Selalu")
        ncp = st.slider("Jumlah Makan Utama per Hari (NCP)", 1.0, 4.0, 3.0)
        ch2o = st.slider("Konsumsi Air Minum (Liter/hari)", 1.0, 3.0, 2.0)
    with col5:
        favc = st.selectbox("Sering makan tinggi kalori? (FAVC)", options=['Tidak', 'Ya'])
        caec = st.selectbox("Seberapa sering ngemil? (CAEC)", options=list(map_caec.keys()))
        calc = st.selectbox("Konsumsi Alkohol? (CALC)", options=list(map_calc.keys()))

    st.subheader("3. Gaya Hidup & Aktivitas")
    col6, col7 = st.columns(2)
    with col6:
        smoke = st.selectbox("Apakah Anda Merokok? (SMOKE)", options=['Tidak', 'Ya'])
        scc = st.selectbox("Memantau asupan kalori? (SCC)", options=['Tidak', 'Ya'])
    with col7:
        faf = st.slider("Frekuensi Aktivitas Fisik (Hari/Minggu)", 0.0, 3.0, 1.0)
        tue = st.slider("Waktu penggunaan teknologi (Jam/hari)", 0.0, 2.0, 1.0)
        mtrans = st.selectbox("Transportasi Utama (MTRANS)", options=list(map_mtrans.keys()))

    submit = st.form_submit_button("🔍 Analisis Sekarang")

if submit and model is not None:
    bmi = weight / (height ** 2)
    
    row = [
        map_gender[gender], 
        age,                
        height,             
        weight,             
        map_yes_no[family_hist], 
        map_yes_no[favc],   
        fcvc,               
        ncp,                
        map_caec[caec],     
        map_yes_no[smoke],  
        ch2o,               
        map_yes_no[scc],    
        faf,                
        tue,                
        map_calc[calc],     
        map_mtrans[mtrans], 
        bmi                 
    ]
    
    columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
               'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
               'CALC', 'MTRANS', 'BMI']
    
    df_input = pd.DataFrame([row], columns=columns)
    
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # Prediksi
    prediction = model.predict(df_input)[0]
    probabilitas = model.predict_proba(df_input)
    confidence = np.max(probabilitas) * 100

    st.divider()
    st.subheader("📋 Hasil Analisis")
    
    st.info(f"Indeks Massa Tubuh (BMI) Anda: **{bmi:.2f}**")
    
    hasil_teks = label_hasil.get(prediction, "Tidak Diketahui")
    
    if prediction in [0, 1]:
        st.success(f"Status: **{hasil_teks}**")
    elif prediction in [5, 6]:
        st.warning(f"Status: **{hasil_teks}**")
    else:
        st.error(f"Status: **{hasil_teks}**")
        
    st.write(f"Tingkat Kepercayaan Model: **{confidence:.2f}%**")
    
    st.write("---")
    st.write("Detail Probabilitas per Kelas:")
    df_prob = pd.DataFrame({
        "Kategori": label_hasil.values(),
        "Probabilitas": probabilitas[0]
    })
    st.bar_chart(df_prob.set_index("Kategori"))