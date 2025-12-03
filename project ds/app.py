import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os

# load model
st.set_page_config(page_title="Deteksi Obesitas", page_icon="", layout="centered")

# load css
def load_css(file, bg_img=None):
    # mastiin file
    if not os.path.exists(file):
        st.error(f"File CSS tidak ditemukan: {file}")
        return
        
    with open(file, "r") as f:
        css = f.read()
    
    # buat bg
    if bg_img and os.path.exists(bg_img):
        with open(bg_img, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()
        css = css.replace("BG_IMAGE", b64)
    elif bg_img:
        st.warning(f"Gambar latar belakang tidak ditemukan: {bg_img}. Background akan kosong.")
        css = css.replace("BG_IMAGE", "")
        
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# memastikan path file selalu benar
DIR = os.path.dirname(__file__)
try:
    load_css(os.path.join(DIR, "style.css"), os.path.join(DIR, "image/lucu.png"))
except Exception as e:
    st.error(f"Gagal memuat aset CSS/Gambar: {e}")


def render_header(text, with_icon=True):
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:8px;'>{text}</h1>",
        unsafe_allow_html=True
    )

    if with_icon:
        try:
            # buat icon
            with open(os.path.join(DIR, "image/icon.jpg"), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style='text-align:center; margin-bottom:15px;'>
                    <img src='data:image/png;base64,{b64}' width='170' height='150' style='border-radius:50%;' />
                </div>
                """,
                unsafe_allow_html=True
            )
        except:
            pass

# model dan mapping

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(DIR, 'model_obesitas_rf.pkl'))
        scaler = joblib.load(os.path.join(DIR, 'scaler_obesitas.pkl'))
        return model, scaler
    except FileNotFoundError:
        st.error("File model/scaler (.pkl) tidak ditemukan! Pastikan 'model_obesitas_rf.pkl' dan 'scaler_obesitas.pkl' ada di folder yang sama.")
        return None, None

model, scaler = load_assets()

# mapping
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

# page routing dan logic

if "page" not in st.session_state:
    st.session_state.page = "start"

# Halaman 1: Start Page
def start_page():
    st.markdown("<div class='card-flag'></div>", unsafe_allow_html=True)
    render_header("Sistem Deteksi Dini Tingkat Obesitas", with_icon=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>Aplikasi ini menggunakan <strong>Random Forest (Akurasi 99%)</strong> untuk memprediksi tingkat obesitas berdasarkan kebiasaan hidup.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
    
    if st.button("Mulai Deteksi"):
        st.session_state.page = "main"
        if 'results' in st.session_state:
            del st.session_state.results
        st.rerun()

# Halaman 2: Main Input Page (Formulir)
def main_page():
    st.markdown("<div class='card-flag'></div>", unsafe_allow_html=True)
    render_header("Form Deteksi Obesitas", with_icon=True)
    
    # Inputan
    st.subheader("1. Data Fisik")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Umur (Tahun)", min_value=10, max_value=80, value=25, key="age_input")
    with col2:
        height = st.number_input("Tinggi Badan (Meter)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, help="Gunakan titik untuk desimal, misal 1.70", key="height_input")
    with col3:
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70, key="weight_input")
    
    gender = st.selectbox("Jenis Kelamin", options=['Laki-laki', 'Perempuan'], key="gender_input")
    family_hist = st.selectbox("Ada riwayat keluarga obesitas?", options=['Tidak', 'Ya'], key="family_hist_input")

    st.subheader("2. Pola Makan")
    col4, col5 = st.columns(2)
    with col4:
        fcvc = st.slider("Konsumsi Sayuran (FCVC)", 1.0, 3.0, 2.0, help="1: Jarang, 3: Selalu", key="fcvc_input")
        ncp = st.slider("Jumlah Makan Utama per Hari (NCP)", 1.0, 4.0, 3.0, key="ncp_input")
        ch2o = st.slider("Konsumsi Air Minum (Liter/hari)", 1.0, 3.0, 2.0, key="ch2o_input")
    with col5:
        favc = st.selectbox("Sering makan tinggi kalori? (FAVC)", options=['Tidak', 'Ya'], key="favc_input")
        caec = st.selectbox("Seberapa sering ngemil? (CAEC)", options=list(map_caec.keys()), key="caec_input")
        calc = st.selectbox("Konsumsi Alkohol? (CALC)", options=list(map_calc.keys()), key="calc_input")

    st.subheader("3. Gaya Hidup & Aktivitas")
    col6, col7 = st.columns(2)
    with col6:
        smoke = st.selectbox("Apakah Anda Merokok? (SMOKE)", options=['Tidak', 'Ya'], key="smoke_input")
        scc = st.selectbox("Memantau asupan kalori? (SCC)", options=['Tidak', 'Ya'], key="scc_input")
    with col7:
        faf = st.slider("Frekuensi Aktivitas Fisik (Hari/Minggu)", 0.0, 3.0, 1.0, key="faf_input")
        tue = st.slider("Waktu penggunaan teknologi (Jam/hari)", 0.0, 2.0, 1.0, key="tue_input")
        mtrans = st.selectbox("Transportasi Utama (MTRANS)", options=list(map_mtrans.keys()), key="mtrans_input")

    # button
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    col_back, col_submit = st.columns([1, 1])

    with col_submit:
        submit = st.button("🔍 Analisis Sekarang", key="analyze_button")
    
    with col_back:
        if st.button("Kembali ke Halaman Awal", key="back_button"):
            st.session_state.page = "start"
            st.rerun()

    
    # LOGIKA PREDIKSI 
    if submit and model is not None and scaler is not None:
        
        # 1. Kumpulkan Data dari session_state
        data = {
            'age': st.session_state.age_input, 'height': st.session_state.height_input, 'weight': st.session_state.weight_input, 
            'gender': st.session_state.gender_input, 'family_hist': st.session_state.family_hist_input,
            'fcvc': st.session_state.fcvc_input, 'ncp': st.session_state.ncp_input, 'ch2o': st.session_state.ch2o_input, 
            'favc': st.session_state.favc_input, 'caec': st.session_state.caec_input, 'calc': st.session_state.calc_input,
            'smoke': st.session_state.smoke_input, 'scc': st.session_state.scc_input, 'faf': st.session_state.faf_input, 
            'tue': st.session_state.tue_input, 'mtrans': st.session_state.mtrans_input
        }
        
        # 2. Preprocessing dan Prediksi
        bmi = data['weight'] / (data['height'] ** 2)
        
        row = [
            map_gender[data['gender']], data['age'], data['height'], data['weight'], map_yes_no[data['family_hist']], 
            map_yes_no[data['favc']], data['fcvc'], data['ncp'], map_caec[data['caec']], map_yes_no[data['smoke']], 
            data['ch2o'], map_yes_no[data['scc']], data['faf'], data['tue'], map_calc[data['calc']], 
            map_mtrans[data['mtrans']], bmi
        ]
        
        columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                   'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                   'CALC', 'MTRANS', 'BMI']
        
        df_input = pd.DataFrame([row], columns=columns)
        numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
        
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        prediction = model.predict(df_input)[0]
        probabilitas = model.predict_proba(df_input)
        confidence = np.max(probabilitas) * 100
        
        # 3. Simpan Hasil ke Session State
        st.session_state.results = {
            'bmi': bmi,
            'prediction': prediction,
            'confidence': confidence,
            'probabilitas': probabilitas[0]
        }
        
        # 4. Pindah Halaman
        st.session_state.page = "result"
        st.rerun()

# Halaman 3: Result Page 
def result_page():
    if 'results' not in st.session_state:
        st.error("Hasil prediksi tidak ditemukan. Silakan kembali ke halaman input.")
        st.session_state.page = "main"
        st.rerun()
        return

    st.markdown("<div class='card-flag'></div>", unsafe_allow_html=True)
    render_header("Hasil Analisis Deteksi Obesitas", with_icon=True)
    
    results = st.session_state.results
    
    st.divider()
    st.subheader("📋 Hasil Prediksi")
    
    bmi = results['bmi']
    prediction = results['prediction']
    confidence = results['confidence']
    probabilitas = results['probabilitas']
    
    st.info(f"Indeks Massa Tubuh (BMI) Anda: **{bmi:.2f}**")
    
    hasil_teks = label_hasil.get(prediction, "Tidak Diketahui")
    
    # Logika tampilan hasil
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
        "Kategori": list(label_hasil.values()),
        "Probabilitas": probabilitas
    })
    # Tampilkan grafik
    st.bar_chart(df_prob.set_index("Kategori"))

    # button
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    col_back, col_finish = st.columns([1, 1])
    
    with col_back:
        if st.button("⬅️ Kembali ke Form Input"):
            st.session_state.page = "main"
            st.rerun()
            
    with col_finish:
        if st.button("✅ Selesai"):
            st.session_state.page = "start"
            st.rerun()

# main routing
if st.session_state.page == "start":
    start_page()
elif st.session_state.page == "main":
    if model is not None and scaler is not None:
        main_page()
    else:
        pass 
elif st.session_state.page == "result":
    result_page()
