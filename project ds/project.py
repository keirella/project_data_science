
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64
import os

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")

# load css
def load_css(file, bg_img=None):
    with open(file, "r") as f:
        css = f.read()
    if bg_img:
        with open(bg_img, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()
        css = css.replace("BG_IMAGE", b64)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# memastikan path file selalu benar
DIR = os.path.dirname(__file__)
load_css(os.path.join(DIR, "style.css"), os.path.join(DIR, "image/lucu.png"))

def render_header(text, with_icon=True):
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:8px;'>{text}</h1>",
        unsafe_allow_html=True
    )

    if with_icon:
        try:
            with open("image/icon.jpg", "rb") as f:
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

if "page" not in st.session_state:
    st.session_state.page = "start"

# halaman start
def start_page():
    # penanda card
    st.markdown("<div class='card-flag'></div>", unsafe_allow_html=True)
    render_header("Prediksi Obesitas", with_icon=True)

    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
    
    if st.button("Mulai"):
        st.session_state.page = "main"
        st.rerun()

# halaman utama
def main_page():
    # penanda card
    st.markdown("<div class='card-flag'></div>", unsafe_allow_html=True)
    render_header("Form Prediksi Obesitas", with_icon=True)

    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    df["BMI"] = df["Weight"] / (df["Height"] * df["Height"])
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["FAVC"] = df["FAVC"].map({"no": 0, "yes": 1})
    df["CAEC"] = df["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})

    X = df[['Gender','Age','Height','Weight','FAVC','FCVC','NCP','CAEC','FAF','TUE','BMI']]
    y = df["NObeyesdad"]

    model = RandomForestClassifier()
    model.fit(X, y)

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Umur", 1, 120)
    height = st.number_input("Tinggi (m)", 0.5, 2.5)
    weight = st.number_input("Berat (kg)", 10.0, 300.0)
    favc = st.selectbox("Sering makan tinggi kalori", ["no", "yes"])
    fcvc = st.slider("Frekuensi makan sayur 1-3", 1, 3)
    ncp = st.slider("Jumlah makan per hari 1-4", 1, 4)
    caec = st.selectbox("Kebiasaan ngemil", ["no","Sometimes","Frequently","Always"])
    faf = st.slider("Aktivitas fisik 0-3", 0.0, 3.0)
    tue = st.slider("Waktu layar 0-2", 0.0, 2.0)

    bmi = weight / (height * height)

    data = np.array([
        1 if gender == "Male" else 0,
        age, height, weight,
        1 if favc == "yes" else 0,
        fcvc, ncp,
        {"no":0,"Sometimes":1,"Frequently":2,"Always":3}[caec],
        faf, tue, bmi
    ]).reshape(1,-1)

    if st.button("Prediksi"):
        hasil = model.predict(data)[0]
        st.success("Hasil prediksi: " + hasil)

    if st.button("Kembali"):
        st.session_state.page = "start"
        st.rerun()

# routing
if st.session_state.page == "start":
    start_page()
else:
    main_page()
