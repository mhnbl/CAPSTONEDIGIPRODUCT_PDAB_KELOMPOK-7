import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
import geopandas
import plotly.io as pio
import altair as alt
from functions import *

class LinearRegression:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * np.eye(X_b.shape[1])).dot(X_b.T).dot(y)
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X_b.dot(np.hstack((self.bias, self.weights)))
        return y_pred

    def set_params(self, **params):
        self.alpha = params['alpha']
        return self

df = pd.read_csv('Data.csv')

st.set_page_config(
    page_title="Renewable Energy",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed")

alt.themes.enable("dark")

st.title("Analisis Konsumsi Energi Terbarukan Menggunakan Metode Supervised Learning")

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analisys", "Business Understanding", "Prediction Model"])

with tab1:
    st.title("Explorary Data Analysis")
        
    col = st.columns((4.5, 4.5), gap='medium')

    with col[0]:
        #
        st.subheader('Electricity Generation By Source')
        years = df['Year'].unique()
        selected_year = st.select_slider('Select Year', options=years)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Fossil Fuels')
            energy_type = 'Fossil Fuels'
            st.altair_chart(make_donut(selected_year, energy_type, 'red'))
        with col2:
            st.write('Nuclear')
            energy_type = 'Nuclear'
            st.altair_chart(make_donut(selected_year, energy_type, 'blue'))
        with col3:
            st.write('Renewables')
            energy_type = 'Renewables'
            st.altair_chart(make_donut(selected_year, energy_type, 'green'))
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("See Explanation"):
            st.write("")

        #
        st.subheader("Renewable Energy Share VS Electricity From Fossil Fuels")
        renewable_v_fossil()

        #
        st.subheader("Top 10 Countries with Highest Average Renewable Sourced Electricity (TWh)")
        top10()
    
    with col[1]:
        #
        st.subheader("Average Annual Electricity Generation")
        st.plotly_chart(plot_average_electricity())
        with st.expander("See Explanation"):
            st.write(' Jika dilihat dari diatas pola konsumsi listrik di berbagai sumber (bahan bakar fosil, nuklir, terbarukan) dari waktu ke waktu (tahun). Dengan menghitung rata-rata untuk setiap tahun, Anda dapat mengidentifikasi potensi peningkatan atau penurunan ketergantungan pada sumber tertentu. Listrik dari bahan bakar fosil yang memiliki  penggunaan yang palig tinggi ditahun 2018.')
            st.write('Penggunaan Listrik dari bahan bakar fosil yang merupakan energi paling banyak digunakan pada tahun 2018. penggunaan Listrik dari terbarukan paling banyak digunakan pada tahun 2020. penggunaan Listrik dari nuklir paling banyak digunakan ditahun 2006.')    

        #
        st.subheader("Renewable Energy Consumption by Region")
        region_consumptions()

        st.subheader('Access to Electricity (% of Population) Over Years')
        map()

with tab2:
    st.subheader("Business Objective")
    st.write("Tujuan bisnis dari proyek ini adalah untuk memahami pola konsumsi energi terbarukan di setiap negara di dunia serta faktor-faktor yang memengaruhinya. Selain itu, tujuan utamanya adalah untuk membangun model regresi yang dapat memprediksi tingkat konsumsi energi terbarukan di masa depan berdasarkan variabel-variabel yang ada dalam dataset ini.")
    st.subheader("Assess Situation")
    st.write("Dengan meningkatnya kesadaran akan isu-isu lingkungan dan pergeseran ke arah energi terbarukan, pemangku kepentingan seperti pemerintah, perusahaan energi, dan lembaga penelitian tertarik untuk memahami perilaku konsumsi energi terbarukan di setiap negara. Analisis ini akan membantu mereka dalam merencanakan kebijakan energi, investasi, dan proyek-proyek infrastruktur terkait energi di masa depan.")    
    st.subheader("Data Mining Goals")
    st.write("Tujuan dari analisis ini adalah:")
    lst = ['Mengidentifikasi faktor-faktor kunci yang mempengaruhi tingkat konsumsi energi terbarukan di setiap negara, seperti akses listrik, akses bahan bakar bersih untuk memasak, kapasitas pembangkit listrik terbarukan per kapita, aliran keuangan ke negara-negara berkembang, dan lain-lain.', 'Membangun model regresi yang dapat memprediksi konsumsi energi terbarukan di masa depan berdasarkan faktor-faktor tersebut.', 'Memberikan wawasan kepada pemerintah, perusahaan energi, dan lembaga penelitian tentang pola konsumsi energi terbarukan di berbagai negara, serta implikasinya terhadap kebijakan dan investasi di sektor energi.']
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.subheader("Project Plan")
    st.write("Rencana proyek ini mencakup beberapa tahap:")
    lst = ['Pengumpulan Data: Data dikumpulkan dari sumber-sumber terpercaya seperti lembaga internasional, database pemerintah, dan penelitian akademis. Data yang relevan mencakup variabel-variabel seperti akses listrik, akses bahan bakar bersih untuk memasak, kapasitas pembangkit listrik terbarukan, aliran keuangan ke negara-negara berkembang, dan lain-lain.','Preprocessing Data: Data yang dikumpulkan kemudian akan dibersihkan dan dipersiapkan untuk analisis. Ini termasuk penanganan nilai-nilai yang hilang, normalisasi data, dan penghapusan outlier jika diperlukan.','Analisis Data: Analisis data dilakukan untuk memahami hubungan antara variabel-variabel yang ada dan tinwgkat konsumsi energi terbarukan. Ini mungkin melibatkan eksplorasi visual, korelasi, dan analisis regresi sederhana.','Analisis Data: Analisis data dilakukan untuk memahami hubungan antara variabel-variabel yang ada dan tinwgkat konsumsi energi terbarukan. Ini mungkin melibatkan eksplorasi visual, korelasi, dan analisis regresi sederhana.','Pembangunan Model Regresi: Model regresi akan dibangun menggunakan metode supervised learning, di mana variabel targetnya adalah tingkat konsumsi energi terbarukan, dan variabel-variabel lainnya digunakan sebagai fitur-fitur prediktif. Model ini akan diuji dan dievaluasi untuk memastikan kinerjanya yang baik.','Interpretasi dan Pelaporan Hasil: Hasil analisis akan diinterpretasikan dan disajikan dalam bentuk yang mudah dimengerti, baik dalam bentuk laporan tertulis maupun presentasi. Implikasi dari temuan ini terhadap kebijakan energi dan investasi akan dibahas.']
    s = ''
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)


with tab3:
    st.subheader("Predict")
    predict()