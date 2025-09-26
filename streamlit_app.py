# =========================================================
# Aplikasi Prediksi Penumpang Kereta Api (eksperimen.py)
# Modifikasi untuk mendukung multi-file upload & AI Chatbot
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
import re

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Prediksi Penumpang KRL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INTEGRASI GROQ API ---
from groq import Groq

# Mengakses API key dari file secrets.toml
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY tidak ditemukan di file .streamlit/secrets.toml. Harap tambahkan API key Anda.")
    groq_api_key = None

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None
# --- AKHIR INTEGRASI GROQ API ---

# Menggunakan gaya visualisasi yang lebih elegan dan serasi dengan tema Streamlit
plt.style.use('dark_background')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- FUNGSI BANTU UNTUK FORMAT HEADER DAN ANGKA ---

def _wrap_header_text(text, max_len=15):
    """Memecah teks header yang panjang menjadi beberapa baris."""
    if len(text) > max_len:
        parts = text.split()
        wrapped_text = ''
        current_len = 0
        for part in parts:
            if current_len + len(part) + 1 <= max_len:
                wrapped_text += ' ' + part
                current_len += len(part) + 1
            else:
                wrapped_text += '\n' + part
                current_len = len(part)
        return wrapped_text.strip()
    return text

def _format_indonesian_numeric(value, decimals=0):
    """
    Memformat angka menjadi string dengan pemisah ribuan titik.
    Jika decimals=0, akan dibulatkan ke integer.
    Jika decimals > 0, akan menampilkan desimal.
    """
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return value

# --- UI Components ---
def create_header():
    """Membuat header aplikasi."""
    st.markdown(
        """
        <div style="background-color: transparent; padding:10px; text-align:center;">
            <h1 style="color:#F63366; font-family: 'Segoe UI', sans-serif; font-weight: bold; font-size: 28px;">
                🚆 Aplikasi Prediksi Jumlah Penumpang KRL Commuter Line Jabodetabek
            </h1>
            <p style="color:#FFFFFF; font-family: 'Segoe UI', sans-serif; font-size: 16px;">
                Unggah file, analisis data, bangun model, dan lihat hasil prediksi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_footer():
    """Membuat footer aplikasi."""
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #191e24;
            color: #FFFFFF;
            text-align: center;
            padding: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 12px;
            border-top: 1px solid #F63366;
            box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
            z-index: 999;
        }
        </style>
        <div class="footer">
            <p>© 2025 - Skripsi Zia Pauzia. All Rights Reserved. Versi 1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_sidebar_menu():
    """Membuat menu navigasi sidebar dan kontrol visualisasi."""
    st.sidebar.markdown(
        """
        <div style="background-color:#F63366; padding:15px; border-radius:10px; text-align:center; margin-bottom: 20px;">
            <h2 style="color:white; font-size:20px; font-weight:bold;">Navigasi Tahapan</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    
    data_uploaded = 'data_loaded' in st.session_state and st.session_state.data_loaded
    if not data_uploaded:
        st.sidebar.info("Mohon unggah file data untuk memulai.")
    else:
        st.sidebar.success("✅ Data berhasil dimuat.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pilihan Visualisasi")
    st.session_state.chart_type_option = st.sidebar.selectbox(
        "Pilih jenis grafik:",
        ('Garis', 'Batang')
    )

    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 1. Unggah Data")
    if st.sidebar.button("📁 Mulai Unggah", key="btn_upload"):
        st.session_state.page = 'upload'
        st.rerun()
    
    st.sidebar.markdown("### 2. Analisis & Evaluasi")
    if st.sidebar.button("🗂️ Pratinjau Data", key="btn_show_data", disabled=not data_uploaded):
        st.session_state.page = 'show_data'
        st.rerun()
    if st.sidebar.button("📊 Analisis Data", key="btn_analysis", disabled=not data_uploaded):
        st.session_state.page = 'data_analysis'
        st.rerun()
    if st.sidebar.button("📈 Modeling", key="btn_modeling", disabled=not data_uploaded):
        st.session_state.page = 'modeling_evaluation'
        st.rerun()
    
    st.sidebar.markdown("### 3. Hasil & Prediksi")
    if st.sidebar.button("🚀 Deployment", key="btn_deployment", disabled=not data_uploaded):
        st.session_state.page = 'deployment'
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fitur Lain")
    if st.sidebar.button("🏠 Halaman Utama", key="btn_home"):
        st.session_state.page = 'home'
        st.rerun()
    if st.sidebar.button("🤖 Tanya AI", key="btn_chatbot"):
        st.session_state.page = 'chatbot'
        st.rerun()

# --- Fungsi-fungsi Respons Chatbot yang Diprogram (Hardcoded) ---
def handle_summary_response():
    """Menampilkan respons ringkasan prediksi."""
    with st.chat_message("assistant"):
        results = st.session_state.model_results
        mape_training = results['mape_training']
        mape_testing = results['mape_testing']
        accuracy_status = ""
        
        if mape_testing <= 10:
            accuracy_status = "Sangat Akurat (< 10%)"
        elif mape_testing <= 20:
            accuracy_status = "Akurat (10% - 20%)"
        elif mape_testing <= 50:
            accuracy_status = "Cukup Akurat (20% - 50%)"
        else:
            accuracy_status = "Tidak Akurat (> 50%)"
            
        st.subheader("Kesimpulan Prediksi")
        st.write(f"""
        Model prediksi Anda memiliki tingkat akurasi yang **{accuracy_status}** dengan rincian:
        - **MAPE Training:** `{_format_indonesian_numeric(mape_training, 2)}%`
        - **MAPE Testing:** `{_format_indonesian_numeric(mape_testing, 2)}%`
        
        Nilai MAPE yang rendah menunjukkan bahwa model memiliki kesalahan persentase rata-rata yang kecil, sehingga prediksi model cukup dapat diandalkan.
        """)

def handle_coefficient_response():
    """Menampilkan respons penjelasan koefisien model."""
    with st.chat_message("assistant"):
        if 'model_results' in st.session_state and st.session_state.data_loaded:
            results = st.session_state.model_results
            model = results['model']
            intercept = model.intercept_
            coefs = model.coef_
            features = results['features']
            
            st.subheader("Penjelasan Koefisien Model Regresi")
            st.markdown(f"**Intercept (Konstanta):** `{_format_indonesian_numeric(intercept, 2)}`")
            st.write("Ini adalah nilai rata-rata jumlah penumpang ketika semua variabel independen bernilai nol.")
            st.markdown("---")
            
            for i, feature in enumerate(features):
                coef = coefs[i]
                st.markdown(f"**Koefisien untuk `{feature}`:** `{_format_indonesian_numeric(coef, 2)}`")
                st.write(f"Setiap kenaikan 1 satuan pada `{feature}` akan meningkatkan atau menurunkan jumlah penumpang sebesar `{_format_indonesian_numeric(coef, 2)}` (dalam ribuan), dengan asumsi variabel lain konstan.")
                st.markdown("---")
        else:
            st.warning("Maaf, model belum dilatih. Mohon proses data terlebih dahulu.")

def handle_5_year_forecast_response():
    """Menampilkan respons prediksi 5 tahun ke depan."""
    with st.chat_message("assistant"):
        if 'df_future' in st.session_state:
            st.subheader("Prediksi Penumpang untuk 5 Tahun ke Depan")
            st.write("Berdasarkan model regresi yang dilatih, berikut adalah prediksi jumlah penumpang (dalam ribuan) untuk 5 tahun ke depan.")
            df_future_display = st.session_state.df_future[['Bulan', 'Tahun', 'Penumpang (000)']].copy()
            styled_df = df_future_display.style.format({
                'Tahun': '{:.0f}',
                'Penumpang (000)': lambda x: _format_indonesian_numeric(x, 0)
            })
            st.dataframe(styled_df)
            
            df_combined_all = pd.concat([st.session_state.df_training, st.session_state.df_testing, st.session_state.df_future], ignore_index=True)
            df_combined_all['Jenis Data'] = np.nan
            df_combined_all.loc[len(st.session_state.df_training)+len(st.session_state.df_testing):, 'Jenis Data'] = 'Prediksi 5 Tahun'
            df_combined_all.loc[:len(st.session_state.df_training)-1, 'Jenis Data'] = 'Training'
            df_combined_all.loc[len(st.session_state.df_training):len(st.session_state.df_training)+len(st.session_state.df_testing)-1, 'Jenis Data'] = 'Testing'

            st.markdown("##### Grafik Tren dan Prediksi 5 Tahun")
            st.line_chart(df_combined_all, x='Bulan ke-n', y='Penumpang (000)', color='Jenis Data')

        else:
            st.warning("Maaf, data prediksi 5 tahun belum tersedia. Mohon proses data terlebih dahulu.")
        
def handle_mape_difference_response():
    """Menjelaskan mengapa MAPE pada training dan testing berbeda."""
    with st.chat_message("assistant"):
        st.subheader("Perbedaan MAPE Training dan Testing")
        st.write("""
        Perbedaan nilai MAPE (Mean Absolute Percentage Error) antara data training dan testing adalah hal yang wajar dan penting.
        - **MAPE Training** mengukur seberapa baik model beradaptasi dengan data yang digunakan untuk melatihnya.
        - **MAPE Testing** mengukur seberapa baik model berkinerja pada data baru yang belum pernah dilihat sebelumnya.
        
        Jika MAPE testing jauh lebih tinggi dari MAPE training, ini bisa menjadi indikasi **overfitting**, di mana model terlalu spesifik terhadap data training. Namun, sedikit perbedaan adalah normal dan menunjukkan model Anda cukup baik dalam menggeneralisasi.
        """)

def handle_long_term_prediction_response():
    """Menjelaskan pro dan kontra prediksi jangka panjang."""
    with st.chat_message("assistant"):
        st.subheader("Pro dan Kontra Prediksi Jangka Panjang")
        st.write("""
        Memprediksi untuk jangka waktu yang lebih panjang (misalnya 10 tahun) bisa jadi tidak akurat karena:
        - **Asumsi Data Rata-rata:** Model mengasumsikan nilai rata-rata dari variabel seperti jarak tempuh dan libur, yang mungkin tidak akurat dalam jangka panjang.
        - **Perubahan Pola:** Faktor-faktor yang tidak ada dalam model (misalnya pembangunan jalur baru, pandemi, atau perubahan kebijakan) bisa sangat memengaruhi tren data di masa depan.
        
        Oleh karena itu, prediksi untuk 5 tahun sudah cukup optimistis, dan untuk 10 tahun harus diperlakukan dengan sangat hati-hati, karena tingkat ketidakpastiannya jauh lebih tinggi.
        """)

def handle_improvement_response():
    """Memberikan saran untuk meningkatkan akurasi model."""
    with st.chat_message("assistant"):
        st.subheader("Saran untuk Meningkatkan Akurasi Model")
        st.write("""
        Beberapa cara yang dapat Anda lakukan untuk meningkatkan akurasi model adalah:
        - **Menambahkan Variabel Baru:** Coba tambahkan variabel lain yang mungkin memengaruhi jumlah penumpang, misalnya harga bahan bakar, harga tiket, atau data acara besar (konser, festival).
        - **Penggunaan Model Lanjutan:** Eksplorasi model regresi yang lebih kompleks, seperti Random Forest, XGBoost, atau model deret waktu seperti Prophet, yang mungkin lebih baik dalam menangkap pola non-linier.
        - **Pembersihan Data (Data Cleaning):** Periksa kembali data untuk outlier atau kesalahan yang mungkin memengaruhi model.
        - **Peningkatan Ukuran Dataset:** Menggunakan lebih banyak data historis (jika tersedia) dapat membantu model belajar pola yang lebih baik.
        """)
        
def handle_specific_data_query(prompt):
    """Menangani pertanyaan spesifik tentang data di bulan dan tahun tertentu."""
    bulan_dict = {
        'januari': 'Januari', 'februari': 'Februari', 'maret': 'Maret', 'april': 'April',
        'mei': 'Mei', 'juni': 'Juni', 'juli': 'Juli', 'agustus': 'Agustus',
        'september': 'September', 'oktober': 'Oktober', 'november': 'November', 'desember': 'Desember'
    }
    
    match = re.search(r'(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(\d{4})', prompt.lower())
    
    if not match:
        return False
        
    bulan_str = bulan_dict[match.group(1)]
    tahun_int = int(match.group(2))
    
    with st.chat_message("assistant"):
        df_all = pd.concat([st.session_state.df_training, st.session_state.df_testing, st.session_state.df_future], ignore_index=True)
        
        data_found = df_all[(df_all['Bulan'] == bulan_str) & (df_all['Tahun'] == tahun_int)]
        
        if not data_found.empty:
            st.subheader(f"Data Penumpang di {bulan_str} {tahun_int}")
            st.dataframe(data_found.style.format({
                'Tahun': '{:.0f}',
                'Bulan ke-n': '{:.0f}',
                'Penumpang (000)': lambda x: _format_indonesian_numeric(x, 0),
                'Total Jarak Tempuh Penumpang': lambda x: _format_indonesian_numeric(x, 0),
                'Rata-rata Jarak Perjalanan Per penumpang': lambda x: _format_indonesian_numeric(x, 2),
                'jumlah_libur_nasional': '{:.0f}',
                'jumlah_cuti_bersama': '{:.0f}',
            }))
        else:
            st.warning(f"Maaf, tidak ditemukan data untuk bulan {bulan_str} tahun {tahun_int}.")
    
    return True

def handle_graph_request(prompt):
    """Menangani permintaan untuk menampilkan grafik."""
    if 'grafik' in prompt.lower() or 'diagram' in prompt.lower():
        with st.chat_message("assistant"):
            st.subheader("Visualisasi Tren Penumpang")
            
            df_combined_all = pd.concat([st.session_state.df_training, st.session_state.df_testing, st.session_state.df_future], ignore_index=True)
            df_combined_all['Jenis Data'] = np.nan
            df_combined_all.loc[len(st.session_state.df_training)+len(st.session_state.df_testing):, 'Jenis Data'] = 'Prediksi 5 Tahun'
            df_combined_all.loc[:len(st.session_state.df_training)-1, 'Jenis Data'] = 'Training'
            df_combined_all.loc[len(st.session_state.df_training):len(st.session_state.df_training)+len(st.session_state.df_testing)-1, 'Jenis Data'] = 'Testing'

            st.markdown("##### Grafik Tren dan Prediksi 5 Tahun")
            st.line_chart(df_combined_all, x='Bulan ke-n', y='Penumpang (000)', color='Jenis Data')

        return True
    return False

def handle_chatbot_response(prompt):
    """Fungsi utama untuk memproses respons chatbot."""
    if 'data_loaded' in st.session_state:
        if "kesimpulan dari hasil prediksi" in prompt.lower():
            handle_summary_response()
            return
        elif "koefisien model regresi" in prompt.lower():
            handle_coefficient_response()
            return
        elif "prediksi jumlah penumpang untuk 5 tahun ke depan" in prompt.lower():
            handle_5_year_forecast_response()
            return
        elif "mape pada data training berbeda dengan data testing" in prompt.lower():
            handle_mape_difference_response()
            return
        elif "model ini bisa digunakan untuk memprediksi lebih dari 5 tahun" in prompt.lower():
            handle_long_term_prediction_response()
            return
        elif "cara meningkatkan akurasi model ini" in prompt.lower():
            handle_improvement_response()
            return
        elif handle_specific_data_query(prompt):
            return
        elif handle_graph_request(prompt):
            return
    
    send_to_groq(prompt)

def show_chatbot_page():
    """Menampilkan antarmuka chatbot dengan Groq API."""
    st.title("🤖 Asisten AI: Tanya Tentang Aplikasi Ini")
    st.info("Anda bisa bertanya tentang konsep statistik, interpretasi hasil, atau cara menggunakan aplikasi ini.")
    
    if groq_client is None:
        st.warning("Integrasi chatbot tidak aktif. Harap pastikan GROQ_API_KEY sudah diatur dengan benar di file .streamlit/secrets.toml")
        return

    # Inisialisasi riwayat chat
    if "groq_messages" not in st.session_state:
        st.session_state.groq_messages = []

    # --- BAGIAN BARU: Tombol Pertanyaan yang Direkomendasikan ---
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        st.subheader("Pertanyaan Cepat:")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            if st.button("Apa kesimpulan dari hasil prediksi?", key="rec_q1"):
                st.session_state.chat_input = "Apa kesimpulan dari hasil prediksi ini berdasarkan dataset yang diupload?"
                st.session_state.chat_trigger = True
        
        with rec_col2:
            if st.button("Jelaskan koefisien model regresi", key="rec_q2"):
                st.session_state.chat_input = "Jelaskan koefisien model regresi yang dihasilkan dan apa signifikansinya."
                st.session_state.chat_trigger = True

        with rec_col3:
            if st.button("Tampilkan prediksi 5 tahun ke depan", key="rec_q3"):
                st.session_state.chat_input = "Tampilkan prediksi jumlah penumpang untuk 5 tahun ke depan."
                st.session_state.chat_trigger = True

        if 'show_reco_questions' in st.session_state and st.session_state.show_reco_questions:
            st.markdown("---")
            st.subheader("Pertanyaan Lanjutan (Setelah Proses Deployment):")
            rec_col4, rec_col5, rec_col6 = st.columns(3)
            with rec_col4:
                if st.button("Perbedaan MAPE Training dan Testing", key="rec_q4"):
                    st.session_state.chat_input = "Mengapa nilai Mean Absolute Percentage Error (MAPE) pada data training dan testing berbeda? Jelaskan alasannya."
                    st.session_state.chat_trigger = True
            with rec_col5:
                if st.button("Prediksi Jangka Panjang", key="rec_q5"):
                    st.session_state.chat_input = "Apakah model regresi linier ini bisa digunakan untuk memprediksi jumlah penumpang untuk jangka waktu yang lebih lama, misalnya 10 tahun ke depan? Jelaskan pro dan kontranya."
                    st.session_state.chat_trigger = True
            with rec_col6:
                if st.button("Saran untuk meningkatkan akurasi model", key="rec_q6"):
                    st.session_state.chat_input = "Berikan saran bagaimana cara meningkatkan akurasi model prediksi ini."
                    st.session_state.chat_trigger = True
    
    if 'chat_trigger' in st.session_state and st.session_state.chat_trigger:
        prompt = st.session_state.chat_input
        st.session_state.chat_trigger = False
        st.session_state.groq_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_chatbot_response(prompt)
    
    for message in st.session_state.groq_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
        st.session_state.groq_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_chatbot_response(prompt)

def send_to_groq(prompt):
    """Mengirim prompt ke Groq API dan menampilkan respons."""
    if 'model_results' in st.session_state and st.session_state.data_loaded:
        results = st.session_state.model_results
        
        # --- MODIFIKASI: Mendapatkan data untuk dimasukkan ke prompt ---
        mape_training = results['mape_training']
        mape_testing = results['mape_testing']
        mae_training = results['mae_training']
        mae_testing = results['mae_testing']
        
        # Lakukan perhitungan VIF
        X_vif = st.session_state.df_training[results['features']]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
        vif_text = vif_data.to_markdown(index=False)
        
        # Dapatkan status akurasi model
        mape_category_data = {
            'Nilai': ['<10%', '10% - 20%', '20% - 50%', '>50%'],
            'Kategori Peramalan': ['Sangat Akurat', 'Akurat', 'Cukup Akurat', 'Tidak Akurat']
        }
        mape_category_df = pd.DataFrame(mape_category_data)
        
        accuracy_status = ""
        mape_testing_real = results['mape_testing']
        if mape_testing_real <= 10:
            accuracy_status = "Sangat Akurat (< 10%)"
        elif mape_testing_real <= 20:
            accuracy_status = "Akurat (10% - 20%)"
        elif mape_testing_real <= 50:
            accuracy_status = "Cukup Akurat (20% - 50%)"
        else:
            accuracy_status = "Tidak Akurat (> 50%)"
        
        # Dapatkan ringkasan OLS
        X = st.session_state.df_training[results['features']]
        y = st.session_state.df_training['Penumpang (000)']
        X_with_const = sm.add_constant(X)
        model_ols = sm.OLS(y, X_with_const).fit()
        ols_summary = model_ols.summary().as_text()

        # Dapatkan sampel data prediksi testing
        df_testing_sample = st.session_state.df_testing[['Bulan', 'Tahun', 'Penumpang (000)', 'Bulan ke-n']].copy()
        df_testing_sample['Y_Prediksi'] = results['y_pred_testing']
        df_testing_sample['Selisih'] = np.abs(df_testing_sample['Penumpang (000)'] - df_testing_sample['Y_Prediksi'])
        
        # --- BAGIAN BARU: CONTEXT PROMPT YANG LEBIH LENGKAP ---
        context_prompt = f"""
        Anda adalah asisten AI yang ahli dalam statistik dan prediksi regresi. Anda akan menjawab pertanyaan pengguna terkait hasil prediksi yang baru saja dibuat. Gunakan semua informasi dan konteks berikut untuk memberikan jawaban yang akurat, terperinci, dan relevan.

        ---

        ### **Tujuan dan Proses Aplikasi**
        Aplikasi ini melakukan prediksi jumlah penumpang KRL Commuter Line Jabodetabek menggunakan algoritma regresi linier berganda berdasarkan data historis yang diupload. Aplikasi telah melalui serangkaian proses:
        1.  Pengecekan korelasi antar variabel.
        2.  Uji asumsi klasik (Normalitas, Homoskedastisitas, Non-Multikolinieritas).
        3.  Pembuatan model regresi berdasarkan data training.
        4.  Melakukan prediksi pada data testing dan data 5 tahun ke depan.
        5.  Menentukan akurasi hasil prediksi berdasarkan nilai MAPE dan MAE pada data training dan data testing dengan memperhatikan kategori nilai MAPE pada tingkat angka berapa nilai MAPE tersebut tergolong.
        6.  Memaparkan hasilnya dengan membandingkan juga antara data aktual dengan data prediksi pada perhitungan.
        7.  Lalu menyimpulkannya dengan mengecek antara nilai prediksi dengan nilai aktual didunia nyata apakah hasil prediksi ini baik atau tidak untuk membantu pengelola pihak PT KAI dalam mengambil keputusan.

        ---

        ### **Konteks Dataset di Dunia Nyata**
        - **`Penumpang (000)`:** Angka dalam ribuan orang. Contoh: 39.861 berarti 39.861.000 penumpang.
        - **`Total Jarak Tempuh Penumpang (000.000 km)`:** Angka dalam juta kilometer. Contoh: 2.234 berarti 2.234.000.000 km.
        - **`Rata-rata Jarak Perjalanan Per Penumpang (km)`:** Angka dalam kilometer. Contoh: 56 berarti 56 km per penumpang.

        ---

        ### **Hasil Analisis dan Evaluasi Model**
        - **Status Akurasi Model**: {accuracy_status}
        - **Tabel Kategori Akurasi MAPE**:
        {mape_category_df.to_markdown(index=False)}
        - **MAE (Mean Absolute Error) Training:** {mae_training}
        - **MAPE (Mean Absolute Percentage Error) Training:** {mape_training}%
        - **MAE (Mean Absolute Error) Testing:** {mae_testing}
        - **MAPE (Mean Absolute Percentage Error) Testing:** {mape_testing}%
        - **Ringkasan Koefisien Model**: {model_ols.summary2().tables[1].to_markdown()}
        - **Nilai VIF (untuk Multikolinieritas)**:
        {vif_text}
        
        - **Sampel Hasil Prediksi (Data Testing)**:
        {df_testing_sample.to_markdown(index=False)}

        ---
        
        Berdasarkan semua informasi di atas, jawablah pertanyaan pengguna dengan analisis yang mendalam dan relevan.
        """
    else:
        context_prompt = "Belum ada data yang diunggah dan diproses. Anda hanya bisa menjawab pertanyaan umum."

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        messages_to_send = []
        if context_prompt:
            messages_to_send.append({"role": "system", "content": context_prompt})
        
        messages_to_send.extend([
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.groq_messages
        ])
        
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages_to_send,
                stream=True,
            )
            
            for chunk in completion:
                full_response += (chunk.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            st.session_state.groq_messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")

def predict_5_years(model, df_training):
    """Melakukan prediksi 5 tahun ke depan dan menyimpan hasilnya di session_state."""
    if not 'df_training' in st.session_state:
        return

    # Ambil rata-rata variabel independen (kecuali Bulan ke-n) dari data training
    avg_total_jarak = df_training['Total Jarak Tempuh Penumpang'].mean()
    avg_rata_jarak = df_training['Rata-rata Jarak Perjalanan Per penumpang'].mean()
    avg_libur_nasional = df_training['jumlah_libur_nasional'].mean()
    avg_cuti_bersama = df_training['jumlah_cuti_bersama'].mean()

    # Buat DataFrame untuk 5 tahun ke depan (60 bulan)
    start_month = len(df_training) + len(st.session_state.df_testing) + 1
    future_months = np.arange(start_month, start_month + 60)
    
    df_future = pd.DataFrame({
        'Bulan ke-n': future_months,
        'Total Jarak Tempuh Penumpang': [avg_total_jarak] * 60,
        'Rata-rata Jarak Perjalanan Per penumpang': [avg_rata_jarak] * 60,
        'jumlah_libur_nasional': [avg_libur_nasional] * 60,
        'jumlah_cuti_bersama': [avg_cuti_bersama] * 60
    })
    
    # Lakukan prediksi
    y_pred_future = model.predict(df_future)
    
    # Tambahkan hasil prediksi ke DataFrame
    df_future['Penumpang (000)'] = y_pred_future
    
    st.session_state.df_future = df_future
    st.session_state.df_future.reset_index(drop=True, inplace=True)
    
    # Tambahkan Bulan dan Tahun ke df_future
    current_year = df_training['Tahun'].iloc[-1]
    current_month_index = df_training['Bulan'].iloc[-1]
    month_mapping = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    reverse_month_mapping = {v: k for k, v in month_mapping.items()}
    
    start_month_num = month_mapping[current_month_index] + 1
    start_year = current_year
    
    months = []
    years = []
    
    for i in range(60):
        month_num = (start_month_num + i - 1) % 12 + 1
        months.append(reverse_month_mapping[month_num])
        
        year_to_add = (start_month_num + i - 1) // 12
        years.append(start_year + year_to_add)

    df_future.insert(0, 'Bulan', months)
    df_future.insert(1, 'Tahun', years)
    
# --- Core Logic Functions ---
def _read_penumpang_file(uploaded_files):
    """
    MODIFIKASI: Menerima list file dan menggabungkannya.
    Fungsi untuk membaca data penumpang dalam format "wide".
    """
    if not uploaded_files:
        return None, "Tidak ada file penumpang yang diunggah."

    # Pemetaan nama bulan ke angka untuk pengurutan
    month_mapping = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    df_list = []
    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            header_row = 3
            if file_extension == '.xlsx':
                df_raw = pd.read_excel(uploaded_file, header=header_row)
            elif file_extension == '.csv':
                df_raw = pd.read_csv(uploaded_file, header=header_row)
            else:
                raise ValueError(f"Tipe file tidak didukung: {uploaded_file.name}")
            
            df_raw.dropna(how='all', inplace=True)
            df_raw.reset_index(drop=True, inplace=True)
            
            df_transposed = df_raw.set_index(df_raw.columns[0]).T.reset_index()
            df_transposed.columns.name = None
            df_transposed.rename(columns={'index': 'Bulan'}, inplace=True)
            df_transposed = df_transposed[df_transposed['Bulan'] != 'Tahunan'].copy()
            
            uploaded_file.seek(0)
            if file_extension == '.xlsx':
                df_year_raw = pd.read_excel(uploaded_file, header=None)
            else:
                df_year_raw = pd.read_csv(uploaded_file, header=None)
            
            tahun = next((int(cell) for row in df_year_raw.values for cell in row if isinstance(cell, (int, str)) and str(cell).isdigit() and len(str(cell)) == 4), None)
            if tahun is None:
                raise ValueError(f"Tidak dapat menemukan tahun pada file: {uploaded_file.name}")
            
            penumpang_col = next((col for col in df_transposed.columns if 'penumpang' in str(col).lower() and '000' in str(col).lower()), None)
            total_jarak_col = next((col for col in df_transposed.columns if 'total jarak' in str(col).lower()), None)
            rata_jarak_col = next((col for col in df_transposed.columns if 'rata-rata jarak' in str(col).lower() or 'rata2' in str(col).lower()), None)
            
            if not all([penumpang_col, total_jarak_col, rata_jarak_col]):
                raise ValueError(f"Tidak dapat menemukan kolom metrik yang diperlukan di {uploaded_file.name}")
            
            df_final = df_transposed[['Bulan', penumpang_col, total_jarak_col, rata_jarak_col]].copy()
            df_final.rename(columns={
                penumpang_col: 'Penumpang (000)',
                total_jarak_col: 'Total Jarak Tempuh Penumpang',
                rata_jarak_col: 'Rata-rata Jarak Perjalanan Per penumpang'
            }, inplace=True)
            
            df_final['Tahun'] = tahun
            
            def safe_numeric_conversion(series, thousands_separator):
                cleaned_series = series.astype(str).str.replace(thousands_separator, '', regex=False)
                return pd.to_numeric(cleaned_series, errors='coerce')

            df_final['Penumpang (000)'] = safe_numeric_conversion(df_final['Penumpang (000)'], '.')
            df_final['Total Jarak Tempuh Penumpang'] = safe_numeric_conversion(df_final['Total Jarak Tempuh Penumpang'], '.')
            df_final['Rata-rata Jarak Perjalanan Per penumpang'] = safe_numeric_conversion(df_final['Rata-rata Jarak Perjalanan Per penumpang'], ',').round(2)

            df_final.dropna(inplace=True)
            df_final.reset_index(drop=True, inplace=True)
            
            # --- Perbaikan: Tambahkan kolom angka bulan untuk pengurutan yang benar ---
            df_final['Bulan_Angka'] = df_final['Bulan'].apply(lambda x: month_mapping.get(x, 0))
            # --- Akhir Perbaikan ---

            df_list.append(df_final)

        except Exception as e:
            st.error(f"ERROR saat membaca file {uploaded_file.name}: {e}")
            return None, f"ERROR saat membaca file {uploaded_file.name}: {e}"

    if not df_list:
        return None, "Tidak ada data yang valid untuk digabungkan."
        
    df_combined = pd.concat(df_list, ignore_index=True)
    # --- Perbaikan: Urutkan berdasarkan Tahun dan Bulan_Angka (kronologis) ---
    df_combined.sort_values(by=['Tahun', 'Bulan_Angka'], inplace=True)
    # --- Akhir Perbaikan ---
    df_combined.reset_index(drop=True, inplace=True)
    
    # Hapus kolom Bulan_Angka setelah pengurutan selesai
    df_combined.drop(columns='Bulan_Angka', inplace=True)
    
    return df_combined, None

def _read_libur_file(uploaded_files):
    """
    MODIFIKASI: Menerima list file dan menggabungkan.
    Fungsi untuk membaca data libur dalam format "long".
    """
    if not uploaded_files:
        return None, "Tidak ada file libur yang diunggah."

    # Pemetaan nama bulan ke angka untuk pengurutan
    month_mapping = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    df_list = []
    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            header_row = 0
            if file_extension == '.xlsx':
                df_raw = pd.read_excel(uploaded_file, header=header_row)
            elif file_extension == '.csv':
                df_raw = pd.read_csv(uploaded_file, header=header_row)
            else:
                raise ValueError(f"Tipe file tidak didukung: {uploaded_file.name}")
            
            df_raw.dropna(how='all', inplace=True)
            df_raw.reset_index(drop=True, inplace=True)
            df_raw.columns = [str(col).strip() for col in df_raw.columns]

            bulan_col = next((col for col in df_raw.columns if 'bulan' in str(col).lower()), None)
            tahun_col = next((col for col in df_raw.columns if 'tahun' in str(col).lower()), None)
            libur_nasional_col = next((col for col in df_raw.columns if 'libur nasional' in str(col).lower()), None)
            cuti_bersama_col = next((col for col in df_raw.columns if 'cuti bersama' in str(col).lower()), None)

            if not all([bulan_col, tahun_col, libur_nasional_col, cuti_bersama_col]):
                raise ValueError(f"Tidak dapat menemukan semua kolom yang diperlukan di {uploaded_file.name}")
            
            df = df_raw[[bulan_col, tahun_col, libur_nasional_col, cuti_bersama_col]].copy()
            df.rename(columns={
                bulan_col: 'Bulan',
                tahun_col: 'Tahun',
                libur_nasional_col: 'Libur Nasional',
                cuti_bersama_col: 'Cuti Bersama'
            }, inplace=True)

            df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
            df['Bulan'] = df['Bulan'].str.strip()
            df.dropna(subset=['Tahun', 'Bulan'], inplace=True)
            
            # --- Perbaikan: Tambahkan kolom angka bulan untuk pengurutan yang benar ---
            df['Bulan_Angka'] = df['Bulan'].apply(lambda x: month_mapping.get(x, 0))
            # --- Akhir Perbaikan ---

            df_list.append(df)
            
        except Exception as e:
            st.error(f"ERROR saat membaca file {uploaded_file.name}: {e}")
            return None, f"ERROR saat membaca file {uploaded_file.name}: {e}"

    if not df_list:
        return None, "Tidak ada data yang valid untuk digabungkan."
    
    df_combined = pd.concat(df_list, ignore_index=True)
    # --- Perbaikan: Urutkan berdasarkan Tahun dan Bulan_Angka (kronologis) ---
    df_combined.sort_values(by=['Tahun', 'Bulan_Angka'], inplace=True)
    # --- Akhir Perbaikan ---
    df_combined.reset_index(drop=True, inplace=True)
    
    # Hapus kolom Bulan_Angka setelah pengurutan selesai
    df_combined.drop(columns='Bulan_Angka', inplace=True)

    return df_combined, None

def _process_and_combine_data(df_penumpang_train, df_libur_train, df_penumpang_test, df_libur_test):
    """
    Menggabungkan data penumpang dan libur untuk data training dan testing.
    """
    try:
        libur_bulanan_train = df_libur_train.groupby(['Bulan', 'Tahun']).agg(
            jumlah_libur_nasional=('Libur Nasional', lambda x: x.notna().sum()),
            jumlah_cuti_bersama=('Cuti Bersama', lambda x: x.notna().sum())
        ).reset_index()

        df_training = pd.merge(df_penumpang_train, libur_bulanan_train, on=['Bulan', 'Tahun'], how='left').fillna(0)
        
        libur_bulanan_test = df_libur_test.groupby(['Bulan', 'Tahun']).agg(
            jumlah_libur_nasional=('Libur Nasional', lambda x: x.notna().sum()),
            jumlah_cuti_bersama=('Cuti Bersama', lambda x: x.notna().sum())
        ).reset_index()

        df_testing = pd.merge(df_penumpang_test, libur_bulanan_test, on=['Bulan', 'Tahun'], how='left').fillna(0)
        
        df_training['Bulan ke-n'] = np.arange(1, len(df_training) + 1)
        start_month_test = len(df_training) + 1
        df_testing['Bulan ke-n'] = np.arange(start_month_test, start_month_test + len(df_testing))

        return df_training, df_testing, None
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None, f"ERROR saat menggabungkan data: {e}"

def latih_dan_evaluasi_regresi(df_training, df_testing):
    """
    Melatih model regresi berganda, membuat prediksi, dan menghitung metrik.
    """
    try:
        X_train = df_training[['Bulan ke-n', 'Total Jarak Tempuh Penumpang', 'Rata-rata Jarak Perjalanan Per penumpang', 
                               'jumlah_libur_nasional', 'jumlah_cuti_bersama']]
        y_train = df_training['Penumpang (000)'].values
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_training = model.predict(X_train)
        
        X_test = df_testing[['Bulan ke-n', 'Total Jarak Tempuh Penumpang', 'Rata-rata Jarak Perjalanan Per penumpang', 
                             'jumlah_libur_nasional', 'jumlah_cuti_bersama']]
        y_test = df_testing['Penumpang (000)'].values
        y_pred_testing = model.predict(X_test)
        
        mae_training = mean_absolute_error(y_train, y_pred_training)
        mape_training = np.mean(np.abs((y_train - y_pred_training) / np.where(y_train == 0, 1e-10, y_train))) * 100
        
        mae_testing = mean_absolute_error(y_test, y_pred_testing)
        mape_testing = np.mean(np.abs((y_test - y_pred_testing) / np.where(y_test == 0, 1e-10, y_test))) * 100

        results = {
            'model': model,
            'y_pred_training': y_pred_training,
            'y_pred_testing': y_pred_testing,
            'mae_training': mae_training,
            'mape_training': mape_training,
            'mae_testing': mae_testing,
            'mape_testing': mape_testing,
            'features': X_train.columns.tolist()
        }
        
        return results, None
    except Exception as e:
        print(f"ERROR: {e}")
        return None, f"ERROR saat melatih atau mengevaluasi model: {e}"

# --- Page Content Functions ---
def show_home():
    """Halaman utama aplikasi."""
    st.title("Selamat Datang!")
    st.write(
        """
        <div style="font-size: 16px; line-height: 1.6;">
            <p>Aplikasi ini dirancang untuk melakukan prediksi jumlah penumpang KRL Commuter Line Jabodetabek menggunakan <b>Algoritma Regresi Linier Berganda</b>. Sebelum melangkah ke tahap pemodelan, dataset Anda harus melewati beberapa <b>Uji Asumsi Klasik</b>.</p>
            <p>Untuk memulai, silakan unggah file dataset Anda di menu "Unggah Data".</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("📚 Ketentuan Dataset")
        st.write("Untuk memastikan aplikasi berjalan dengan baik, pastikan dataset Anda memenuhi ketentuan berikut:")
        
        st.markdown(
            """
            - **Format File**: File harus dalam format **Excel (.xlsx)** atau **CSV (.csv)**.
            - **Data Penumpang**: 
              - Data harus dalam format **"wide"** (bulan sebagai baris).
              - Baris judul utama tabel (`Bulan`, `Penumpang (000)`, dsb.) harus berada di **baris ke-4** file Excel.
            - **Data Hari Libur**: 
              - Data harus dalam format **"long"** (tanggal sebagai baris).
              - Baris judul tabel (`Bulan`, `Tahun`, `Libur Nasional`, `Cuti Bersama`) harus berada di **baris pertama** file Excel.
            """
        )

    with st.container(border=True):
        st.subheader("🔗 Sumber Data")
        st.write("Anda bisa mendapatkan data yang diperlukan dari sumber-sumber berikut:")
        
        col_sumber1, col_sumber2 = st.columns(2)
        
        with col_sumber1:
            st.markdown("##### 1. Data Jumlah Penumpang")
            st.markdown(
                """
                Data statistik jumlah penumpang KRL Commuter Line secara bulanan dapat diperoleh dari website resmi **Badan Pusat Statistik (BPS)**.
                - **Sumber**: [BPS, Statistik Transportasi](https://www.bps.go.id/subject/17/transportasi.html)
                """
            )
        
        with col_sumber2:
            st.markdown("##### 2. Data Hari Libur Nasional & Cuti Bersama")
            st.markdown(
                """
                Informasi hari libur nasional dan cuti bersama di Indonesia ditetapkan melalui **Surat Keputusan Bersama (SKB) 3 Menteri**.
                - **Sumber**: [SKB 3 Menteri (Pencarian Google)](https://www.google.com/search?q=SKB+3+Menteri+cuti+bersama)
                """
            )
    
    st.markdown("---")
    st.subheader("Uji Asumsi Klasik yang Harus Dipenuhi")
    st.write("Berikut adalah penjelasan singkat mengenai uji asumsi klasik yang penting dalam analisis regresi:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("##### 1. Normalitas")
            st.markdown(
                """
                Asumsi ini mengasumsikan bahwa residual (perbedaan antara nilai yang diamati dan nilai yang diprediksi) terdistribusi secara normal.
                """
            )
            
    with col2:
        with st.container(border=True):
            st.markdown("##### 2. Homoskedastisitas")
            st.markdown(
                """
                Ragam dari residual harus konsisten di seluruh tingkat variabel penjelas. Scatterplot dari residual versus nilai prediksi tidak boleh menampilkan pola yang terlihat.
                """
            )
    
    with col3:
        with st.container(border=True):
            st.markdown("##### 3. Non-Multikolinieritas")
            st.markdown(
                """
                Multikolinieritas terjadi ketika variabel independen memiliki korelasi yang tinggi satu sama lain. Dapat dideteksi dengan nilai VIF (Variance Inflation Factor).
                """
            )

def show_upload():
    """Menampilkan konten untuk halaman unggah data."""
    st.title("📁 Unggah Data Excel/CSV")
    st.info("Silakan unggah file data Anda: dua file penumpang dan dua file hari libur. Anda bisa mengunggah lebih dari satu file untuk setiap kategori.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Training")
            uploaded_penumpang_training = st.file_uploader(
                "Pilih file data penumpang (Training)",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
                key="penumpang_training"
            )
            uploaded_libur_training = st.file_uploader(
                "Pilih file data hari libur (Training)",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
                key="libur_training"
            )
        with col2:
            st.subheader("Data Testing")
            uploaded_penumpang_testing = st.file_uploader(
                "Pilih file data penumpang (Testing)",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
                key="penumpang_testing"
            )
            uploaded_libur_testing = st.file_uploader(
                "Pilih file data hari libur (Testing)",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
                key="libur_testing"
            )
    
    st.markdown("---")
    
    if st.button("Proses dan Simpan Data"):
        if uploaded_penumpang_training and uploaded_libur_training and uploaded_penumpang_testing and uploaded_libur_testing:
            
            # --- MODIFIKASI: Menambahkan validasi file tumpang tindih ---
            train_files = {file.name for file in uploaded_penumpang_training} | {file.name for file in uploaded_libur_training}
            test_files = {file.name for file in uploaded_penumpang_testing} | {file.name for file in uploaded_libur_testing}
            
            overlapping_files = train_files.intersection(test_files)
            
            if overlapping_files:
                st.error(f"❌ TERDETEKSI FILE GANDA: File berikut muncul di data training dan testing: {', '.join(overlapping_files)}. Harap unggah file yang berbeda untuk training dan testing.")
                return
            # --- Akhir modifikasi ---
            
            with st.spinner('Memproses data...'):
                df_penumpang_train, error_p_train = _read_penumpang_file(uploaded_penumpang_training)
                df_libur_train, error_l_train = _read_libur_file(uploaded_libur_training)
                df_penumpang_test, error_p_test = _read_penumpang_file(uploaded_penumpang_testing)
                df_libur_test, error_l_test = _read_libur_file(uploaded_libur_testing)

                if any([error_p_train, error_l_train, error_p_test, error_l_test]):
                    st.error("Terjadi kesalahan saat membaca file. Mohon periksa terminal untuk detail.")
                    return
                
                df_training, df_testing, error_combine = _process_and_combine_data(
                    df_penumpang_train, df_libur_train, df_penumpang_test, df_libur_test
                )

                if error_combine:
                    st.error(f"Terjadi kesalahan saat menggabungkan data: {error_combine}")
                    return
                
                results, error_model = latih_dan_evaluasi_regresi(df_training, df_testing)
                if error_model:
                    st.error(f"Gagal melatih model: {error_model}")
                    return
                
                # --- MODIFIKASI: Hapus panggilan predict_5_years dari sini ---
                
                # Simpan data dan hasil di session_state
                st.session_state.df_penumpang_train = df_penumpang_train
                st.session_state.df_libur_train = df_libur_train
                st.session_state.df_penumpang_test = df_penumpang_test
                st.session_state.df_libur_test = df_libur_test
                st.session_state.df_training = df_training
                st.session_state.df_testing = df_testing
                st.session_state.model_results = results
                st.session_state.data_loaded = True
                
                st.success("Data berhasil diproses dan disimpan!")
                st.balloons()
                st.session_state.page = 'show_data'
                st.rerun()
        else:
            st.warning("Mohon unggah keempat file yang diperlukan untuk melanjutkan.")

def show_data():
    """Menampilkan konten untuk halaman pratinjau data."""
    st.title("🗂️ Pratinjau Data")
    st.write("Verifikasi data yang telah diunggah dan diproses.")

    if st.session_state.data_loaded:
        with st.expander("Tampilkan Data Mentah", expanded=False):
            st.subheader("Data Penumpang Training (Mentah)")
            # --- MODIFIKASI: Format hanya kolom numerik yang relevan ---
            df_to_display = st.session_state.df_penumpang_train.copy()
            st.dataframe(df_to_display.style.format({
                'Tahun': '{:.0f}', # Tahun tidak diformat
                'Penumpang (000)': lambda x: _format_indonesian_numeric(x, 0),
                'Total Jarak Tempuh Penumpang': lambda x: _format_indonesian_numeric(x, 0),
                'Rata-rata Jarak Perjalanan Per penumpang': lambda x: _format_indonesian_numeric(x, 2),
            }))
        
            st.subheader("Data Libur Training (Mentah)")
            st.dataframe(st.session_state.df_libur_train)

            st.subheader("Data Penumpang Testing (Mentah)")
            # --- MODIFIKASI: Format hanya kolom numerik yang relevan ---
            df_to_display = st.session_state.df_penumpang_test.copy()
            st.dataframe(df_to_display.style.format({
                'Tahun': '{:.0f}', # Tahun tidak diformat
                'Penumpang (000)': lambda x: _format_indonesian_numeric(x, 0),
                'Total Jarak Tempuh Penumpang': lambda x: _format_indonesian_numeric(x, 0),
                'Rata-rata Jarak Perjalanan Per penumpang': lambda x: _format_indonesian_numeric(x, 2),
            }))

            st.subheader("Data Libur Testing (Mentah)")
            st.dataframe(st.session_state.df_libur_test)

        with st.expander("Tampilkan Tabel Data Regresi", expanded=True):
            st.subheader("Tabel Data Regresi")
            st.write("Tabel ini menampilkan data yang sudah diolah dan siap untuk digunakan dalam model regresi.")
            
            df_regr_train = st.session_state.df_training.copy()
            df_regr_train = df_regr_train[[
                'Penumpang (000)', 'Bulan ke-n', 'Total Jarak Tempuh Penumpang',
                'Rata-rata Jarak Perjalanan Per penumpang', 'jumlah_libur_nasional', 'jumlah_cuti_bersama'
            ]]
            df_regr_train.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5']
            
            st.markdown("##### Data Training")
            # --- MODIFIKASI: Format hanya kolom yang relevan ---
            st.dataframe(df_regr_train.style.format({
                'Y': lambda x: _format_indonesian_numeric(x, 0),
                'X1': '{:.0f}', # Tidak diformat
                'X2': lambda x: _format_indonesian_numeric(x, 0),
                'X3': lambda x: _format_indonesian_numeric(x, 2),
                'X4': lambda x: _format_indonesian_numeric(x, 0),
                'X5': lambda x: _format_indonesian_numeric(x, 0),
            }))
            
            df_regr_test = st.session_state.df_testing.copy()
            df_regr_test = df_regr_test[[
                'Penumpang (000)', 'Bulan ke-n', 'Total Jarak Tempuh Penumpang',
                'Rata-rata Jarak Perjalanan Per penumpang', 'jumlah_libur_nasional', 'jumlah_cuti_bersama'
            ]]
            df_regr_test.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5']
            
            st.markdown("##### Data Testing")
            # --- MODIFIKASI: Format hanya kolom yang relevan ---
            st.dataframe(df_regr_test.style.format({
                'Y': lambda x: _format_indonesian_numeric(x, 0),
                'X1': '{:.0f}', # Tidak diformat
                'X2': lambda x: _format_indonesian_numeric(x, 0),
                'X3': lambda x: _format_indonesian_numeric(x, 2),
                'X4': lambda x: _format_indonesian_numeric(x, 0),
                'X5': lambda x: _format_indonesian_numeric(x, 0),
            }))
        
        st.markdown("---")
        st.info("Data siap untuk dianalisis. Silakan lanjut ke menu Analisis Data.")
    else:
        st.warning("Tidak ada data yang tersedia untuk ditampilkan. Silakan unggah data terlebih dahulu.")

def show_data_analysis():
    """Menampilkan konten untuk halaman Analisis Data."""
    st.title("📊 Analisis Data & Uji Asumsi Klasik")
    st.write("Visualisasi dan statistik data untuk mengevaluasi dataset sebelum pemodelan.")
    
    chart_type = st.session_state.chart_type_option
    
    if 'df_training' in st.session_state and st.session_state.data_loaded:
        df_training = st.session_state.df_training
        
        with st.expander("Ringkasan Statistik", expanded=True):
            st.subheader("📋 Ringkasan Statistik Data Training")
            st.write("Berikut adalah ringkasan statistik dari data training dengan format yang lebih sederhana.")
            df_desc = df_training.describe()
            styled_df_desc = df_desc.style.format(lambda x: _format_indonesian_numeric(x, 0))
            st.dataframe(styled_df_desc.set_properties(**{'background-color': '#191e24', 'color': 'white'}))
        
        with st.expander("Visualisasi Tren", expanded=True):
            st.subheader("📈 Visualisasi Tren Jumlah Penumpang")
            st.write("Grafik ini menunjukkan tren jumlah penumpang sepanjang periode data yang diunggah (training dan testing).")
            df_combined_all = pd.concat([st.session_state.df_training, st.session_state.df_testing], ignore_index=True)
            chart_data = pd.DataFrame({
                'Bulan ke-n': df_combined_all['Bulan ke-n'],
                'Jumlah Penumpang': df_combined_all['Penumpang (000)']
            })
            # --- MODIFIKASI: Pilihan grafik disesuaikan dengan input sidebar ---
            if chart_type == 'Garis':
                st.line_chart(chart_data, x='Bulan ke-n', y='Jumlah Penumpang')
            elif chart_type == 'Batang':
                st.bar_chart(chart_data, x='Bulan ke-n', y='Jumlah Penumpang')
            # --- Akhir Modifikasi ---

        with st.expander("Korelasi Antar Variabel", expanded=True):
            st.subheader("📈 Korelasi Antar Variabel")
            st.write("Matriks korelasi mengukur hubungan linier antar variabel. Nilai yang mendekati 1 atau -1 menunjukkan korelasi yang kuat.")
            df_corr = df_training[['Bulan ke-n', 'Total Jarak Tempuh Penumpang', 'Rata-rata Jarak Perjalanan Per penumpang', 'jumlah_libur_nasional', 'jumlah_cuti_bersama', 'Penumpang (000)']]
            corr_matrix = df_corr.corr()
            renamed_columns = {col: _wrap_header_text(col) for col in corr_matrix.columns}
            renamed_corr_matrix = corr_matrix.rename(columns=renamed_columns, index=renamed_columns)
            st.dataframe(renamed_corr_matrix.style.background_gradient(cmap='RdYlBu', vmin=-1, vmax=1).format(lambda x: _format_indonesian_numeric(x, 2)))
            st.info("""
                **Kesimpulan**:
                - Nilai korelasi yang mendekati 1 (seperti antara `Bulan ke-n` dan `Penumpang (000)`) menunjukkan hubungan positif yang kuat.
                - Nilai yang mendekati -1 (seperti antara `jumlah_libur_nasional` dan `Penumpang (000)`) menunjukkan hubungan negatif yang kuat.
                - Nilai yang mendekati 0 menunjukkan tidak ada hubungan linier yang kuat.
            """)

        with st.expander("Hasil Pemodelan OLS & Uji Asumsi Klasik", expanded=True):
            st.subheader("📝 Hasil Pemodelan OLS (Data Training)")
            try:
                X = df_training[['Bulan ke-n', 'Total Jarak Tempuh Penumpang', 'Rata-rata Jarak Perjalanan Per penumpang', 'jumlah_libur_nasional', 'jumlah_cuti_bersama']]
                y = df_training['Penumpang (000)']
                X_with_const = sm.add_constant(X)
                model_ols = sm.OLS(y, X_with_const).fit()
                
                st.markdown("### 1. Ringkasan Model")
                summary_metrics_data = {
                    "Metrik": ["R-squared", "Adj. R-squared", "F-statistic", "Prob (F-statistic)"],
                    "Nilai": [model_ols.rsquared, model_ols.rsquared_adj, model_ols.fvalue, model_ols.f_pvalue],
                    "Penjelasan": [
                        "Besarnya keragaman pada variabel respon yang dapat dijelaskan oleh variabel penjelas.",
                        "R-squared yang disesuaikan dengan jumlah variabel penjelas.",
                        "Mengukur signifikansi model secara keseluruhan.",
                        "Nilai p-value dari uji F. Jika lebih kecil dari 0.05, model signifikan."
                    ]
                }
                summary_metrics_df = pd.DataFrame(summary_metrics_data)
                st.dataframe(summary_metrics_df.style.format({"Nilai": lambda x: _format_indonesian_numeric(x, 2)}))

                st.markdown("### 2. Koefisien Regresi")
                st.write("Tabel berikut menampilkan nilai koefisien, p-value, dan selang kepercayaan untuk setiap variabel penjelas.")
                df_coef = pd.DataFrame(model_ols.summary2().tables[1])
                st.dataframe(df_coef.style.format(lambda x: _format_indonesian_numeric(x, 2)))
                st.info("""
                    **Kesimpulan**:
                    - **P>|t|** (p-value) yang kurang dari 0.05 mengindikasikan bahwa variabel tersebut signifikan dalam memprediksi jumlah penumpang.
                    - **Coef** menunjukkan seberapa besar perubahan pada variabel terikat jika variabel penjelas berubah satu satuan.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 1. Asumsi Normalitas")
                    st.write("Plot Normal Q-Q untuk residual. Jika residual terdistribusi normal, titik-titik akan mengikuti garis lurus.")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    stats.probplot(model_ols.resid, dist="norm", plot=ax)
                    ax.set_title("Normal Q-Q Plot Residual", color='white')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("##### 2. Asumsi Homoskedastisitas")
                    st.write("Plot residual vs fitted value. Sebaran titik yang acak menunjukkan asumsi homoskedastisitas terpenuhi.")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(model_ols.fittedvalues, model_ols.resid, color='#F63366', alpha=0.7)
                    ax.axhline(y=0, color='white', linestyle='--')
                    ax.set_title("Residual vs Fitted Value", color='white')
                    st.pyplot(fig)

            except Exception as e:
                st.warning(f"Tidak dapat menghasilkan ringkasan OLS. Pastikan data tidak memiliki varians nol. Error: {e}")

    else:
        st.warning("Data training belum diunggah. Silakan unggah data terlebih dahulu.")

def show_modeling_evaluation():
    """Menampilkan konten untuk halaman Modeling dan Evaluasi."""
    st.title("📈 Modeling & Evaluasi")
    st.write("Model regresi linier berganda dibangun dan dievaluasi untuk mengukur performanya.")
    
    if 'df_training' in st.session_state and 'df_testing' in st.session_state and 'model_results' in st.session_state and st.session_state.data_loaded:
        df_training = st.session_state.df_training
        df_testing = st.session_state.df_testing
        results = st.session_state.model_results
        model = results['model']

        with st.expander("Metrik Evaluasi", expanded=True):
            st.subheader("🎯 Metrik Evaluasi Model")
            st.write("Evaluasi model pada data training dan testing.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="MAE (Training)", value=f"{_format_indonesian_numeric(results['mae_training'], 3)}")
                st.caption("Rata-rata kesalahan absolut pada data pelatihan.")
            with col2:
                st.metric(label="MAPE (Training)", value=f"{_format_indonesian_numeric(results['mape_training'], 2)}%")
                st.caption("Rata-rata kesalahan persentase pada data pelatihan.")
            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label="MAE (Testing)", value=f"{_format_indonesian_numeric(results['mae_testing'], 3)}")
                st.caption("Rata-rata kesalahan absolut pada data pengujian.")
            with col4:
                st.metric(label="MAPE (Testing)", value=f"{_format_indonesian_numeric(results['mape_testing'], 2)}%")
                st.caption("Rata-rata kesalahan persentase pada data pengujian.")
        
        with st.expander("Persamaan Model", expanded=True):
            st.subheader("📝 Persamaan Model")
            st.write("Persamaan regresi linier berganda yang dihasilkan.")
            
            intercept = model.intercept_
            coefs = model.coef_
            features = results['features']
            
            eq_string = f"Y' = {_format_indonesian_numeric(intercept, 2)}".replace(",", "@").replace(".", ",").replace("@", ".")
            for i, coef in enumerate(coefs):
                sign = '+' if coef >= 0 else '-'
                eq_string += f" {sign} {_format_indonesian_numeric(abs(coef), 2)}".replace(",", "@").replace(".", ",").replace("@", ".") + f" (X_{i+1})"
            
            st.markdown(f"Persamaan model: `{eq_string}`")

            st.write("Di mana:")
            for i, feature in enumerate(features):
                st.write(f"$X_{i+1}$ = **{feature}**")
        
    else:
        st.warning("Data atau model belum tersedia. Silakan unggah data dan jalankan Modeling terlebih dahulu.")

def show_deployment():
    """Menampilkan konten untuk halaman Deployment."""
    st.title("🚀 Deployment")
    st.write("Hasil prediksi jumlah penumpang dan visualisasi akhir dari model.")
    
    # Menandai bahwa halaman deployment telah dikunjungi
    st.session_state.show_reco_questions = True

    if 'df_training' in st.session_state and 'df_testing' in st.session_state and 'model_results' in st.session_state and st.session_state.data_loaded:
        df_training = st.session_state.df_training
        df_testing = st.session_state.df_testing
        results = st.session_state.model_results

        # --- MODIFIKASI: Panggil fungsi prediksi 5 tahun di sini ---
        predict_5_years(results['model'], df_training)
        
        mape_testing_real = results['mape_testing']
        
        if mape_testing_real <= 10:
            accuracy_status = "Sangat Akurat (< 10%)"
            status_color = "#28a745"
            status_text = "Model ini memiliki tingkat akurasi prediksi yang sangat tinggi. Perbedaan rata-rata antara nilai prediksi dan aktual sangat kecil."
        elif mape_testing_real <= 20:
            accuracy_status = "Akurat (10% - 20%)"
            status_color = "#007bff"
            status_text = "Model ini memiliki tingkat akurasi prediksi yang akurat. Hasil prediksi model mendekati nilai aktual."
        elif mape_testing_real <= 50:
            accuracy_status = "Cukup Akurat (20% - 50%)"
            status_color = "#ffc107"
            status_text = "Model ini memiliki tingkat akurasi prediksi yang cukup akurat. Namun, ada potensi untuk meningkatkan performa model."
        else:
            accuracy_status = "Tidak Akurat (> 50%)"
            status_color = "#dc3545"
            status_text = "Model ini memiliki tingkat akurasi prediksi yang rendah. Diperlukan evaluasi ulang terhadap model atau data yang digunakan."

        with st.container(border=True):
            st.subheader("Tingkat Akurasi Model")
            st.markdown(f'<div style="font-size:20px; font-weight:bold; color:white; background-color:{status_color}; padding:10px; border-radius:5px; text-align:center;">Akurasi Prediksi: {accuracy_status}</div>', unsafe_allow_html=True)
            st.markdown("### Tabel Kategori Persentase Nilai MAPE")
            mape_category_df = pd.DataFrame({
                'Nilai': ['<10%', '10% - 20%', '20% - 50%', '>50%'],
                'Kategori Peramalan': ['Peramalan Sangat Akurat', 'Peramalan Akurat', 'Peramalan Cukup Akurat', 'Peramalan Tidak Akurat']
            })
            st.table(mape_category_df.style.set_properties(**{'background-color': '#191e24', 'color': 'white'}))
            st.write(f"Model ini memiliki nilai MAPE sebesar **{_format_indonesian_numeric(mape_testing_real, 2)}%** pada data testing, yang termasuk dalam kategori **{accuracy_status}**.")
            st.write(status_text)
        
        with st.expander("Hasil Prediksi", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Data Training", "Data Testing", "Prediksi 5 Tahun"])
            
            prediction_df_training_real = pd.DataFrame({
                'Bulan ke-n': df_training['Bulan ke-n'].values,
                'Bulan': df_training['Bulan'].values,
                'Y Aktual': df_training['Penumpang (000)'].values,
                'Y Prediksi': results['y_pred_training'].flatten(),
                'Selisih': np.abs(df_training['Penumpang (000)'].values - results['y_pred_training'].flatten())
            })
            
            with tab1:
                st.write("Tabel ini menampilkan perbandingan antara jumlah penumpang yang diprediksi model dengan data aktual pada periode training.")
                # --- MODIFIKASI: Format hanya kolom yang relevan ---
                styled_df = prediction_df_training_real.style.format({
                    'Y Aktual': lambda x: _format_indonesian_numeric(x, 0),
                    'Y Prediksi': lambda x: _format_indonesian_numeric(x, 0),
                    'Selisih': lambda x: _format_indonesian_numeric(x, 0),
                    'Bulan ke-n': '{:.0f}', # Tidak diformat
                })
                renamed_columns = {col: _wrap_header_text(col) for col in styled_df.columns}
                st.dataframe(styled_df.set_caption("Tabel Hasil Prediksi (Data Training)"))
                
            prediction_df_testing_real = pd.DataFrame({
                'Bulan ke-n': df_testing['Bulan ke-n'].values,
                'Bulan': df_testing['Bulan'].values,
                'Y Aktual': df_testing['Penumpang (000)'].values,
                'Y Prediksi': results['y_pred_testing'].flatten(),
                'Selisih': np.abs(df_testing['Penumpang (000)'].values - results['y_pred_testing'].flatten())
            })

            with tab2:
                st.write("Tabel ini menampilkan perbandingan antara jumlah penumpang yang diprediksi model dengan data aktual yang terjadi pada periode pengujian.")
                # --- MODIFIKASI: Format hanya kolom yang relevan ---
                styled_df = prediction_df_testing_real.style.format({
                    'Y Aktual': lambda x: _format_indonesian_numeric(x, 0),
                    'Y Prediksi': lambda x: _format_indonesian_numeric(x, 0),
                    'Selisih': lambda x: _format_indonesian_numeric(x, 0),
                    'Bulan ke-n': '{:.0f}', # Tidak diformat
                })
                renamed_columns = {col: _wrap_header_text(col) for col in styled_df.columns}
                st.dataframe(styled_df.set_caption("Tabel Hasil Prediksi (Data Testing)"))
            
            with tab3:
                st.write("Tabel ini menampilkan prediksi jumlah penumpang untuk 5 tahun ke depan.")
                df_future_display = st.session_state.df_future[['Bulan', 'Tahun', 'Penumpang (000)']].copy()
                # --- MODIFIKASI: Format hanya kolom numerik yang relevan ---
                styled_df = df_future_display.style.format({
                    'Tahun': '{:.0f}', # Tahun tidak diformat
                    'Penumpang (000)': lambda x: _format_indonesian_numeric(x, 0)
                })
                st.dataframe(styled_df)


        with st.expander("Visualisasi Tren dan Prediksi", expanded=True):
            st.subheader("Visualisasi Tren dan Prediksi")
            st.write("Grafik di bawah ini memvisualisasikan tren data historis dan perbandingan dengan hasil prediksi.")
            
            df_combined_training = pd.DataFrame({
                'Bulan ke-n': df_training['Bulan ke-n'],
                'Aktual Training': df_training['Penumpang (000)'],
                'Prediksi Training': results['y_pred_training']
            })
            df_combined_testing = pd.DataFrame({
                'Bulan ke-n': df_testing['Bulan ke-n'],
                'Aktual Testing': df_testing['Penumpang (000)'],
                'Prediksi Testing': results['y_pred_testing']
            })
            df_combined = pd.concat([df_combined_training, df_combined_testing], ignore_index=True)
            
            chart_type = st.session_state.chart_type_option
            if chart_type == 'Garis':
                st.line_chart(df_combined, x='Bulan ke-n', y=['Aktual Training', 'Prediksi Training', 'Aktual Testing', 'Prediksi Testing'])
            elif chart_type == 'Batang':
                df_combined_long = pd.melt(df_combined, id_vars=['Bulan ke-n'], var_name='Jenis Data', value_name='Jumlah Penumpang')
                
                chart = alt.Chart(df_combined_long).mark_bar().encode(
                    x=alt.X('Bulan ke-n:O', axis=alt.Axis(title='Bulan ke-n')),
                    y=alt.Y('Jumlah Penumpang:Q', title='Jumlah Penumpang'),
                    color=alt.Color('Jenis Data:N', legend=alt.Legend(title="Jenis Data")),
                    xOffset='Jenis Data:N',
                    tooltip=['Bulan ke-n', 'Jenis Data', 'Jumlah Penumpang']
                ).properties(
                    title="Grafik Tren dan Prediksi Jumlah Penumpang"
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
    else:
        st.warning("Data atau model belum tersedia. Silakan unggah data dan jalankan Modeling terlebih dahulu.")

# --- Main Application Logic ---
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Tambahkan session_state untuk mengontrol tampilan chatbot
    if 'show_chatbot' not in st.session_state:
        st.session_state.show_chatbot = False

    create_header()
    create_sidebar_menu()
    
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'upload':
        show_upload()
    elif st.session_state.page == 'show_data':
        show_data()
    elif st.session_state.page == 'data_analysis':
        show_data_analysis()
    elif st.session_state.page == 'modeling_evaluation':
        show_modeling_evaluation()
    elif st.session_state.page == 'deployment':
        show_deployment()
    elif st.session_state.page == 'chatbot':
        show_chatbot_page()
    
    create_footer()

if __name__ == "__main__":
    main()