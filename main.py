import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Ujaran Kebencian", page_icon="💬", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 Deteksi Ujaran Kebencian</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan komentar untuk mengetahui apakah mengandung ujaran kebencian.</p>", unsafe_allow_html=True)

# Fungsi preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data dari file CSV
@st.cache_data
def load_data():
    df = pd.read_csv("Hasil prepocessing.csv")  # Ganti nama file jika berbeda
    df = df[['databersih', 'label']]  # Pastikan kolom sesuai
    df['databersih'] = df['databersih'].astype(str).apply(preprocess)
    return df

# Load dataset
try:
    data = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# Split data
X = data['databersih']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize teks
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Latih model SVM dengan probabilitas
svm_model = SVC(probability=True)  # Pastikan probability=True
param_grid = {
    'C': [1],
    'kernel': ['linear'],
    'gamma': ['scale']
}
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train_vectorized, y_train)
best_model = grid_search.best_estimator_

# Input dari pengguna
databersih = st.text_area("📝 Masukkan Komentar", height=150)

if st.button("🔍 Prediksi"):
    if databersih.strip() == "":
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")
    else:
        teks_bersih = preprocess(databersih)
        vektor = vectorizer.transform([teks_bersih])  # Pastikan input dalam bentuk 2D array

        # Perbaikan: Memastikan vektor memiliki dimensi yang benar untuk predict_proba
        if vektor.shape[0] == 1:
            prediksi = best_model.predict(vektor)[0]

            if prediksi == 'ujaran kebencian':
                st.error("🚨 Ini adalah Ujaran Kebencian!")
                st.markdown("<h2 style='color:red;'>❌ Ujaran Kebencian Terdeteksi</h2>", unsafe_allow_html=True)
            else:
                st.success("✅ Ini bukan ujaran kebencian.")
                st.markdown("<h2 style='color:green;'>👍 Aman, tidak terdeteksi ujaran kebencian</h2>", unsafe_allow_html=True)

            # Prediksi probabilitas
            proba = best_model.predict_proba(vektor)[0]
            proba_ujaran_kebencian = proba[1] * 100  # Probabilitas untuk 'ujaran kebencian'
            proba_bukan_ujaran_kebencian = proba[0] * 100  # Probabilitas untuk 'bukan ujaran kebencian'

            st.markdown(f"<p>Probabilitas Ujaran Kebencian: <strong>{proba_ujaran_kebencian:.2f}%</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Probabilitas Bukan Ujaran Kebencian: <strong>{proba_bukan_ujaran_kebencian:.2f}%</strong></p>", unsafe_allow_html=True)

            # Menampilkan akurasi model pada data uji
            y_test_pred = best_model.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, y_test_pred)
            st.markdown(f"<p>📊 Akurasi Model pada Data Uji: {accuracy*100:.2f}%</p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Format input tidak sesuai. Coba lagi dengan komentar yang valid.")


# Footer
st.markdown(""" 
<hr>
<div style='text-align: center;'>
    <small>© 2025 - Sistem Deteksi Komentar ujaran kebencian (preprocessing ga sempurna)</small>
</div>
""", unsafe_allow_html=True)
