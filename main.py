import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import StackingClassifier
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
    df = pd.read_csv("Hasil prepocessing.csv")
    df = df[['databersih', 'label']]
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

# Vectorisasi
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Base
svm_clf = SVC(kernel='linear', probability=True)
nb_clf = MultinomialNB()

# Hybrid Model: Stacking SVM + NB
stacking_model = StackingClassifier(
    estimators=[('svm', svm_clf), ('nb', nb_clf)],
    final_estimator=SVC(kernel='linear', probability=True),
    cv=3
)

# Training model hybrid
stacking_model.fit(X_train_vectorized, y_train)

# Input pengguna
databersih = st.text_area("📝 Masukkan Komentar", height=150)

if st.button("🔍 Prediksi"):
    if databersih.strip() == "":
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")
    else:
        teks_bersih = preprocess(databersih)
        vektor = vectorizer.transform([teks_bersih])

        if vektor.shape[0] == 1:
            prediksi = stacking_model.predict(vektor)[0]

            if prediksi == 'ujaran kebencian':
                st.error("🚨 Ini adalah Ujaran Kebencian!")
                st.markdown("<h2 style='color:red;'>❌ Ujaran Kebencian Terdeteksi</h2>", unsafe_allow_html=True)
            else:
                st.success("✅ Ini bukan ujaran kebencian.")
                st.markdown("<h2 style='color:green;'>👍 Aman, tidak terdeteksi ujaran kebencian</h2>", unsafe_allow_html=True)

            # Probabilitas prediksi
            proba = stacking_model.predict_proba(vektor)[0]
            proba_ujaran_kebencian = proba[1] * 100
            proba_bukan_ujaran_kebencian = proba[0] * 100

            st.markdown(f"<p>Probabilitas Ujaran Kebencian: <strong>{proba_ujaran_kebencian:.2f}%</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Probabilitas Bukan Ujaran Kebencian: <strong>{proba_bukan_ujaran_kebencian:.2f}%</strong></p>", unsafe_allow_html=True)

            # Akurasi dan F1-Score model
            y_test_pred = stacking_model.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, pos_label='ujaran kebencian')  # sesuaikan label jika perlu
            st.markdown(f"<p>📊 Akurasi Model pada Data Uji: {accuracy*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p>🎯 F1-Score Model pada Data Uji: <strong>{f1:.2f}</strong></p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Format input tidak sesuai. Coba lagi dengan komentar yang valid.")

# Footer
st.markdown(""" 
<hr>
<div style='text-align: center;'>
    <small>© 2025 - Sistem Deteksi Komentar Ujaran Kebencian (preprocessing ga sempurna (hybrid))</small>
</div>
""", unsafe_allow_html=True)
