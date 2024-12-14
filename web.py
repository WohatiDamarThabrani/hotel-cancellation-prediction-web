import streamlit as st
import joblib
import numpy as np
import pandas as pd



# Load model
model = joblib.load('./random_forest_model.pkl')  # Pastikan path sesuai

# Judul aplikasi
st.title("Prediksi Pembatalan Hotel")
st.write("Masukkan fitur untuk memprediksi apakah reservasi akan dibatalkan atau tidak.")

# Input pengguna
number_of_adults = st.number_input("Jumlah Dewasa", min_value=0, step=1, max_value=4)
number_of_children = st.number_input("Jumlah anak-anak", min_value=0, step=1, max_value=10)
number_of_week_nights = st.number_input("Jumlah malam yang dipesan saat weekdays", min_value=0, step=1, max_value=17)
type_of_meal = st.number_input("Pemilihan tipe meal, Input angka yang sesuai (meal plan 1 = 0, meal plan 2 = 1, meal plan 3 = 2, not selected = 4)", min_value=0, max_value=4)
car_parking_space = st.number_input("Apakah customer memesan tempat parkir mobil khusus (Ya = 1, Tidak = 0, )", min_value=0, max_value=1)
room_type = st.number_input("Tipe kamar yang dipesan, Inputkan angka yang sesuai dengan tipe kamar yang dipesan (Tipe kamar 1 = 0, Tipe kamar 2 = 1, Tipe kamar 3 = 2, Tipe kamar 4 = 3, Tipe kamar 5 = 4, Tipe kamar 6 = 5, Tipe kamar 7 = 6)", min_value=0, max_value=6)
lead_time = st.number_input("Jarak antara tanggal pemesanan dan tanggal check-in (dalam hari)", min_value=0, step=1, max_value=443)
market_segment_type = st.number_input("Jenis segmen pasar yang terkait dengan pemesanan, input angka yang sesuai (aviation = 0, complementary = 1, corporate = 2, offline = 3, online = 4)", min_value=0, step=1, max_value=4)
average_price = st.number_input("Rata-rata biaya per malam (dalam dollar)", min_value=0, step=1, max_value=540)
special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, step=1, max_value=5)
month = st.number_input("Bulan ketika pemesanan hotel dilakukan (January = 1, February = 2, dan seterusnya)", min_value=1, step=1, max_value=12)
year = st.number_input("Tahun ketika pemesanan hotel dilakukan", min_value=2015, step=1, max_value=2018)

# Prediksi
if st.button("Prediksi"):
    features = np.array([[number_of_adults, number_of_children, number_of_week_nights, type_of_meal, car_parking_space, room_type, lead_time, market_segment_type, average_price, special_requests, month, year]])
    features = pd.DataFrame(features, columns=['number_of_adults', 'number_of_children', 'number_of_week_nights', 'type_of_meal', 'car_parking_space', 'room_type', 'lead_time', 'market_segment_type', 'average_price', 'special_request', 'month', 'year'])
    prediction = model.predict(features)
    st.write("Hasil Prediksi:", "Dibatalkan" if prediction[0] == 1 else "Tidak Dibatalkan")
