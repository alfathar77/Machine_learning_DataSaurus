

#IMPORT LIBRARY

import numpy as np # for math. operations
import pandas as pd # for data manipulation operations
import matplotlib.pyplot as plt  # Import matplotlib's pyplot module for plotting
# import statsmodels.api as plt # for building of models
from sklearn.linear_model import LinearRegression # for building of machine learning models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, f1_score

import seaborn as sns # for advanced data visualizations
st.title('Machine Learning By Datasaurus')
sns.set() # activate
dataset = pd.read_excel ('AQIJakarta.xlsx')
dataset.head()

st.write(dataset['stasiun'])

data_drop = dataset.drop(['tanggal','stasiun','max','critical'],axis=1)

data_drop.describe(include='all')

print(data_drop)

data_drop.isnull().sum() #.sum() to see the number of missing in each variable

#False (0) is the data exist, but True (1) is data missing
#There are 2 variable have missing data value, that are pm10 (68), so2 (43), co (77), o3 (63), no2(66) from 4173 and 4345 (5%) total calculated data

# Mengganti nilai yang kosong/NaN dengan nilai 0
data_filled = data_drop.fillna(0)

st.write(data_filled)

data_filled.isnull().sum()

x = data_filled.iloc[:, :5].values #mengambil dari kolom 0-4
y = data_filled.iloc[:, 5].values #mengambil kolom pada indeks ke 5

print(x)

print(y)

# Persiapkan data pelatihan dan uji
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

# Inisialisasi model Naive Bayes
model = RandomForestClassifier()

# Latih model dengan data pelatihan
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
predictions = model.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, predictions)
print(f"Akurasi: {accuracy}")

# Berikut adalah contoh penggunaan Gaussian Naive Bayes untuk klasifikasi dan membuat confusion matrix dengan Python menggunakan library scikit-learn:

# Membuat model Gaussian Naive Bayes
RandomForestClassifier_model = RandomForestClassifier()
RandomForestClassifier_model.fit(X_train, y_train)

# Membuat prediksi
y_pred = RandomForestClassifier_model.predict(X_test)

print(y_pred)

# Membuat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

# Mengambil nilai dari confusion matrix
TP = conf_matrix[3][3]  # True Positive
TN = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1] + conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][3] + conf_matrix[3][0] + conf_matrix[3][1] + conf_matrix[3][2]  # True Negative
FP = conf_matrix[0][2] + conf_matrix[1][2] + conf_matrix[2][2]  # False Positive
FN = conf_matrix[2][3] + conf_matrix[3][2]  # False Negative

print(f"True Positive (TP): {TP}")
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")

# Menghitung Total Positif dan Total Negatif
total_positif = TP + FN
total_negatif = TN + FP

# Menghitung Persentase TP, TN, FP, FN
persentase_TP = (TP / total_positif) * 100
persentase_TN = (TN / total_negatif) * 100
persentase_FP = (FP / total_negatif) * 100
persentase_FN = (FN / total_positif) * 100

print(f"Persentase True Positive (TP): {persentase_TP:.2f}%")
print(f"Persentase True Negative (TN): {persentase_TN:.2f}%")
print(f"Persentase False Positive (FP): {persentase_FP:.2f}%")
print(f"Persentase False Negative (FN): {persentase_FN:.2f}%")

# Menampilkan laporan klasifikasi
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Menghitung jumlah Correctly Classified
correctly_classified = TP + TN

# Menghitung total jumlah data
total_data = sum(sum(row) for row in conf_matrix)

# Menghitung persentase Correctly Classified
persentase_correctly_classified = (correctly_classified / total_data) * 100

print(f"Persentase Correctly Classified: {persentase_correctly_classified:.2f}%")

# Menghitung jumlah Incorrectly Classified
incorrectly_classified = FP + FN

# Menghitung total jumlah data
total_data = sum(sum(row) for row in conf_matrix)

# Menghitung persentase Incorrectly Classified
persentase_incorrectly_classified = (incorrectly_classified / total_data) * 100

print(f"Persentase Incorrectly Classified: {persentase_incorrectly_classified:.2f}%")

# Sample data (replace with your actual data)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model (replace with your actual model)
model = RandomForestClassifier()

# Calculate accuracy for different percentages of training data
accuracy_scores = []
for i in np.arange(0.1, 1.0, 0.1):
    X_train_subset = X_train[:int(len(X_train) * i)]
    y_train_subset = y_train[:int(len(y_train) * i)]
    model.fit(X_train_subset, y_train_subset)
    y_pred = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plotting the performance graph
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0.1, 1.0, 0.1), accuracy_scores, marker='o', linestyle='--', color='b') # Changed accuracy_score to accuracy_scores
plt.title('Naive Bayes Classifier Performance')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

from datetime import datetime

# Fungsi untuk menghasilkan prediksi CO (contoh sederhana)
def predict_co(stasiun, tanggal):
    # Contoh data prediksi CO
    # Dalam aplikasi nyata, Anda akan menggunakan model prediksi yang lebih kompleks
    np.random.seed(0)  # Untuk reproduktifitas
    co_concentration = np.random.uniform(0, 10)  # Konsentrasi CO antara 0 dan 10 ppm
    return co_concentration


# Fungsi untuk menghasilkan prediksi CO (contoh sederhana)
def predict_co(stasiun, tanggal):
    # Contoh data prediksi CO
    np.random.seed(0)  # Untuk reproduktifitas
    co_concentration = np.random.uniform(0, 50)  # Konsentrasi CO antara 0 dan 50 ppm
    return co_concentration


# Judul aplikasi
st.title('Prediksi Konsentrasi Karbon Monoksida (CO) di Stasiun DKI')

# Input untuk nama stasiun
station = st.text_input('Masukkan Nama Stasiun:', '')

# Input untuk tanggal
date_input = st.date_input('Pilih Tanggal:', datetime.today())

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    if station:
        # Menghasilkan prediksi
        co_concentration = predict_co(station, date_input)
        
        # Menampilkan hasil prediksi
        st.write(f'**Prediksi Konsentrasi CO untuk Stasiun: {station} pada Tanggal: {date_input}**')
        st.write(f'Konsentrasi CO: {co_concentration:.2f} ppm')
    else:
        st.error('Silakan masukkan nama stasiun.')

import streamlit as st

def check_safety_level(co_concentration):
    if co_concentration <= 4:
        return "Baik (Good)"
    elif 5 <= co_concentration <= 9:
        return "Sedang (Moderate)"
    elif 10 <= co_concentration <= 35:
        return "Tidak Sehat (Unhealthy for Sensitive Groups)"
    else:
        return "Tidak Sehat (Unhealthy)"

# Judul aplikasi
st.title("Cek Tingkat Keamanan Konsentrasi CO")

# Input dari pengguna
co_concentration = st.number_input("Masukkan konsentrasi CO (ppm):", min_value=0.0)

# Tombol untuk memeriksa tingkat keamanan
if st.button("Cek Tingkat Keamanan"):
    result = check_safety_level(co_concentration)
    st.write(f"Tingkat Keamanan: {result}")

# Menambahkan footer
st.write('---')
st.write('Aplikasi ini menggunakan model prediksi konsentrasi CO sederhana.')
st.write(check_safety_level)
