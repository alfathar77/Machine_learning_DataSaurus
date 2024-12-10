
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_excel('AQIJakarta.xlsx')

data = data.drop(columns=['tanggal'])

X = data.drop('no2', axis=1)  # Fitur
y = data['categori']                # Target

X = pd.get_dummies(X)

# Menangani nilai yang hilang
X.fillna(X.mean(), inplace=True)

#Membagi Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Membangun Model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Melakukan Prediksi
y_pred_nb = nb_model.predict(X_test)

# Membuat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_nb)

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
class_report = classification_report(y_test, y_pred_nb)
print("\nClassification Report:")
print(class_report)

# Evaluasi Model Naive Bayes
print("Naive Bayes Model:")
print("Akurasi:", accuracy_score(y_test, y_pred_nb))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred_nb))
print("Matriks Kebingungan:\n", confusion_matrix(y_test, y_pred_nb))

# Membangun Model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)  # Melatih model

# Melakukan prediksi pada data uji
y_pred = nb_model.predict(X_test)

# Menampilkan laporan klasifikasi
report = classification_report(y_test, y_pred)
print(report)