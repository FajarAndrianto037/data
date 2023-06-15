#Modul Library
import streamlit as st
import numpy as np
import pandas as pd

#Modul library Metode 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# #modul library data testing dan training
from sklearn.model_selection import train_test_split

# #modul library score tingkat akurasi
from sklearn.metrics import accuracy_score

def load_dataset():
	url = 'https://raw.githubusercontent.com/Purnomo-Ribut/projek-uas/main/online_classroom_data.csv'
	df = pd.read_csv(url,  header='infer', index_col=False)
	df = df.replace(",",".",regex=True)
	df = df.drop(columns=["Unnamed: 0"])
	return df

st.title('Prediksi E-LEARNING STUDENTS REACTIONS Dengan Naive Bayes ')
st.write ("""Purnomo Ribut | 200411100156""")
dataset, modelling, implementasi = st.tabs(["Dataset", "Modelling", "Implementasi"])

with dataset:
	st.write(""" link dataset :  https://www.kaggle.com/datasets/marlonferrari/elearning-student-reactions""")
	(""" 
		Dataset ini dikompilasi setelah 4 bulan Kelas Pengantar Algoritma di Universitas Brasil.

		Sistem penilaian tradisional diadopsi untuk evaluasi kinerja siswa, dan, pada saat yang sama, lingkungan online memungkinkan siswa berbagi posting, jawaban, dan mengklasifikasikan produksi dengan reaksi berbasis emoji.

		Kelas ini berbasis proyek dan evaluasi keterampilan mengikuti apa yang disebut "Keterampilan Abad ke-21", dalam skala dari 0 hingga 10 setiap Keterampilan: 

		* Keterampilan Berpikir Kritis dan Pemecahan Masalah - disebut sebagai SK1

		* Keterampilan Criativity dan Inovation - dinamai SK2

		* Keterampilan Konstan dan Belajar Mandiri - dinamai SK3

		* Keterampilan Kolaborasi dan Pengarahan Diri - dinamai SK4

		* Tanggung Jawab Sosial dan Budaya - dinamai SK5

		* total_post  merupakan jumlah posting yang dibuat oleh siswa
		
		* helpful_post merupakan Jumlah Reaksi "membantu" yang diterima oleh siswa

		* nice_code_post merupakan Jumlah Reaksi "kode bagus" yang diterima oleh siswa

		* collaborative_post merupakan Jumlah Reaksi "kolaboratif" yang diterima oleh siswa

		* confused_post merupakan Jumlah Reaksi "bingung" yang diterima oleh siswa

		* creative_post merupakan Jumlah Reaksi "kreatif" yang diterima oleh siswa

		* bad_post merupakan Jumlah Reaksi "buruk" yang diterima oleh siswa

		* amazing post merupakan Jumlah Reaksi "luar biasa" yang diterima oleh siswa

		* timeonline merupakan Jumlah waktu (dalam detik) yang dihabiskan oleh siswa di lingkungan

		* approved nerupakan Apakah siswa disetujui di Kelas? 1 YA - 0 TIDAK.
		""")
	st.dataframe(load_dataset())
with modelling:
	def tambah_input(nama_metode): 
		inputan=dict()
		if nama_metode=="K-Nearst Neighbors" :
			K= st.slider("K", 1,15)
			inputan["K"]=K
		elif nama_metode=="Decission Tree":
			kriteria =st.selectbox("pilih kriteria",("entropy", "gini"))
			inputan["kriteria"]= kriteria
			max_depth =st.slider ("max depth",2,15)
			inputan["max_depth"]=max_depth
		return inputan

	def pilih_kelas(nama_metode, inputan):
		data = load_dataset()
		# model=None
		if nama_metode   == "K-Nearst Neighbors":
			model =KNeighborsClassifier(n_neighbors = inputan["K"])
		elif nama_metode == "Decission Tree":
			model =DecisionTreeClassifier(criterion= inputan["kriteria"], max_depth= inputan["max_depth"])
		elif nama_metode == "Naive Baiyes GaussianNB":
			model = GaussianNB()

		# #fitur
		X = data.iloc[:, :-1].astype(float)
		# #hasil
		y = data.Approved.astype(int)

		#Proses Klasifikasi
		#split unnormalized data
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
		model.fit(X_train, y_train)
		st.write("Accuracy  = ", model.score(X_test, y_test))
		st.session_state["model"] = nama_metode
		st.session_state["inputan"] = inputan


	metode = st.selectbox("Hasil metode akurasi berdasarkan dataset menggunakan:",('Naive Baiyes GaussianNB', 'K-Nearst Neighbors', 'Decission Tree'))
	inputan = tambah_input(metode)
	pilih_kelas(metode, inputan)
	
	

with implementasi:
	total_posts = st.number_input('total_post')
	helpful_post = st.number_input('helpful_post')
	nice_code_post = st.number_input('nice_code_post')
	collaborative_post = st.number_input('collaborative_post')
	confused_post = st.number_input('confused_post')
	creative_post = st.number_input('creative_post')
	bad_post = st.number_input('bad_post')
	amazing_post = st.number_input('amazing_post')
	timeonline = st.number_input('timeonline')
	sk1_classroom = st.number_input('sk1_classroom')
	sk2_classroom = st.number_input('sk2_classroom')
	sk5_classroom = st.number_input('sk5_classroom')
	sk3_classroom = st.number_input('sk3_classroom')
	sk4_classroom = st.number_input('sk4_classroom')
	submitted = st.button("Submit")
	

	if submitted:
		if "model" in st.session_state:
			data = load_dataset()
			# model=None
			nama_metode = st.session_state.model
			inputan = st.session_state.inputan
			if nama_metode == "K-Nearst Neighbors":
				model =KNeighborsClassifier(n_neighbors = inputan["K"])
			elif nama_metode == "Decission Tree":
				model =DecisionTreeClassifier(criterion= inputan["kriteria"], max_depth= inputan["max_depth"])
			elif nama_metode == "Naive Baiyes GaussianNB":
				model = GaussianNB()	

			# #fitur
			X = data.iloc[:, :-1].astype(float)
			# #hasil
			y = data.Approved.astype(int)
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
			model.fit(X_train, y_train)

			# st.write("tets")
			# st.write(model)
			inputs = np.array([[total_posts ,helpful_post, nice_code_post, collaborative_post, confused_post, creative_post, bad_post, amazing_post, timeonline, sk1_classroom, sk2_classroom, sk5_classroom, sk3_classroom,sk4_classroom]])
			prediksi_reaksi = model.predict(inputs)

			if prediksi_reaksi	== 1:
				hasil_analisa = "Siswa disetujui di kelas"
			else:
				hasil_analisa = "Siswa tidak disetujui di kelas."
			st.write("hasil prediksi : " + str(prediksi_reaksi))
			st.success(hasil_analisa)


