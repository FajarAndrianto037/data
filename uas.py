import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


st.write("# Heart disease Prediction App")
st.sidebar.header('User Input Features')
data = pd.read_csv('https://raw.githubusercontent.com/FajarAndrianto037/data/main/Heart_Disease_Dataset.csv')
st.dataframe(data)


# Find the Key From Dictionary
def get_key(val, my_dict):
  for key, value in my_dict.items():
    if val == value:
      return key

label = {'Failed': 0, 'Completed': 1}

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.number_input('Enter your age (Umur): ')
    sex  = st.sidebar.selectbox('Sex (Jenis kelamin)',(0,1))
    cp = st.sidebar.selectbox('Chest pain type (Jenis Nyeri Data)',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure (Tekanan Darah): ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl (Kolestrol): ')
    fbs = st.sidebar.selectbox('Fasting blood sugar (Gula Darah)',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results (Hasil Elektrokadiografi): ')
    tha = st.sidebar.number_input('Maximum heart rate achieved (Detak Jantung Maksimum): ')
    exa = st.sidebar.selectbox('Exercise induced angina (Induksi Anggina): ',(0,1))
    old = st.sidebar.number_input('oldpeak (ST Depresion)')
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen (Slope): ')
    ca = st.sidebar.selectbox('number of major vessels (Nilai CA)',(0,1,2,3))
    thal = st.sidebar.selectbox('thal (Nilai Thal)',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal 
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

st.subheader('Data Penyakit')
st.write(data.target.value_counts())
st.write(sns.countplot(x="target", data=data))
st.pyplot(plt.show())

countNoDisease = len(data[data.target == 0])
countHaveDisease = len(data[data.target == 1])
st.write("Persentase pasien tidak memiliki penyakit jantung: {:.2f}%".format((countNoDisease / (len(data.target))*100)))
st.write("Persentase pasien memiliki penyakit jantung: {:.2f}%".format((countHaveDisease / (len(data.target))*100)))


def submit():
    st.subheader('model regression')
    y = data.target.values
    x_data = data.drop(['target'], axis = 1)
    # Normalize
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
    #transpose matrices
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    accuracies = {}

    lr = LogisticRegression()
    lr.fit(x_train.T,y_train.T)
    acc = lr.score(x_test.T,y_test.T)*100

    st.write("Test Accuracy {:.2f}%".format(acc))

    st.subheader('model KNN')
    from sklearn.neighbors import KNeighborsClassifier
    
    scoreList = []
    for i in range(1,20):
        knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
        knn2.fit(x_train.T, y_train.T)
        scoreList.append(knn2.score(x_test.T, y_test.T))

    acc = max(scoreList)*100
    accuracies['KNN'] = acc
    st.write("Maximum KNN Score is {:.2f}%".format(acc))

    st.subheader('Naive Bayes')
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(x_train.T, y_train.T)

    acc = nb.score(x_test.T,y_test.T)*100
    accuracies['Naive Bayes'] = acc
    st.write("Accuracy of Naive Bayes: {:.2f}%".format(acc))

# create button submit
submitted = st.sidebar.button("Submit")
if submitted:
    submit()
