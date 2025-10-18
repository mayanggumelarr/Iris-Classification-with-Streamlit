import streamlit as st
import pickle
import numpy as np

# Import model
with open('rf_model', 'rb') as file_model:
    rf_model = pickle.load(file_model)

# Buat title
st.title("Klasifikasi Spesies Bunga Iris")
st.write("Masukkan hasil pengukuran untuk klasifikasi!")

# Identifikasi inputan
Sepal_Length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
Sepal_Width= st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
Petal_Length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
Petal_Width = st.slider("Petal Width (cm)", min_value=1.0, max_value=2.5, step=0.1)

if st.button("Prediksi"):
    features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
    prediction = rf_model.predict(features)
    result = prediction[0]
    st.success(result)
    if result == 0 :
        st.write("Spesies Iris-Setosa")
    elif result == 1:
        st.write("Spesies Iris-Versicolor")
    else:
        st.write("Spesies Iris-Virginica")