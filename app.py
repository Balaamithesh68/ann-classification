import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open ('one_hot.pkl','rb') as file:
    one_hot=pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)



st.title("Customer Churn Prediction")


#user input

geography=st.selectbox('Geography',one_hot.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit_score')
salary=st.number_input('Estimated Salary')
tenure=st.slider('tenure',0,10)
products=st.slider('Number of Products',0,4)
cr_card=st.selectbox('Has Credit Card',[0,1])
active=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[products],
    'HasCrCard':[cr_card],
    'IsActiveMember':[active],
    'EstimatedSalary':[salary]
})

geo_encoder = one_hot.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=one_hot.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

input_scal=scaler.transform(input_data)

predict_model=model.predict(input_scal)

probability=predict_model[0][0]

if probability >0.5:
    st.write("the customer is likely to churn")
else:
    st.write("the customer is not likely to churn")