import streamlit as st
import pickle
import numpy as np

import pickle
ohe_sex = pickle.load(open('ohe_sex.pkl', 'rb'))
ohe_embarked = pickle.load(open('ohe_embarked.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))

st.title("Passenger Survival Prediction")

Pclass = st.selectbox('Pclass', [1, 2, 3])
Name = st.text_input('Enter Your Name...')
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.number_input('Age', min_value=0.5, max_value=100.0, step=0.5)
SibSp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, step=1)
Parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, step=1)
Fare =  st.number_input('Fare', min_value=0.0, step=0.1)
Embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

if st.button('Predict Survival'):
    
    test_df_sex = ohe_sex.transform(np.array(Sex).reshape(1,1))
    test_df_embarked = ohe_embarked.transform(np.array(Embarked).reshape(1,1))
    # test_df_age = np.array(Age).reshape(1,1)
    
    input_array = np.array([Pclass, SibSp, Parch, Fare, Age], dtype=object).reshape(1, 5)
    
    df_transformed = np.concatenate((input_array, test_df_sex, test_df_embarked), axis=1)
    
    prediction = clf.predict(df_transformed)
    
    
    if prediction[0]==1:
        st.success(f"{Name}  has survived..")
    else:
        st.error("passenger was not survived...")