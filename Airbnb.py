import numpy as np
import pandas as pd
import streamlit as st
import datetime
import pickle

st.header('AIRBNB TRAVEL DESTINATION PREDICTOR')

st.write('''
Imagine yourself as a touristor who visited the Airbnb website to book your holiday destination. \
This app uses statistical machine learning to predict, using ML, the top destination you might be interested to travel.
Please fill up your informations that you believe suits you the most, and notice your predictions!
''')


col1, col2 = st.columns(2)

gender = col1.selectbox("Select Gender",
                            ["Male", "Female", "Others", "Unkown"])

age = col1.slider("Set the age",
                    1, 100, step=1)

language = col2.selectbox("Select the language",['English','Spanish','French','German','Italian','Dutch','Portugese','Others'])

browser = col2.selectbox("Select the browser", 
                        ["Chrome","Safari","Firefox","IE","Other Browser"])

def model_pred(gender,age,language,browser):

    with open('C:/Users/admin/Desktop/Python_Programs/airbnb.pkl','rb') as file:
        Classification_model = pickle.load(file)
        
    input_ftrs = [age,3,6,6]
    
    if gender == "Unkown":
        input_ftrs+=[1,0,0,0] 
    elif gender == "Female":
        input_ftrs+=[0,1,0,0]
    elif gender == "Male":
        input_ftrs+=[0,0,1,0]
    else:
        input_ftrs+=[0,0,0,1]
        
    input_ftrs+=[1,0,0]
    
    if language == "English":
        input_ftrs+=[0]
    elif language == "Spanish":
        input_ftrs+=[92.25]
    elif language == "French":
        input_ftrs+=[92.06]
    elif language == "German":
        input_ftrs+=[72.61]
    elif language == "Italian":
        input_ftrs+=[89.4]
    elif language == "Dutch":
        input_ftrs+=[63.22]
    elif language == "Portugese":
        input_ftrs+=[95.45]
    else:
        input_ftrs+=[72.14]
         
    
    if language == "Dutch":
        input_ftrs+=[1,0,0,0,0,0,0,0]
    elif language == "English":
        input_ftrs+=[0,1,0,0,0,0,0,0]
    elif language == "French":
        input_ftrs+=[0,0,1,0,0,0,0,0]
    elif language == "German":
        input_ftrs+=[0,0,0,1,0,0,0,0]
    elif language == "Italian":
        input_ftrs+=[0,0,0,0,1,0,0,0]
    elif language == "Others":
        input_ftrs+=[0,0,0,0,0,1,0,0]
    elif language == "Portugese":
        input_ftrs+=[0,0,0,0,0,0,1,0]
    else:
        input_ftrs+=[0,0,0,0,0,0,0,1]
        
    input_ftrs+=[1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0]

    if browser == "Chrome":
        input_ftrs+=[1,0,0,0,0]
    elif browser == "Firefox":
        input_ftrs+=[0,1,0,0,0]
    elif browser == "IE":
        input_ftrs+=[0,0,1,0,0]
    elif browser == "Other Browser":
        input_ftrs+=[0,0,0,1,0]
    else:
        input_ftrs+=[0,0,0,0,1]
        
    input_ftrs=[input_ftrs]
    
    return Classification_model.predict_proba(input_ftrs)

if st.button("Predict Country"):

    country = model_pred(gender,age,language,browser)
    
    dest_country = ""

    
    if np.argmax(country) == 10:
        dest_country = "United States"
    else:
        country[0][7] = 0
        country[0][10] = 0
        if np.argmax(country) == 0:
            dest_country = "Australia"
        elif np.argmax(country) == 1:
            dest_country = "Canada"
        elif np.argmax(country) == 2:
            dest_country = "Germany"
        elif np.argmax(country) == 3:
            dest_country = "Spain"
        elif np.argmax(country) == 4:
            dest_country = "France"
        elif np.argmax(country) == 5:
            dest_country = "Great Britain"
        elif np.argmax(country) == 6: 
            dest_country = "Italy"
        elif np.argmax(country) == 8:
            dest_country = "Netherland"
        elif np.argmax(country) == 9:
            dest_country = "Portugal"
        else:
            dest_country = "Other Country"


    st.text("Predicted country of the destination: "+ str(dest_country))