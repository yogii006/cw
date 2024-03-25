import pickle
import streamlit as st
import numpy as np
mode = pickle.load(open('model.pkl','rb'))
st.title("Food Delivery Time Prediction")
a = st.number_input("enter the age of the delivery man")
b = st.number_input("enter the rating of the delivery man")
c = st.number_input("enter the distance of delivery ")
order = st.radio("Choose the value type of order",['Meal','Snacks','Drinks','Buffet'])
if order == 'Meal':
    d = 0
elif order == 'Snacks':
    d = 1
elif order == 'Drinks':
    d = 2
elif order == 'Buffet':
    d = 3
vehicle = st.radio("Choose the value type of vehicle",['motorcycle ','scooter ','electric_scooter ','bicycle '])
if vehicle == 'motorcycle ':
    e = 0
elif vehicle == 'scooter ':
    e = 1
elif vehicle == 'electric_scooter ':
    e = 2
elif vehicle == 'bicycle ':
    e = 3
if st.button('predict'):
    features = np.array([[a, d, e, b, c]])
    result1 = mode.predict(features)
    result = result1[0][0][0]
    st.write("Predicted Delivery Time in Minutes = ", result)

# print("Predicted Delivery Time in Minutes = ", mode.predict(features))
    # 21.0	4.5	26.471617	80.313564	26.481617	80.323564	0.0	1.0	12.0	1.492347