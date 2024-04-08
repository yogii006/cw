import pickle
import streamlit as st
model = pickle.load(open('model.pkl','rb'))
import numpy as np

st.title("Netflix stock price prdiction")
with st.sidebar:
    a = st.number_input("enter the opening of the stock")
    b = st.number_input("enter the high of the stock")
    c = st.number_input("enter the low of the stock")
    d = st.number_input("enter the close of the stock")
    #features = [Open, High, Low, Adj Close, Volume]
    features = np.array([[a, b, c, d]])
# st.button("predict")  
if st.button('predict'):
    result = model.predict(features)
    p = result[0][0]
    col1, col2, col3, col4,col5  = st.columns(5)
    with col1:
        st.header("Open")
        st.header(a)
    with col2:
        st.header("High")
        st.header(b)
    with col3:
        st.header("Low")
        st.header(c)
    with col4:
        st.header("Close")
        st.header(d)
    with col5:
        st.header("Predicted")
        st.header(p)

