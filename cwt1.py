import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
a = st.number_input("Applicant's Annual Income",value=0)
b = st.number_input("Applicant's Monthly Inhand Salary",value=0)
c = st.number_input("Applicant's Number of Bank Accounts",value=0)
d = st.number_input("Applicant's Number of Credit cards",value=0)
e = st.number_input("Applicant's Interest rate",value=0)
f = st.number_input("Applicant's Number of Loans",value=0)
g = st.number_input("Average number of days delayed by the person",value=0)
h = st.number_input("Number of delayed payments",value=0)
i = st.number_input("Credit Mix (Bad: 0, Standard: 1, Good: 3)",value=0)
j = st.number_input("Applicant's Outstanding Debt",value=0)
k = st.number_input("Credit History Age",value=0)
l = st.number_input("Monthly Balance",value=0)
mode = pickle.load(open("model.pkl",'rb'))
features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
result = mode.predict(features)

st.button("Reset", type="primary")
if st.button('Predict'):
    st.write(result)
else:
    st.write('Enter valid values')
