import numpy as np
import pandas as pd
import sklearn
df = pd.read_csv("/Datasets/train.csv")
df["Credit_Mix"] = df["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})
from sklearn.model_selection import train_test_split
x = np.array(df[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(df[["Credit_Score"]]) 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
import pickle
pickle.dump(model,open("model.pkl",'wb')) 