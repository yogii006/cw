import numpy as np
import pandas as pd
df = pd.read_csv("deliverytime.txt")
# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
  
# Calculate the distance between each pair of points
df['distance'] = np.nan

for i in range(len(df)):
    df.loc[i, 'distance'] = distcalculate(df.loc[i, 'Restaurant_latitude'], 
                                        df.loc[i, 'Restaurant_longitude'], 
                                        df.loc[i, 'Delivery_location_latitude'], 
                                        df.loc[i, 'Delivery_location_longitude'])
df.info()
#splitting data
df['Type_of_order'].replace('Meal ', 0, inplace=True)
df['Type_of_order'].replace('Snack ', 1, inplace=True)
df['Type_of_order'].replace('Drinks ', 2, inplace=True)
df['Type_of_order'].replace('Buffet ', 3, inplace=True)
df['Type_of_vehicle'].replace('motorcycle ', 0, inplace=True)
df['Type_of_vehicle'].replace('scooter ', 1, inplace=True)
df['Type_of_vehicle'].replace('electric_scooter ', 2, inplace=True)
df['Type_of_vehicle'].replace('bicycle ', 3, inplace=True)
df.dropna()
from sklearn.model_selection import train_test_split
x = np.array(df[["Delivery_person_Age",'Type_of_order','Type_of_vehicle', 
                   "Delivery_person_Ratings", 
                   "distance"]])
y = np.array(df[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)
# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(xtrain.shape[1], 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))  # Output layer

model.summary()
# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=5)
import pickle
pickle.dump(model,open('model.pkl','wb'))