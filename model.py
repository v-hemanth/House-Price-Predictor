#Import Libraries
import numpy as np
import pandas as pd
import joblib
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
#load data
df = pd.read_csv("data/cat_encoded_data.csv")
 
# Split data
X= df.drop('price', axis=1)
y= df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51)
 
# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
 
 
###### Load Model
 
model = joblib.load('ML_house_Price_predictor_rfr.pkl')
 
 
# it help to get predicted value of house  by providing features value 
def predict_house_price(bath,balcony,total_sqft_int,bhk,price_per_sqft,area_type,availability,location):
 
  x =np.zeros(len(X.columns)) 
  x[0]=bath
  x[1]=balcony
  x[2]=total_sqft_int
  x[3]=bhk
  x[4]=price_per_sqft
 
  if "availability"=="Ready To Move":
    x[8]=1
 
  if 'area_type'+area_type in X.columns:
    area_type_index = np.where(X.columns=="area_type"+area_type)[0][0]
    x[area_type_index] =1
 
  if 'location_'+location in X.columns:
    loc_index = np.where(X.columns=="location_"+location)[0][0]
    x[loc_index] =1
 
  # feature scaling
  x = sc.transform([x])[0]
 
  return model.predict([x])[0]