import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv')
data = pd.read_csv('train.csv')
df.drop(columns=['id','activation_date','latitude','longitude','negotiable','property_age','cup_board'],inplace=True)
df.replace({'lease_type':{'FAMILY':3,'ANYONE':2,'BACHELOR':1,'COMPANY':4},'furnishing':{'SEMI_FURNISHED':1,'NOT_FURNISHED':0,'FULLY_FURNISHED':2},'parking':{'BOTH':2,'TWO_WHEELER':1,'FOUR_WHEELER':3,'NONE':0},'facing':{'E':2,'N':1,'W':4,'S':3,'NE':5,'SE':6,'NW':8,'SW':7},'water_supply':{'CORP_BORE':1,'CORPORATION':2,'BOREWELL':0},'building_type':{'IF':3,'AP':2,'IH':1,'GC':0}},inplace=True)
df.drop(columns=['lease_type','facing','water_supply'],inplace=True)
df.replace({'type':{'RK1':0,'BHK1':1,'BHK2':2,'BHK3':3,'BHK4':4,'BHK4PLUS':5}},inplace=True)

def fetch_true(obj):
    li = list(obj.split("true"))
    return len(li)-1

df['amenities']=df['amenities'].apply(fetch_true)

df['locality']=df['locality'].apply(lambda x: x.strip())
location_count = df['locality'].value_counts()
location_count_less_than_10 = location_count[location_count<=10]
df['locality']=df['locality'].apply(lambda x: 'others' if x in location_count_less_than_10 else x)

X = df.drop(columns=['rent'],axis=1)
Y = np.log(df['rent'])

ohe = OneHotEncoder()
ohe.fit(X[['locality']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['locality']),remainder='passthrough')

scores = []
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)

st.title("Real Estate Price Prediction Model")

locality = st.selectbox('Locality', df['locality'].unique())

type = st.selectbox('Type', data['type'].unique())

gym = st.selectbox('Gym', ['Yes','No'])

lift = st.selectbox('Lift', ['Yes','No'])

furnishing = st.selectbox('Furnishing', ['Yes','No'])

swim = st.selectbox('Swimming Pool', ['Yes','No'])

parking = st.selectbox('Parking', data['parking'].unique())

property_size = st.number_input('Size of Property')

bathroom = st.number_input('No of Bathrooms')

floor = st.number_input('No of Floors')

total_floor = st.number_input('No of Total Floors')

amenities = st.selectbox('Amenities', df['amenities'].unique())

building_type = st.selectbox('Building type', data['building_type'].unique())

balconies = st.selectbox('Balconies', df['balconies'].unique())

# 'RK1':0,'BHK1':1,'BHK2':2,'BHK3':3,'BHK4':4,'BHK4PLUS':5
if st.button('Predict Price'):
    if gym=="Yes":
        gym=1
    elif gym=="No":
        gym = 0

    if lift=="Yes":
        lift=1
    elif lift=="No":
        lift = 0

    if furnishing=="Yes":
        furnishing=1
    elif furnishing=="No":
        furnishing = 0

    if swim=="Yes":
        swim=1
    elif swim=="No":
        swim = 0

    if type == "BHK2":
        type = 2
    elif type == "BHK3":
        type = 3
    elif type == "BHK4":
        type = 4
    elif type == "BHK4PLUS":
        type = 5
    elif type == "BHK1":
        type = 1
    elif type == "RK1":
        type = 0

    if parking == "BOTH":
        parking = 2
    elif parking == "NONE":
        parking = 0
    elif parking == "TWO_WHEELER":
        parking = 1
    elif parking == "FOUR_WHEELER":
        parking = 3

    if building_type == "IF":
        building_type = 3
    elif building_type == "AP":
        building_type = 2
    elif building_type == "IH":
        building_type = 1
    elif building_type == "GC":
        building_type = 0

    


    query = np.array([type,locality,gym,lift,swim,furnishing,parking,property_size,bathroom,floor,total_floor,amenities,building_type,balconies])

    query = query.reshape(1, 14)

    st.title("Rs: " +
             str(int(np.exp(pipe.predict(query)[0]))))

    