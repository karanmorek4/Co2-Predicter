from pyexpat import model
from statistics import mode

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv("Fuel.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


X=cdf.iloc[:,:3]
Y=cdf.iloc[:,-1]

#create instance of liner regression

regression=LinearRegression()

regression.fit(X,Y)

pickle.dump(regression,open('model.pkl','wb'))

#lets test the model
#loading the model to compare the resuls
# model=pickle.load(open('model.pkl','rb'))
# print(model.predict([[2.6,8,10.1]]))