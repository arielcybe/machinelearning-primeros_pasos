import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #for plotting purpose
#from sklearn.preprocessing import linear_model   #for implementing multiple linear regression
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Datasets/vino.csv")
print(df)

X=df[["acidez_fija","acidez_volatil","acido_citrico","azucar_res","cloruros","SO2_libre","SO2_total","densidad","pH","sulfatos",
"calidad"]]
y=df["alcohol"]
reg = LinearRegression().fit(X, y)
print (str(reg.score(X, y)))
print (str(reg.coef_))
print (str(reg.intercept_))
print (str(reg.predict(np.array([[ 7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 5]]))))
#https://codeburst.io/multiple-linear-regression-sklearn-and-statsmodels-798750747755
