import joblib
import numpy as np
import pandas as pd
from sklearn import pipeline,preprocessing,impute
model=joblib.load("ForestFire.joblib")
dayno="sun mon tue wed thu fri sat".split(" ")
monthno="jan feb mar apr may jun jul aug sep oct nov dec".split(" ")
# X=float(input("Enter x-axis spatial coordinate within the Montesinho park map: 1 to 9"))
# Y =float(input("Enter  y-axis spatial coordinate within the Montesinho park map: 2 to 9"))
# month=monthno.index(input("Enter month of the year: jan to dec"))+1
# day =dayno.index(input("Enter day of the week: mon to sun"))+1
# FFMC =float(input("Enter  FFMC index from the FWI system: 18.7 to 96.20"))
# DMC =float(input("Enter  DMC index from the FWI system: 1.1 to 291.3" ))
# DC =float(input("Enter  DC index from the FWI system: 7.9 to 860.6" ))
# ISI =float(input("Enter  ISI index from the FWI system: 0.0 to 56.10"))
# temp =float(input("Enter  temperature in Celsius degrees: 2.2 to 33.30"))
# RH =float(input("Enter  relative humidity in %: 15.0 to 100"))
# wind =float(input("Enter  wind speed in km/h: 0.40 to 9.40" ))
# rain =float(input("Enter  outside rain in mm/m2 : 0.0 to 6.4" ))
mypipeline=pipeline.Pipeline(
                                        [("standard", preprocessing.StandardScaler()),
                                        ("imputer", impute.SimpleImputer(strategy="median"))]
                            )
X=6
Y=3
month=9
day=2
FFMC,DMC,DC,ISI,temp,RH,wind,rain=91.6,108.4,764,6.2,23,34,2.2,0
data=np.array([[X,Y,month,day]])
newdata=np.array([[FFMC,DMC,DC,ISI,temp,RH,wind,rain]])
print(newdata)
newdata=mypipeline.fit_transform(newdata)
print(newdata)
data=arr = np.hstack([data,newdata])
print(model.predict(data))