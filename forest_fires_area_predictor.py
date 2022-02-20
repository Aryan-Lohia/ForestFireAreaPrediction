import joblib
import numpy as np
import pandas as pd
from sklearn import pipeline,preprocessing,impute
model=joblib.load("ForestFire.joblib")
dayno="sun mon tue wed thu fri sat".split(" ")
monthno="jan feb mar apr may jun jul aug sep oct nov dec".split(" ")
X=float(input("Enter x-axis spatial coordinate within the Montesinho park map: 1 to 9\n"))
Y =float(input("Enter  y-axis spatial coordinate within the Montesinho park map: 2 to 9\n"))
month=monthno.index(input("Enter month of the year: jan to dec\n"))+1
day =dayno.index(input("Enter day of the week: mon to sun\n"))+1
FFMC =float(input("Enter  FFMC index from the FWI system: 18.7 to 96.20\n"))
DMC =float(input("Enter  DMC index from the FWI system: 1.1 to 291.3\n" ))
DC =float(input("Enter  DC index from the FWI system: 7.9 to 860.6\n" ))
ISI =float(input("Enter  ISI index from the FWI system: 0.0 to 56.10\n"))
temp =float(input("Enter  temperature in Celsius degrees: 2.2 to 33.30\n"))
RH =float(input("Enter  relative humidity in %: 15.0 to 100\n"))
wind =float(input("Enter  wind speed in km/h: 0.40 to 9.40\n" ))
rain =float(input("Enter  outside rain in mm/m2 : 0.0 to 6.4\n" ))
mypipeline=pipeline.Pipeline(
                                        [("standard", preprocessing.StandardScaler()),
                                        ("imputer", impute.SimpleImputer(strategy="median"))]
                            )
data=np.array([[X,Y,month,day]])
newdata=np.array([[FFMC,DMC,DC,ISI,temp,RH,wind,rain]])
print(newdata)
newdata=mypipeline.fit_transform(newdata)
print(newdata)
data=arr = np.hstack([data,newdata])
print(model.predict(data))
