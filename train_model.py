import pandas as pd
import numpy as np
from sklearn import pipeline, preprocessing, impute, model_selection, linear_model, metrics, tree, neighbors, ensemble, \
    neural_network
import joblib
def train_model():
    data=pd.read_csv("forestfires.csv")
    newdata=data["FFMC DMC DC ISI temp RH wind rain".split(" ")]
    mypipeline=pipeline.Pipeline(
                                            [("standard", preprocessing.StandardScaler()),
                                            ("imputer", impute.SimpleImputer(strategy="median"))]
                                )
    data=pd.concat([data["X,Y,month,day".split(',')],pd.DataFrame(mypipeline.fit_transform(newdata),columns="FFMC DMC DC ISI temp RH wind rain".split(" ")),data['area']],axis=1)
    for train_block,test_block in model_selection.ShuffleSplit(n_splits=1, random_state=42, test_size=0.2).split(data):
         pass
    trainingData=data.iloc[train_block]
    testData=data.iloc[test_block]
    model=linear_model.LinearRegression()
    #model=tree.DecisionTreeRegressor()
    #model=linear_model.PoissonRegressor()
    #model=linear_model.TheilSenRegressor()
    #model=linear_model.TweedieRegressor()
    #model=linear_model.HuberRegressor()
    #model=linear_model.PassiveAggressiveRegressor()
    #model=neighbors.KNeighborsRegressor()
    #model=ensemble.RandomForestRegressor()
    #model=ensemble.HistGradientBoostingRegressor()
    #model=neural_network.MLPRegressor()
    model.fit(trainingData["X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain".split(",")],trainingData['area'])
    scores=model_selection.cross_val_score(model,trainingData["X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain".split(",")],trainingData['area'],scoring="neg_mean_squared_error")
    # scorehandler(model,scores)
    joblib.dump(model, "ForestFire.joblib")
def scorehandler(model,scores):
    rmse=np.sqrt(-scores)
    with open("resultslog.txt",mode="a") as result:
        result.write(f"{model}\nMean: {rmse.mean()}\nStandard Deviation: {rmse.std()}\n\n")
if __name__=="__main__":
    train_model()
