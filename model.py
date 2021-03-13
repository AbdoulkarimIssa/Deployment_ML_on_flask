import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

raw_data = pd.read_csv('house_data.csv')
data_na = raw_data.dropna()
data = data_na[data_na["price"] < 8000]
data = data.reset_index(drop = True)
xtrain, xtest, ytrain, ytest = train_test_split(data[["surface", "arrondissement"]], data[["price"]], test_size=0.3)
lr = LinearRegression()
lr_baseline = lr.fit(xtrain[["surface"]], ytrain)
baseline_pred = lr_baseline.predict(xtest[["surface"]])
lrs = []
for i in np.unique(xtrain["arrondissement"]):

    # On génère un jeu de données par arrondissement
    tr_arr = xtrain['arrondissement']==i
    te_arr = xtest['arrondissement']==i

    xtrain_arr = xtrain[tr_arr]
    ytrain_arr = ytrain[tr_arr]

    xtest_arr = xtest[te_arr]
    ytest_arr = ytest[te_arr]

    lr = LinearRegression()
    lr.fit(xtrain_arr[["surface"]], ytrain_arr)
    lrs.append(lr)
#on ajoute ici les different modèles

paris1=lrs[0]
paris2=lrs[1]
paris3=lrs[2]
paris4=lrs[3]
paris10=lrs[4]

pickle.dump(paris1, open('paris1.pkl','wb'))
pickle.dump(paris2, open('paris2.pkl','wb'))
pickle.dump(paris3, open('paris3.pkl','wb'))
pickle.dump(paris4, open('paris4.pkl','wb'))
pickle.dump(paris10, open('paris10.pkl','wb'))



