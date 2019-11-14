import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
le = preprocessing.LabelEncoder()

training=pd.read_csv('UNSW_NB15_training-set.csv')#Loading Training Dataset
testing = pd.read_csv('UNSW_NB15_testing-set.csv')#Loading Testing Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
Y_train = training.label
X_train = training.iloc[:,2:43]
X_train = X_train.drop(columns=["proto","service", "state"])
Y_test = testing.label
X_test = testing.iloc[:,2:43]
X_test = X_test.drop(columns=["proto","service", "state"])
Y_test = Y_test.fillna(0)
X_test = X_test.fillna(0)
model = ExtraTreesClassifier()
model.fit(X_test,Y_test)
importance = model.feature_importances_*100
print(importance)

col = X_train.columns
nimp_col = []
for ind,i in enumerate(list(importance)):
  if(i<2):
    nimp_col.append(col[ind])
    X_train = X_train.drop(columns=col[ind])
Final = X_train
print(Final)
X_test = X_test.drop(columns=nimp_col)
logisticRegr = LogisticRegression()
model = logisticRegr.fit(Final, Y_train)
score = logisticRegr.score(X_test, Y_test)
print(score)

plt.figure(figsize=(10,10))
plt.bar(col,importance)
plt.xticks(col, col, rotation='vertical')

y=training.attack_cat#Assigning Target variable or dependant Variable
x=Final
y_enc = le.fit_transform(y)#converting Strings to corresponding integer representation
print(y_enc)
model = KNeighborsClassifier(n_neighbors=5)#KNN for 5 neighbours
model.fit(x,y_enc)#Training

y_test=testing.attack_cat#Assigning Target variable or dependant Variable
print(Final.columns)
x_test=testing.drop(columns=nimp_col)
x_test=x_test.drop(columns=["proto","service", "state","id","attack_cat","label","dur"])#dropping unwanted columns
x_test = x_test.fillna(0)
print(x_test.columns)
y_test = y_test.fillna("Normal")
y_enc_test = le.fit_transform(y_test)#converting Strings to corresponding integer representation
y_pred = model.predict(x_test)

from sklearn import metrics
print(metrics.accuracy_score(y_enc_test, y_pred))

mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(le.inverse_transform(y_pred))
print(mapping)
print(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sn
labels = list(mapping.keys())
cm = confusion_matrix(y_enc_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in labels],columns = [i for i in labels])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
