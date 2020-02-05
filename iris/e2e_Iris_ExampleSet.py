# Databricks notebook source
# compare algorithms
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow

storage_account_name = "e2edevelopment"
storage_account_access_key = "sxmLpkte93LHpWg4SwKKuJplBcFgKXk6KjN9K35dJhIsDb+vI+dNfDB42B82tGkrgJJo8E7DPRxqHLSd6hbA8w=="

file_location = "wasbs://e2edbstorage@e2edevelopment.blob.core.windows.net/e2edbstorage/E2Edata/Iris.csv"
file_type = "csv"

spark.conf.set("fs.azure.account.key."+storage_account_name+".blob.core.windows.net",storage_account_access_key)

Iris = spark.read.csv("wasbs://e2edbstorage@e2edevelopment.blob.core.windows.net/E2Edata/Iris.csv")
data = Iris.toPandas()
print(data.head())
#data = pd.read_csv(file_location)

mlflow.set_experiment("/Users/tnormile@e2evapoutlook.onmicrosoft.com/my-experiment-test-2")

new_header = data.iloc[0]
data = data[1:]
data.columns = new_header

print(data['Species'].value_counts())

tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')
display(plt.show())


#mlflow.log_artifact("wasbs://e2edbstorage@e2edevelopment.blob.core.windows.net/E2Edata/output1.txt")

# COMMAND ----------

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

print(X.shape)
print(y.shape)
k_range = list(range(1,26))
scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X,y)
  y_pred = knn.predict(X)
  scores.append(metrics.accuracy_score(y, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel("Accuracy Score")
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

display(plt.show())

# COMMAND ----------

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

logreg = LogisticRegression()
logreg.fit(X,y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=5)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# COMMAND ----------

k_range = list(range(1, 26))
scores = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  scores.append(metrics.accuracy_score(y_test, y_pred))
  
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuraacy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest Neighbors')
display(plt.show())

# COMMAND ----------

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# COMMAND ----------

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
knn.predict([[6, 3, 4, 2]])

# COMMAND ----------

with mlflow.start_run(nested=True):
  mlflow.log_metric("sepal-length",2,step=1)
with open("output1.txt", "w") as f:
   f.write("Hello World")
mlflow.log_artifact("output1.txt")
