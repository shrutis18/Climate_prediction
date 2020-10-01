# Team members: Heena Agarwal, Kajal Dalvi, Pulkit Kalia, Shruti Singh

# General description: Temperature and precipitation level forecasting for next 30 years based on past data since 1930.
# We are filtering data to take temperature and precipitation level for each month since 1930.
# We are making use of SARIMA model for forecasting, we have calculated model parameters to train using hyper-parameter tuning.
# Training data is taken till 2018 for training and 2019, 2020 data for testing. And are forecasting predictions till 2050
# and plotting the same using matplotlib (plots to present in slides and report)

# Course Big Data framework: Spark
# Course Big Data algorithm: Time series prediction using statistic model(SARIMA)

# Data source: https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/

# System used: local Mac with 8 GB RAM and Google Cloud DataProc running Debian(same as assignment) with 5 nodes

from pyspark.sql import SQLContext
import pyspark
from pyspark.sql import Row
import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession

from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error

import numpy as np
import glob
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pmdarima as pm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict


# precipitation attributes in data had characters, mapped them to necessary number of reports/hours observed
preci_attr = {'A': 6, 'B': 12, 'C': 18, 'D': 24, 'E':12, 'F': 24, 'G':  24, 'H': 0, 'I': 0, ' ':1, '':1}

def cleanData(data):    # retrieving month and year to group based on that tuple: ((month, year), (temperature, precipitation, 1))
    precipitation = float(data[-4].strip()) * preci_attr[data[-3]]
    if data[-4] == '99.9':          # handing for missing data
        precipitation = 0

    return ((datetime.strptime(data[1], "%Y-%m-%d").month, datetime.strptime(data[1], "%Y-%m-%d").year),
            (float(data[6].strip()), precipitation, 1))


sc = SparkContext("local", "project")
spark = SparkSession(sc)

path="/Users/prakashdalvi/Desktop/MS/BigData/Project/data/*/*.csv"      # reading data for all years
rdd = sc.textFile(path)
filtered_rdd = rdd.map(lambda row: next(csv.reader(row.splitlines(), skipinitialspace=True))).filter(lambda x: x[0]!='STATION')

# reduceByKey to calculate mean based on month and year
data=filtered_rdd.map(lambda d: cleanData(d)).reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1], a[2]+b[2]))

# Getting mean temperature and precipitation, for each month and year
data=data.map(lambda x: (x[0][0], x[0][1], x[1][0]/x[1][2], x[1][1]/x[1][2]))
# filtering 1929 data as it not sufficient and for complete year
frame = data.filter(lambda d: d[1] != '1929').toDF().toPandas()

frame.columns = ['index', 'month', 'year', 'temperature', 'precipitation']
df=frame.sort_values(["year", "month"], ascending = (True, True))


# ------------------------------------- SARIMA model for temperature forecasting -------------------------------------------

train=df[0:1068]    # training data till year 2018
test=df[1068:1085]  # test data from 2019 to May 2020

# Below is SARIMA model training and predicting values of temperature till 2020, to check mean error on data
my_order = (2, 1, 1)
my_seasonal_order = (2, 1, 1, 12)
model = sm.tsa.statespace.SARIMAX(train['temperature'], order=my_order, seasonal_order=my_seasonal_order, trend='n')
model_fit = model.fit()
yhat = model_fit.forecast(17)


# Plotting graph for actual and predicted values from Jan 2019 and May 2020
from datetime import datetime, timedelta
from collections import OrderedDict
dates = ["2019-01-01", "2020-05-30"]
start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
x=OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()
x=list(x)

plt.plot(x,(yhat-32)*(5/9), label="predicted")
plt.plot(x,(test["temperature"]-32)*(5/9), label="actual")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xticks(rotation=90)
plt.ylabel("Temperature (celsius)")
plt.show()

print("Mean squared error for temperature " + str(mean_squared_error(test["temperature"], yhat)))

# pred represents values predicted for temperature using model till year 2050. Plotting the same

pred = model_fit.forecast(385)
pred=(pred-32)*(5/9)
from datetime import datetime, timedelta
from collections import OrderedDict
dates = ["2019-01-01", "2051-01-30"]
start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
x_1=OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()
x_1=list(x_1)

# Plotting using matplotlib
plt.figure(figsize=(30,20))
plt.xticks(np.arange(0, 600, 12))
plt.plot(x_1,pred, label="predicted")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation=90,size=20)
plt.ylabel("Temperature (celsius)",size=30)
plt.show()


# ------------------------------------- SARIMA model for precipitation forecasting -------------------------------------------

# training data till year 2018.. test data from 2019 to May 2020
train=df[0:1068]
test=df[1068:1085]

# Below is SARIMA model training and predicting values of precipitation till 2020, to check mean error on data
my_order = (1, 1, 3)
my_seasonal_order = (1, 1, 1, 12)
model = sm.tsa.statespace.SARIMAX(train['precipitation'], order=my_order, seasonal_order=my_seasonal_order, trend='n')
model_fit = model.fit()

# Plotting graph for actual and predicted values from Jan 2019 and May 2020
yhat = model_fit.forecast(17)
dates = ["2019-01-01", "2020-05-30"]
start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
x=OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()
x=list(x)

plt.plot(x,yhat, label="predicted")
plt.plot(x,test["precipitation"], label="actual")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xticks(rotation=90)
plt.ylabel("Precipitation")
plt.show()

print("Mean squared error for precipitation " + str(mean_squared_error(test["precipitation"], yhat)))

# pred represents values predicted for precipitation using model till year 2050. Plotting the same
pred = model_fit.forecast(385)
dates = ["2019-01-01", "2051-01-30"]
start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
x_1=OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()
x_1=list(x_1)

# Plotting using matplotlib
plt.figure(figsize=(30,20))
plt.xticks(np.arange(0, 600, 12))
plt.plot(x_1,pred, label="predicted")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation=90,size=20)
plt.ylabel("Precipitation",size=30)
plt.show()
