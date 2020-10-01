# Team members: Heena Agarwal, Kajal Dalvi, Pulkit Kalia, Shruti Singh


# climate data was taken from https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/
# co2, energy use and population were taken from https://data.worldbank.org/indicator

# General description: We have used the climate data, co2_emission, energy_use, population data for 1960-2014 years and have
# done filtering based on country code and year. Since our climate data set has Latitude and Longitude, we have used reverse geocoder (Google API)
# to find country code using that and then joined the climate data with the other datasets using country code and year to give final a
# df that has country code, year, temp, precipitation, co2 emission, energy use and population.
# We are finding correlation of the factors like CO2 emission, energy and population on temperature and precipitation using
# fixed effects linear regression model which has time dependent X's and y and control of time independent variables as well.
# In our case temp/precip is y and CO2 emission, population, energy use are X's which are all time dependent variables and
# year and countries are control/dummy variables. We have then calculated precdicted y, t_stats and p-value to comment on whether or
# not hypothesis is rejected or accepted.

# System used: local Dell Laptop with 8 GB RAM and Google Cloud DataProc running Debian(same as assignment) with 5 nodes

##DATA FRAMEWORKS USED ARE SPARK AND TENSORFLOW
##ALGORITHMS/CONCEPTS USED ARE HYPOTHESIS TESTING, CORRELATION
## In this file we have used Spark to retrieve data in the required form using map, reducebykey ,filter, create data frame, join etc transformations
# of spark and collect action. We have used tensor flow to perform linear regression, find betas and predict y and hypothesis testing.


import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession
import reverse_geocoder as rg
import pycountry
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# creating dictionary so that lat and long which have already mapped to country don't have to run geocoder code again
country_mappings = {}


# finding country code using reverse geocoder which gives 2 char country code and used pycountry lib to get 3 char country code
def findCC(coordinates):
    if coordinates in country_mappings:
        return country_mappings[coordinates]
    else:
        result = rg.search(coordinates)
        country_code = list(result[0].items())[5][1]
        cc = pycountry.countries.get(alpha_2=country_code)
        if cc is None:
            cc = "NA"
        else:
            cc = cc.alpha_3
        country_mappings[coordinates] = cc
        return cc


# precipitation attributes in data had characters, mapped them to necessary number of reports/hours observed
preci_attr = {'A': 6, 'B': 12, 'C': 18, 'D': 24, 'E': 12, 'F': 24, 'G': 24, 'H': 0, 'I': 0, '': 1, ' ': 1}


def cleanData(
        data):  # retrieving country code, year to group based on that tuple: ((country code, year), (temperature, precipitation, snow, 1))
    precipitation = float(data[-4].strip()) * preci_attr[data[-3]]
    snow = float(data[-2].strip())
    if data[-4] == '99.9':  # handing for missing data
        precipitation = 0
    if snow == 999.9:
        snow = 0

    return ((findCC((data[2], data[3])), datetime.strptime(data[1], "%Y-%m-%d").year),
            (float(data[6].strip()), precipitation, snow, 1))


# calculation mean temp, precipitation, total snow for all countries and for 1960-2014 years
path = "D:\\BDA_Project_files\\files\\1960.csv"
rdd = sc.textFile(path)
filtered_rdd = rdd.map(lambda row: next(csv.reader(row.splitlines(), skipinitialspace=True))).filter(
    lambda x: x[0] != 'STATION' and (x[2] != '' and x[3] != ''))
data = filtered_rdd.map(lambda d: cleanData(d))
data = data.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
data = data.map(lambda x: (x[0][0], x[0][1], x[1][0] / x[1][3], x[1][1] / x[1][3], x[1][2]))



##grouping and joining data using spark transformations to get a data frame that has country_code, year, temperature, precipitation,
# co2_emission, energyuse, population data

data_country = data
data_co2 = sc.textFile("C:\\Users\\Yojana\\Desktop\\BigDatasets\\other_data\\co2_emission.csv")
data_energy = sc.textFile("C:\\Users\\Yojana\\Desktop\\BigDatasets\\other_data\\energy_use.csv")
data_population = sc.textFile("C:\\Users\\Yojana\\Desktop\\BigDatasets\\other_data\\population.csv")

#data_country = data_country.map(lambda x: x.split(',')).filter(lambda x: x[1] != '_1')
data_co2 = data_co2.map(lambda x: x.split(',')).filter(lambda x: x[0] != 'Country Name')
data_energy = data_energy.map(lambda x: x.split(',')).filter(lambda x: x[0] != 'Country Name')
data_population = data_population.map(lambda x: x.split(',')).filter(lambda x: x[0] != 'Country Name')

data_country = data_country.map(lambda x: (x[1], x[2], x[3], x[4]))
data_co2 = data_co2.map(lambda x: (x[1], x[2], x[3]))
data_energy = data_energy.map(lambda x: (x[1], x[2], x[3]))
data_population = data_population.map(lambda x: (x[1], x[2], x[3]))

country_data = spark.createDataFrame(data_country, ['country_code', 'year', 'temperature', 'precipitation'])
co2_data = spark.createDataFrame(data_co2, ['country_code', 'year', 'co2'])
energy_data = spark.createDataFrame(data_energy, ['country_code', 'year', 'energy'])
population_data = spark.createDataFrame(data_population, ['country_code', 'year', 'population'])

df = country_data.join(co2_data, on=['country_code', 'year'], how='inner')
df = df.join(energy_data, on=['country_code', 'year'], how='inner')
df = df.join(population_data, on=['country_code', 'year'], how='inner')

data = df.rdd.map(list)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
newdf = data.toDF().toPandas()
newdf.columns = ['country_code', 'year', 'temperature', 'precipitation', 'co2_emission', 'energy_use', 'population']

##  country_code |    year   |   temperature    |     precipitation   |    co2     |    energy    |     population

dummy = pd.get_dummies(newdf[['country_code', 'year']], prefix=['CC', 'YEAR'], drop_first=True)
tf_df = pd.concat([dummy, newdf], axis=1)
tf_df.drop('country_code', axis=1, inplace=True)
test = tf_df.apply(pd.to_numeric)

##HYPOTHESIS TESTING

# Step 1: get data
features, temp = test.drop(['temperature', 'year', 'precipitation'], axis=1), test['temperature']
features, precipitation = test.drop(['temperature', 'year', 'precipitation'], axis=1), test['precipitation']

# Step 2: standardize features (so all features are on the same scale):
featuresZ = zscore(features)
temp = zscore(temp)
precipitation = zscore(precipitation)

# add bias (also known as intercept)
featuresZ_pBias = np.c_[np.ones((featuresZ.shape[0], 1)), featuresZ]
featuresZ_pBias.shape

#creating nd array of X and y1 and y2 where y1 is temperature and y2 is precipitation
X = tf.constant(featuresZ_pBias, dtype=tf.float32, name="X")
y1 = tf.constant(temp.reshape(-1, 1), dtype=tf.float32, name="y1")
y2 = tf.constant(precipitation.reshape(-1, 1), dtype=tf.float32, name="y2")

Xt = tf.transpose(X)
penalty = tf.constant(1.0, dtype=tf.float32, name="penalty")
I = tf.constant(np.identity(featuresZ_pBias.shape[1]), dtype=tf.float32, name="I")

#calculating betas for both temperature and precipitation as dependent variable
beta1 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X) + penalty * I), Xt), y1)
beta2 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X) + penalty * I), Xt), y2)

# creates a session to run the operations
#calculate betas and y_predicted for both temperature and precipitation
with tf.Session() as sess:
    beta_value_temp = beta1.eval()
    beta_value_precip = beta2.eval()
    prediction_temp = tf.matmul(X, beta1)
    prediction_temp = prediction_temp.eval()
    prediction_precip = tf.matmul(X, beta2)
    prediction_precip = prediction_precip.eval()
    y1 = y1.eval()
    y2 = y2.eval()

print("predicted temp")
print(prediction_temp)
print("predicted precipitation")
print(prediction_precip)
print("Correlation of CO2 emission, Energy Use and Population on Temperature")
print(beta_value_temp)
print("Correlation of CO2 emission, Energy Use and Population on Precipation")
print(beta_value_precip)

#TABULAR FORM OF STATISTICS ON CLIMATE WILL BE MENTIONED IN THE REPORT SPECIFING ALL THE HYPOTHESIS
## Temperature statistics
## Calculating sum of square errors, degree of freedom, t_stats, plt_beta, p_value and finding hypothesis result
rss_temp = np.sum((prediction_temp - y1) ** 2)
m = newdf.shape[1]
dof = y1.shape[0] - (m + 1)
s_sqr = rss_temp / dof
features_mean = np.mean(featuresZ_pBias)
var = np.sum(((featuresZ_pBias - features_mean)) ** 2)
denr_temp = np.sqrt(s_sqr / var)
carbon_betas_temp = beta_value_temp[-3][0]
energy_betas_temp = beta_value_temp[-2][0]
population_betas_temp = beta_value_temp[-1][0]
t_stats_carbon_temp = carbon_betas_temp / denr_temp
t_stats_energy_temp = energy_betas_temp / denr_temp
t_stats_population_temp = population_betas_temp / denr_temp
plt_beta_carbon_temp = stats.t.sf(abs(t_stats_carbon_temp), df=dof)
plt_beta_energy_temp = stats.t.sf(abs(t_stats_energy_temp), df=dof)
plt_beta_pop_temp = stats.t.sf(abs(t_stats_population_temp), df=dof)
print("t-stats and p-value for carbon and temperature: ", t_stats_carbon_temp, 2 * plt_beta_carbon_temp)
print("t-stats and p-value for energy and temperature: ", t_stats_energy_temp, 2 * plt_beta_energy_temp)
print("t-stats and p-value for population and temperature: ", t_stats_population_temp, 2 * plt_beta_pop_temp)

##Precipitation statistics
## Calculating sum of square errors, degree of freedom, t_stats, plt_beta, p_value and finding hypothesis result
rss_precip = np.sum((prediction_precip - y1) ** 2)
s_sqr = rss_precip / dof
denr_precip = np.sqrt(s_sqr / var)
carbon_betas_precip = beta_value_precip[-3][0]
energy_betas_precip = beta_value_precip[-2][0]
population_betas_precip = beta_value_precip[-1][0]
t_stats_carbon_precip = carbon_betas_precip / denr_precip
t_stats_energy_precip = energy_betas_precip / denr_precip
t_stats_population_precip = population_betas_precip / denr_precip
plt_beta_carbon_precip = stats.t.sf(abs(t_stats_carbon_precip), df=dof)
plt_beta_energy_precip = stats.t.sf(abs(t_stats_energy_precip), df=dof)
plt_beta_pop_precip = stats.t.sf(abs(t_stats_population_precip), df=dof)
print("t-stats and p-value for carbon and precipitation: ", t_stats_carbon_precip, 2 * plt_beta_carbon_precip)
print("t-stats and p-value for energy and precipitation: ", t_stats_energy_precip, 2 * plt_beta_energy_precip)
print("t-stats and p-value for population and precipation: ", t_stats_population_precip, 2 * plt_beta_pop_precip)

list_attributes = [plt_beta_carbon_temp, plt_beta_energy_temp, plt_beta_pop_temp, plt_beta_carbon_precip,
                   plt_beta_energy_precip, plt_beta_pop_precip]

#accept/reject null hypothesis based on p-value and threshold alpha which by default is considered as 0.05
for element in list_attributes:
    if element < 0.05:
        print("Reject Null hypotheis")
    else:
        print("Accept Null hypothesis")
