#Team Members: Heena Agarwal, Kajal Dalvi, Pulkit Kalia, Shruti Singh

#Data taken from: https://data.worldbank.org/indicator
#https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/

#General idea: This code aggreagates temperature, precipitation, co2 per year,country. The resulting rdd is
#merged with external datasets and calculated betas(co2/greenshouse emissions, precipitation, energy, population) country wise. Top 10 countries with highest Co2 emission,
#Bottom 10 countries with co2 emission were taken and their respective betas were calculated.
#Also, Top 10 countries with high betas values are shown which helps in analysing if co2 emission is a major factor
#in that country or not.
#We have also created a world map depicting Co2 emission per country, color coded by amount of co2 released which will be included in the presentation.

# Algorithms used: Linear Regression, MapReduce
# Big Data framework: Spark

# System used: local Mac with 8 GB RAM and Google Cloud DataProc running Debian(same as assignment) with 5 nodes

from pyspark.sql import SQLContext
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
import glob
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import reverse_geocoder as rg
import pprint
import pycountry
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
from pyspark.sql import Row
import csv
import reverse_geocoder as rg
import statsmodels.formula.api as smf

sc = SparkContext("local", "project")
spark = SparkSession(sc)

country_mappings = {}

#Code to find country code using reverse_geocoder package
def findCC(coordinates):
    if coordinates in country_mappings:
        return country_mappings[coordinates]
    else:
        result = rg.search(coordinates)
        country_code=list(result[0].items())[5][1]
        cc = pycountry.countries.get(alpha_2=country_code)
        if cc is None:
            cc = "NA"
        else:
            cc = cc.alpha_3
        country_mappings[coordinates] = cc
        return cc

#created a map so that reverse_geocoder library is not called again and again: improves speed
preci_attr = {'A': 6, 'B': 12, 'C': 18, 'D': 24, 'E':12, 'F': 24, 'G':  24, 'H': 0, 'I': 0, ' ':1, '': 1}

#imputing the missing data as described in the documenation of the data
def cleanData(data):
    precipitation = float(data[-4].strip()) * preci_attr[data[-3]]
    snow = float(data[-2].strip())
    if data[-4] == '99.9':
        precipitation = 0
    if snow == 999.9:
        snow = 0

    return ((findCC((data[2], data[3])), datetime.strptime(data[1], "%Y-%m-%d").year),
            (float(data[6].strip()), precipitation, snow, 1))


path = "/Users/Desktop/MS/BigData/Project/data/run/*/*.csv"
rdd = sc.textFile(path)
filtered_rdd = rdd.map(lambda row: next(csv.reader(row.splitlines(), skipinitialspace=True)))\
            .filter(lambda x: x[0] != 'STATION' and x[2] != '' and x[3] != '')

#calculating mean of temperature, precipitation per country and year
data = filtered_rdd.map(lambda d: cleanData(d)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
data = data.map(lambda x: (x[0][0], x[0][1], x[1][0] / x[1][3], x[1][1] / x[1][3], x[1][2]))
data_country=data

#reading secondary data
data_co2 = sc.textFile("C:\\MS Stony Brook\\sem3\\big data\\project_data\\other_data\\co2*.csv")
data_energy = sc.textFile("C:\\MS Stony Brook\\sem3\\big data\\project_data\\other_data\\Energy*.csv")
data_population = sc.textFile("C:\\MS Stony Brook\\sem3\\big data\\project_data\\other_data\\Population_final.csv")

#slitting the csv files
data_co2=data_co2.map(lambda x: x.split(',')).filter(lambda x: x[0]!='Country Name')
data_energy=data_energy.map(lambda x: x.split(',')).filter(lambda x: x[0]!='Country Name')
data_population=data_population.map(lambda x: x.split(',')).filter(lambda x: x[0]!='Country Name')


data_country= data_country.map(lambda x: (x[1],x[2],x[3],x[4]))
data_co2 = data_co2.map(lambda x: (x[1],x[2],x[3]))
data_energy = data_energy.map(lambda x: (x[1],x[2],x[3]))
data_population = data_population.map(lambda x: (x[1],x[2],x[3]))

#creating spark dataframe
country_data = spark.createDataFrame(data_country,['country_code','year','temperature','precipitation'])
co2_data = spark.createDataFrame(data_co2,['country_code','year','co2'])
energy_data = spark.createDataFrame(data_energy,['country_code','year','energy'])
population_data = spark.createDataFrame(data_population,['country_code','year','population'])

#joining the dataframe together to form a single dataframe
df = country_data.join(co2_data, on=['country_code','year'], how='inner')
df = df.join(energy_data, on=['country_code','year'], how='inner')
df = df.join(population_data, on=['country_code','year'], how='inner')


top_countries_co2=df
top_countries_co2 = top_countries_co2.withColumn('co2', col('co2').cast('float'))

#calculating avg Co2 emission per country
countries=top_countries_co2.groupBy('country_code').avg('co2').collect()
countries.sort(key = lambda x: x[1])  


countries=pd.DataFrame(countries)
least_countries_co2=set()
highest_countries_co2=set()
i=0
while i < 10:
    least_countries_co2.add(countries[i]['country_code'])
    i=i+1
    
i=90
while i < 100:
    highest_countries_co2.add(countries[i]['country_code'])
    i=i+1
    
    
data= df.rdd.map(list)

# (country_code, (year,temperature,precipitation,co2,energy,population))
data1=data.map(lambda x: (x[0],(x[1],x[2],x[3],x[4],x[5],x[6]))).groupByKey().mapValues(list)


#Method to calculate betas, checking the effect of co2 + precipitation + energy + population 
#on temperature country wise
def calculate_betas(data):
    v = data[1]
    temperature=list()
    precipitation=list()
    co2=list()
    energy=list()
    population=list()
    for values in v:
        temperature.append(float(values[1]))
        precipitation.append(float(values[2]))
        co2.append(float(values[3]))
        energy.append(float(values[4]))
        population.append(float(values[5]))
    
    #standardizing the data
    temperature=(temperature-np.mean(temperature))/np.std(temperature)
    precipitation=(precipitation-np.mean(precipitation))/np.std(precipitation)
    co2=(co2-np.mean(co2))/np.std(co2)
    energy=(energy-np.mean(energy))/np.std(energy)
    population=(population-np.mean(population))/np.std(population)
    df=pd.DataFrame(np.column_stack([temperature, precipitation,co2,energy,population]), 
                               columns=['temperature', 'precipitation', 'co2','energy','population'])
    lm = smf.ols(formula='temperature ~  co2 + precipitation + energy + population', data=df).fit()
    coeffs=lm.params
    #Here coeffs are betas
    return (data[0],(coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4]))



betas=data1.map(lambda x: calculate_betas(x)).sortBy(lambda x: x[1][1],ascending=False)
print('Countries with highest positive beta for Co2: ')
print(betas.take(10))

most_emitting_countries=betas.filter(lambda x: x[0] in highest_countries_co2)
least_emitting_countries=betas.filter(lambda x: x[0] in least_countries_co2)

print('Countries with highest average Co2 emission: ')
print(most_emitting_countries.take(10))
print('Countries with lowest average Co2 emission: ')
print(least_emitting_countries.take(10))