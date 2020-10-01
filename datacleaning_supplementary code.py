#cleaning dataset of factors affecting climate #co2 emission, energy use and population
#filled missing values by imputing it with nearest neighbor(KNNImputer).
#converted dataset as per the required format so to group them by country code and year


import pandas as pd
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)


def cleanCO2data(df):
    df = df.drop(columns=['Indicator Code', 'Indicator Name'])
    df1 = (df.set_index(["Country Name", "Country Code"])
           .stack()
           .reset_index(name='CO2_emission')
           .rename(columns={'level_2': 'Year'}))

    return df1


def cleanForestdata(df):
    df = df.drop(columns=['Indicator Code', 'Indicator Name'])
    df1 = (df.set_index(["Country Name", "Country Code"])
           .stack()
           .reset_index(name='ForestArea')
           .rename(columns={'level_2': 'Year'}))

    return df1


def cleanEnergydata(df):
    df = df.drop(columns=['Indicator Code', 'Indicator Name'])
    df1 = (df.set_index(["Country Name", "Country Code"])
           .stack()
           .reset_index(name='EnergyUse')
           .rename(columns={'level_2': 'Year'}))

    return df1


def cleanPopulationdata(df):
    df = df.drop(columns=['Indicator Code', 'Indicator Name'])
    df1 = (df.set_index(["Country Name", "Country Code"])
           .stack()
           .reset_index(name='Population')
           .rename(columns={'level_2': 'Year'}))

    return df1


data_co2=pd.read_csv('CO2_Emission.csv')
data_co2.dropna(axis=0,thresh=35,inplace=True)
data_co2.iloc[:, 4:]=imputer.fit_transform(data_co2.iloc[:, 4:])
data_co2.to_csv('co2_data_final.csv',index=False)
co2 = pd.read_csv('co2_data_final.csv')
final_format_co2 = cleanCO2data(co2)
final_format_co2.to_csv("co2_emission_final.csv", index=False)

energy_data=pd.read_csv('EnergyUse.csv')
energy_data.dropna(axis=0,thresh=35,inplace=True)
energy_data.iloc[:, 4:]=imputer.fit_transform(energy_data.iloc[:, 4:])
energy_data.to_csv('energy_data_final.csv',index=False)
energy = pd.read_csv('energy_data_final.csv')
final_format_energy = cleanEnergydata(energy)
final_format_energy.to_csv("Energy_Use_final.csv", index=False)

forest_data=pd.read_csv('forest_data.csv')
forest_data.dropna(axis=0,thresh=20,inplace=True)
forest_data.iloc[:, 4:]=imputer.fit_transform(forest_data.iloc[:, 4:])
forest_data.to_csv('forest_data_final.csv',index=False)
forest = pd.read_csv('forest_data_final.csv')
final_format_forest = cleanForestdata(forest)
final_format_forest.to_csv("Forest_Area_final.csv", index=False)

population_data=pd.read_csv('Population_growth.csv')
population_data.dropna(axis=0,thresh=35,inplace=True)
population_data.iloc[:, 4:]=imputer.fit_transform(population_data.iloc[:, 4:])
population_data.to_csv('population_data_final.csv',index=False)
population = pd.read_csv('population_data_final.csv')
final_format_population = cleanPopulationdata(population)
final_format_population.to_csv("Population_final.csv", index=False)
