#This code is for visualization purpose which will be included in the presentation.
#main idea is to plot the average co2 emission on the worl map per country, we could see that similar countries, 
# i.e. low co2 emitting, moderate emitting and high co2 emitting countries were in close proximity with each other.
#This suggests that Co2 emission has far reaching consequences and co2 levels depends on neighbouring countries
#and economic activities as those countries had similar demographics and economic conditions.

#%matplotlib inline
import os
os.environ["PROJ_LIB"]= "C:\\Users\\pulki\\Anaconda3\\Library\\share";
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4
import pandas as pd
import folium
from folium.plugins import MarkerCluster



#loading the data
country_codes=pd.read_csv("C:\\MS Stony Brook\\sem3\\big data\\project_data\\worldmap_data\\data.csv")
co2_country=pd.read_csv("C:\\MS Stony Brook\\sem3\\big data\\project_data\\worldmap_data\\Co2_per_country.csv")


#converting to string
country_codes['code_3']=country_codes['code_3'].str.strip()
co2_country['code_3']=co2_country['code_3'].str.strip()


#merging the dataframe
df=pd.merge(country_codes,co2_country, on='code_3',how = 'inner')


#color coding on average co2 emission
def color_producer(co2):
    if co2 < 2:
        return 'green'
    elif 2 <= co2 < 10:
        return 'orange'
    else:
        return 'red'
#empty map
world_map= folium.Map(tiles="cartodbpositron",control_scale=True)
marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of average co2 emission
for i in range(len(df)):
        lat = df.iloc[i]['Latitude']
        long = df.iloc[i]['Longitude']
        co2= df.iloc[i]['co2']
        radius=5
        popup_text = """Country : {}<br>
                    Average co2 emission(kt) : {}<br>"""
        popup_text = popup_text.format(df.iloc[i]['Country'],
                                   df.iloc[i]['co2']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, fill_color=color_producer(co2),
                            color = 'grey',
                            fill_opacity=0.7).add_to(marker_cluster)
#show the map
world_map



