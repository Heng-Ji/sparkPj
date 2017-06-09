
# coding: utf-8

# In[249]:

import arcgis
from arcgis.gis import GIS
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os
import pandas as pd
import numpy as np
# spark_path = r'C:\Users\heng9188\Downloads\spark-2.1.1-bin-hadoop2.7'
# os.environ['SPARK_HOME'] = spark_path
# sys.path.append(spark_path + r'\python')

# try:
#     from pyspark import SparkContext
#     from pyspark.ml.clustering import KMeans
#     print('Successfully importing Spark Modules')
# except ImportError as e:
#     print(e)


# In[285]:

class project(object):
    def __init__(self, input_path):
        try:
            self.raw_data = pd.read_csv(input_path, low_memory=False)
            print('successfully loading data')
        except:
            print('failed loading data')
#         self.raw_data = self.raw_data[['datetime', 'shape', 'duration', 'latitude', 'longitude']]
#         self.parse_data('datetime', 'datetime', 'datetime')
#         self.parse_data('shape', 'shape', 'string')
#         self.parse_data('duration', 'duration', 'numeric')
#         self.parse_data('latitude', 'latitude', 'numeric')
#         self.raw_data.dropna(inplace=True)
#         self.raw_data = self.raw_data[(self.raw_data['duration'] != 0) & (self.raw_data['latitude'] != 0)]
        
        
#         raw_data['datetimenew'] = raw_data['datetime'].astype('str')
#         raw_data['latitude'] = pd.to_numeric(raw_data['latitude'], errors='coerce')
    def parse_data(self, input_col, output_col, to_type):
        if to_type == 'datetime':
            self.raw_data[output_col] = pd.to_datetime(self.raw_data[input_col], errors='coerce')
        elif to_type == 'string':
            self.raw_data[output_col] = self.raw_data[input_col].astype(str)
        elif to_type == 'numeric':
            self.raw_data[output_col] = pd.to_numeric(self.raw_data[input_col], errors='coerce')
    
    def kmeans_model(self, start, end):
        lati = self.to_array('latitude')
        long = self.to_array('longitude')
        geo_data = list(zip(lati, long))
        prev = float('inf')
        diff = 0.01
        for i in range(start, end):
            kmeans = KMeans(n_clusters=i).fit(geo_data)
            dist = self.get_distance(kmeans, geo_data)
            if i != start and prev > dist and (prev - dist) / prev < diff:
                self.raw_data['cluster'] = kmeans.predict(geo_data)
                break
            prev = dist
        
    
    def to_array(self, col):
        result = self.raw_data.as_matrix(columns=[col])
#         result = [item for sublist in result.tolist() for item in sublist]
        result = list(map(lambda x: x[0], result.tolist()))
        return result
        
        
    def get_distance(self, model, data):
        res = model.score(data)
        return np.sqrt(-(res / len(data)))
# #         print(model.cluster_centers_)
        
    
    def output_data(self, path):
        self.raw_data.to_csv(path, index=False)
        


# In[286]:

def main():
    intput_path = 'C:/Users/heng9188/IdeaProjects/spark_project/py.csv'
    output_path = 'C:/Users/heng9188/IdeaProjects/spark_project/py.csv'
    pj = project(intput_path)
#     pj.kmeans_model(10)
    pj.kmeans_model(70, 101)


# In[287]:

if __name__ == '__main__':
    main()


# In[ ]:



