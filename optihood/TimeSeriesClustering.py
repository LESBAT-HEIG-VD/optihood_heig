# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:38:05 2024

@author: stefano.pauletta
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans
from dtaidistance import dtw
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesClustering:

    def __init__(self, meteo_data, demand_data, n_clusters=5):
        self.meteo = meteo_data
        self.agg_demand = demand_data
        self.n_clusters = n_clusters
        self.cluster_DB = None
        self.code_BK = None
        self.meteo_cluster = None
        self.results = None

    def build_autoencoder(self, input_dim, encoding_dim):
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
        decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder
    
    def build_autoencoder_h(self, input_shape, encoding_dim):
        # Input layer (3D: samples, time steps, features)
        input_layer = tf.keras.layers.Input(shape=input_shape)  # e.g., (None, 24, 11)
    
        # Flatten the input (if necessary for dense layer)
        # flattened = tf.keras.layers.Flatten()(input_layer)
    
        # Feature reduction using a dense layer
        reduced = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = tf.keras.layers.Dense(input_shape[1], activation='sigmoid')(reduced)
        # Reshape back to original time dimension with reduced features
        # reshaped_output = tf.keras.layers.Reshape((input_shape[0], encoding_dim))(reduced)
    
        # Autoencoder model
        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')  # Consider using DTW loss
    
        return autoencoder


    def decompose_series(self, series):
        decomposition = seasonal_decompose(series, model='additive', period=365)
        return decomposition.resid.dropna()

    def dynamic_time_warping(self,clustering_input):
        distance_matrix = np.zeros((len(clustering_input), len(clustering_input)))
        for i in range(len(clustering_input)):
            for j in range(i+1, len(clustering_input)):
                distance = dtw.distance(clustering_input[i], clustering_input[j])
                distance_matrix[i, j] = distance_matrix[j, i] = distance
        return distance_matrix

    def normalize_data_hh(self, df ):
        scaler = StandardScaler()
        return scaler.fit_transform(df)
    
    def normalize_data(self, df ,variables):
        scaler = StandardScaler()
        return scaler.fit_transform(df.loc[:,variables])

    def apply_pca(self, data, n_components=0.95):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    def do_clustering(self, clustering_vars):
        # Resample and process the data
        self.meteo_daily=self.meteo.resample('D').agg({'tre200h0': 'mean',
                                                            'gls': 'sum',
                                                            'str.diffus': 'sum',
                                                            'ground_temp': 'mean',
                                                            'pressure': 'mean'})
        self.meteo_daily['week_end'] = [1000 if d.weekday() >= 5 else 0 for d in self.meteo_daily.index]
        self.agg_demand_daily = self.agg_demand.resample('D').sum()
        
        try:
            self.cluster_DB = pd.merge(self.meteo_daily.tz_localize(None), self.agg_demand_daily, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(self.meteo_daily, self.agg_demand_daily, how='inner', left_index=True, right_index=True)
        
        self.clustering_input = self.cluster_DB.loc[:, clustering_vars]
        
        # if use_seasonal_decomp:
        #     #####♣NOT WORKING#########
        #     # Decompose time series and use residuals for clustering
        #     self.clustering_input = self.clustering_input.apply(self.decompose_series, axis=0)
        
        pca = PCA()
        pca.fit(self.clustering_input)
        # Calculate explained variance ratio for each principal component
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        # Choose the number of components that explains 95% of variance
        optimal_components = np.argmax(explained_variance >= 0.95) + 1
        encoding_dim = optimal_components 
        # Use autoencoder for dimensionality reduction
        input_dim = self.clustering_input.shape[1]
        
        

        # Normalize data and apply PCA if no autoencoder is used
        # self.clustering_input = self.normalize_data(self.clustering_input, clustering_vars)
        # self.clustering_input = self.apply_pca(self.clustering_input,optimal_components)


        # Use regular KMeans clustering
        model = KMeans(n_clusters=self.n_clusters)
        self.code_BK = model.fit_predict(self.clustering_input)
        # Calculate the cluster centers
        self.cluster_centers_ = np.zeros((self.n_clusters, self.clustering_input.shape[1]))
    
        # For each cluster, find the center
        for i in range(self.n_clusters):
            cluster_members = self.clustering_input[self.code_BK == i]
            # Compute the average of the cluster members as the center
            self.cluster_centers_[i] = np.mean(cluster_members, axis=0)
    # Post-process and generate clustering results
        # self.postprocessing_clusters()
        return None
    
    # Define the function to flatten the 24-hour data into a single row per day
    def flatten_daily_profiles(self,group):
        # Convert the 24-hour group to a single 1D array (24 values per feature)
        return pd.Series(group.values.flatten())
    
    def do_clustering_hh(self, clustering_vars):
        # Resample and process the data
        self.meteo['week_end'] = [1000 if d.weekday() >= 5 else 0 for d in self.meteo.index]
        n_features = len( clustering_vars) 
        try:
            self.cluster_DB = pd.merge(self.meteo.tz_localize(None), self.agg_demand, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(self.meteo, self.agg_demand, how='inner', left_index=True, right_index=True)
        
        self.cluster_DB.drop(columns=['time.yy', 'time.mm', 'time.dd', 'time.hh'],inplace=True)
        # Reshape the data into daily profiles (one row per day, 24 columns for each variable)
        self.cluster_DB = self.cluster_DB.loc[:,clustering_vars].sort_index()
        self.reshaped_cl_data = self.cluster_DB.values.reshape(365, 24, n_features)
        daily_weather_profiles = self.cluster_DB.loc[:,clustering_vars].resample('D').apply(self.flatten_daily_profiles)  #.apply(lambda x: x.values).apply(pd.Series)
        # Renaming columns for clarity (24 hours for each variable)
        hourly_columns = [f'{var}_Hour_{i}' for var in clustering_vars for i in range(24)]
        daily_weather_profiles.columns = hourly_columns
        
        self.clustering_input=daily_weather_profiles

        n_days = len(np.unique(pd.to_datetime(self.cluster_DB.index).date))  # Count the number of unique days

         
        normalized_weather_profiles = self.normalize_data_hh(self.clustering_input)#daily_weather_profiles)
        # Convert back to a DataFrame
        self.clustering_input = pd.DataFrame(normalized_weather_profiles, index=daily_weather_profiles.index, columns=daily_weather_profiles.columns)

        model = TimeSeriesKMeans(n_clusters=self.n_clusters,  random_state=42) #metric="dtw",
        self.code_BK = model.fit_predict(self.reshaped_cl_data)
        self.cluster_centers_ = np.zeros((self.n_clusters, 24,n_features))
        for i in range(self.n_clusters):
            cluster_members = self.reshaped_cl_data[self.code_BK == i]
            self.cluster_centers_[i] = np.mean(cluster_members, axis=0)

            
        return None
    
    
