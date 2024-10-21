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

    def normalize_data(self, df ):
        scaler = StandardScaler()
        return scaler.fit_transform(df)

    def apply_pca(self, data, n_components=0.95):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    def do_clustering(self, clustering_vars, method="kmeans", use_dtw=False, use_autoencoder=False, use_seasonal_decomp=False):
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
        
        if use_seasonal_decomp:
            #####♣NOT WORKING#########
            # Decompose time series and use residuals for clustering
            self.clustering_input = self.clustering_input.apply(self.decompose_series, axis=0)
        
        pca = PCA()
        pca.fit(self.clustering_input)
        # Calculate explained variance ratio for each principal component
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        # Choose the number of components that explains 95% of variance
        optimal_components = np.argmax(explained_variance >= 0.95) + 1
        encoding_dim = optimal_components 
        # Use autoencoder for dimensionality reduction
        input_dim = self.clustering_input.shape[1]
        
        
        if use_autoencoder:                        
            autoencoder = self.build_autoencoder(input_dim, encoding_dim)
            autoencoder.fit(self.clustering_input, self.clustering_input, epochs=50, batch_size=32, shuffle=True)
            encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
            self.clustering_input = encoder_model.predict(self.clustering_input)
        else:
            # Normalize data and apply PCA if no autoencoder is used
            self.clustering_input = self.normalize_data(self.clustering_input, clustering_vars)
            self.clustering_input = self.apply_pca(self.clustering_input)

        if use_dtw:
            # Use Dynamic Time Warping (DTW) for distance calculation
            distance_matrix = self.dynamic_time_warping(self.clustering_input)
            model = KMeans(n_clusters=self.n_clusters)
            self.code_BK = model.fit_predict(distance_matrix)
            # Calculate the cluster centers
            self.cluster_centers_ = np.zeros((self.n_clusters, self.clustering_input.shape[1]))
        
            # For each cluster, find the center
            for i in range(self.n_clusters):
                cluster_members = self.clustering_input[self.code_BK == i]
                # Compute the average of the cluster members as the center
                self.cluster_centers_[i] = np.mean(cluster_members, axis=0)
        elif method == "kshape":
            # Adjust reshaping based on the new latent space dimension after autoencoder
            n_days = self.clustering_input.shape[0]  # e.g., 365
            if use_autoencoder:
                latent_dim = self.clustering_input.shape[1]  # latent space dimension after autoencoder
                # KShape expects 3D array: (n_samples, n_timestamps, n_features)
                self.clustering_input_reshaped = self.clustering_input.reshape((n_days,1, latent_dim))  # Reshape to (n_days, latent_dim, 1)
            else:
                latent_dim = self.clustering_input.shape[1]  # original number of features
                self.clustering_input_reshaped = self.clustering_input.reshape((n_days,1, latent_dim))  # Reshape to (n_days, n_features, 1)
            
            model = KShape(n_clusters=self.n_clusters)
            self.code_BK = model.fit_predict(self.clustering_input_reshaped)
            self.cluster_centers_ = np.array([np.mean(self.clustering_input_reshaped[self.code_BK == i], axis=0) for i in range(self.n_clusters)])
        else:
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
    
    def do_clustering_hh(self, clustering_vars, method="kmeans", use_dtw=False, use_autoencoder=False, use_seasonal_decomp=False):
        # Resample and process the data
        self.meteo['week_end'] = [1000 if d.weekday() >= 5 else 0 for d in self.meteo.index]
        
        try:
            self.cluster_DB = pd.merge(self.meteo.tz_localize(None), self.agg_demand, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(self.meteo, self.agg_demand, how='inner', left_index=True, right_index=True)
        
        self.cluster_DB.drop(columns=['time.yy', 'time.mm', 'time.dd', 'time.hh'],inplace=True)
        # Reshape the data into daily profiles (one row per day, 24 columns for each variable)
        daily_weather_profiles = self.cluster_DB.loc[:,clustering_vars].resample('D').apply(self.flatten_daily_profiles)  #.apply(lambda x: x.values).apply(pd.Series)
        # Renaming columns for clarity (24 hours for each variable)
        hourly_columns = [f'{var}_Hour_{i}' for var in clustering_vars for i in range(24)]
        daily_weather_profiles.columns = hourly_columns
        
        self.clustering_input=daily_weather_profiles
        n_features = len( clustering_vars) 
        
        # Convert to a numpy array
        # self.clustering_input = np.array(self.clustering_input.tolist()) 
        
        # Calculate n_days explicitly based on the number of grouped days
        n_days = len(np.unique(pd.to_datetime(self.cluster_DB.index).date))  # Count the number of unique days

         
        # Seasonal Decomposition (optional)
        if use_seasonal_decomp:
            self.clustering_input = np.array([self.decompose_series(pd.Series(profile)).values for profile in self.clustering_input])
        
        # Autoencoder (optional)
        if use_autoencoder:
            # Reshape the data to (n_days, 24, n_features) before feeding it to the autoencoder
            self.clustering_input_reshaped = self.clustering_input.reshape((n_days, 24, n_features))
            
            # Build and train the autoencoder for feature reduction, not time reduction
            input_dim = self.clustering_input_reshaped.shape[2]  # e.g., 24 * number of original features (hours per day * num of variables)
            # Perform PCA
            pca = PCA()
            pca.fit(self.clustering_input)
            # Calculate explained variance ratio for each principal component
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            # Choose the number of components that explains 95% of variance
            optimal_components = np.argmax(explained_variance >= 0.95) + 1

            encoding_dim = optimal_components  # Reduce to 5 features while keeping 24 hours
            
            input_shape1 = (24, input_dim)  # For example, (24, 11)
            # Build and train the autoencoder for feature reduction, not time reduction
            autoencoder = self.build_autoencoder_h(input_shape=input_shape1, encoding_dim=encoding_dim)
            # Train the autoencoder using the original data
            autoencoder.fit(self.clustering_input_reshaped, self.clustering_input_reshaped, 
                epochs=50, batch_size=32, shuffle=True)
            # After training, use the encoder part of the model to reduce the dimensionality
            # Extract the encoder from the autoencoder
            encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
    
            
            # Apply the encoder to transform the data to the reduced feature space
            self.clustering_input_reshaped = encoder_model.predict(self.clustering_input_reshaped)  # Transformed data in reduced dimension
            
            # After encoding, the shape will be (n_days, 24, 5)
            # self.clustering_input = self.clustering_input_reshaped.reshape((n_days, 24 * encoding_dim))
            self.clustering_input = self.clustering_input_reshaped
        else:
            normalized_weather_profiles = self.normalize_data(daily_weather_profiles)
            # Convert back to a DataFrame
            self.clustering_input = pd.DataFrame(normalized_weather_profiles, index=daily_weather_profiles.index, columns=daily_weather_profiles.columns)
                        
            
            
            # Apply PCA if not using autoencoder
           # self.clustering_input = self.apply_pca(self.clustering_input)
        
        # DTW or KShape handling
        if use_dtw:
            self.clustering_input_reshaped = self.clustering_input.reshape((n_days, 24, encoding_dim))
            
            distance_matrix = np.zeros((n_days, n_days))
            for i in range(n_days):
                for j in range(i + 1, n_days):
                    distance = dtw.distance(self.clustering_input_reshaped[i].flatten(), self.clustering_input_reshaped[j].flatten())
                    distance_matrix[i, j] = distance_matrix[j, i] = distance
            
            model = KMeans(n_clusters=self.n_clusters)
            self.code_BK = model.fit_predict(distance_matrix)
            
            self.cluster_centers_ = np.zeros((self.n_clusters, 24, n_features))
            for i in range(self.n_clusters):
                cluster_members = self.clustering_input_reshaped[self.code_BK == i]
                self.cluster_centers_[i] = np.mean(cluster_members, axis=0)
        
        elif method == "kshape":
            if use_autoencoder:
                latent_dim = self.clustering_input.shape[1]
                self.clustering_input_reshaped = self.clustering_input.reshape((n_days, 24, latent_dim))
            else:
                self.clustering_input_reshaped = self.clustering_input.reshape((n_days, 24, n_features))
            
            model = KShape(n_clusters=self.n_clusters)
            self.code_BK = model.fit_predict(self.clustering_input_reshaped)
            
            self.cluster_centers_ = np.array([np.mean(self.clustering_input_reshaped[self.code_BK == i], axis=0) for i in range(self.n_clusters)])
        
        else:
            self.clustering_input_reshaped = self.clustering_input.reshape((n_days, 24, n_features))
            clustering_input_flattened = self.clustering_input_reshaped.reshape((n_days, -1))
            
            model = KMeans(n_clusters=self.n_clusters,random_state=42)
            model.fit(self.clustering_input)
            self.code_BK = model.fit_predict(self.clustering_input)
            #cluster_labels = model.labels_
            
            self.cluster_centers_ = np.zeros((self.n_clusters, 24, n_features))
            for i in range(self.n_clusters):
                cluster_members = self.clustering_input_reshaped[self.code_BK == i]
                self.cluster_centers_[i] = np.mean(cluster_members, axis=0)
        
        return None
    
    def do_clustering_h(self, clustering_vars):
        # Resample and process the data
        self.meteo['week_end'] = [1000 if d.weekday() >= 5 else 0 for d in self.meteo.index]
        
        try:
            self.cluster_DB = pd.merge(self.meteo.tz_localize(None), self.agg_demand, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(self.meteo, self.agg_demand, how='inner', left_index=True, right_index=True)
        
        self.cluster_DB.drop(columns=['time.yy', 'time.mm', 'time.dd', 'time.hh'],inplace=True)
        # Reshape the data into daily profiles (one row per day, 24 columns per variable)
        daily_weather_profiles = self.cluster_DB.loc[:, clustering_vars].resample('D').apply(self.flatten_daily_profiles)
        
        # Create new column names for clarity, indicating hours for each variable
        hourly_columns = [f'{var}_Hour_{i}' for var in clustering_vars for i in range(24)]
        daily_weather_profiles.columns = hourly_columns
        
        # Normalize the data for clustering
        self.clustering_input = self.normalize_data(daily_weather_profiles)
        
        # Apply PCA to reduce the dimensionality while keeping the 24-hour structure
        pca_model = PCA(n_components=0.95)
        pca_result = pca_model.fit_transform(self.clustering_input)
        
        # Perform clustering on the reduced features
        model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.code_BK = model.fit_predict(pca_result)
        
        # Calculate cluster centers in the original feature space (24-hour profile)
        self.cluster_centers_ = np.zeros((self.n_clusters, 24, len(clustering_vars)))
        
        # Inverse transform the PCA results to get cluster centers in the original feature space
        cluster_centers_pca = model.cluster_centers_
        cluster_centers_original_space = pca_model.inverse_transform(cluster_centers_pca)
        
        # Reshape the cluster centers back to the (24 hours, n_features) form for each cluster
        for i in range(self.n_clusters):
            self.cluster_centers_[i] = cluster_centers_original_space[i].reshape(24, len(clustering_vars))
    
        return None


    # def postprocessing_clusters(self):
    #     logging.info("Cluster post-processing started")
    #     flows = [x for x in self._optimizationResults.keys() if x[1] is not None]
    #     for flow in flows:
    #         extrapolated_results = None
    #         dailyIndex = pd.period_range(datetime(self._timeIndexReal.year[0],
    #                                               self._timeIndexReal.month[0],
    #                                               self._timeIndexReal.day[0]),
    #                                      freq='D', periods=(self._timeIndexReal[-1] -
    #                                                         self._timeIndexReal[0] +
    #                                                         timedelta(seconds=3600)).days)
    #         for i in range(len(dailyIndex)):
    #             temp = self._optimizationResults[flow]['sequences'].loc[
    #                 self._clusterDate[
    #                     list(clusterSize.keys())[clusterBook.iloc[i, 0] - 1]], :]
    #             if extrapolated_results is not None:
    #                 extrapolated_results = pd.concat([extrapolated_results, temp])
    #             else:
    #                 extrapolated_results = temp
    #         extrapolated_results.index = self._timeIndexReal
    #         extrapolated_results.columns = ['flow']
    #         self._optimizationResults[flow]['sequences'] = extrapolated_results

    #     logging.info("Cluster post-processing finished")
    #     return None
