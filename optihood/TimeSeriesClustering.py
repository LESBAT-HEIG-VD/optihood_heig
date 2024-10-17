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
        self.meteo_daily = meteo_data
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

    def decompose_series(self, series):
        decomposition = seasonal_decompose(series, model='additive', period=365)
        return decomposition.resid.dropna()

    def dynamic_time_warping(self):
        distance_matrix = np.zeros((len(self.clustering_input), len(self.clustering_input)))
        for i in range(len(self.clustering_input)):
            for j in range(i+1, len(self.clustering_input)):
                distance = dtw.distance(self.clustering_input[i], clustering_input[j])
                distance_matrix[i, j] = distance_matrix[j, i] = distance
        return distance_matrix

    def normalize_data(self, df, clustering_vars):
        scaler = StandardScaler()
        return scaler.fit_transform(df[clustering_vars])

    def apply_pca(self, data, n_components=0.95):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    def do_clustering(self, clustering_vars, method="kmeans", use_dtw=False, use_autoencoder=False, use_seasonal_decomp=False):
        # Resample and process the data
        self.meteo_daily['week_end'] = [1000 if d.weekday() >= 5 else 0 for d in self.meteo_daily.index]
        self.agg_demand_daily = self.agg_demand.resample('D').sum()
        
        try:
            self.cluster_DB = pd.merge(self.meteo_daily.tz_localize(None), self.agg_demand_daily, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(self.meteo_daily, self.agg_demand_daily, how='inner', left_index=True, right_index=True)
        
        self.clustering_input = self.cluster_DB.loc[:, clustering_vars]
        
        if use_seasonal_decomp:
            # Decompose time series and use residuals for clustering
           self.clustering_input = self.clustering_input.apply(self.decompose_series, axis=0)
        
        if use_autoencoder:
            # Use autoencoder for dimensionality reduction
            input_dim = self.clustering_input.shape[1]
            encoding_dim = int(input_dim / 2)
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
            # Use K-Shape Clustering
            self.clustering_input_reshaped = self.clustering_input.reshape((len(self.clustering_input), -1, len(clustering_vars)))
            model = KShape(n_clusters=self.n_clusters)
            self.code_BK = model.fit_predict(self.clustering_input_reshaped)
            self.cluster_centers_ = model.cluster_centers_
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
