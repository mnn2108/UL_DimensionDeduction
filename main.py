# main.py

import pandas as pd
import numpy as np
import math 
import time

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics, decomposition
from sklearn.model_selection import learning_curve, train_test_split

from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



print ('HELLO Assignment 3 - UL Dim Reduction')


sick = pd.read_csv("dataset_38_sick_cleanup.csv")
sick_nonan = sick.dropna()
X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
X = X_pre.to_numpy() # Features
y = (sick_nonan.Class).to_numpy() # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
	



if (1):
	print (' --------  PART 1  CLUSTERING ----------')
	x = range(1, 10, 1)
	homogeneity_scores = np.zeros(len(x), 2)
	completeness_scores = np.zeros(len(x), 2)
	adjusted_rand_scores = np.zeros(len(x), 2)
	silhouette_scores  = np.zeros(len(x), 2)
	distortion_scores = np.zeros(len(x), 1)
	aic_scores = np.zeros(len(x), 1)
	bic_scores = np.zeros(len(x), 1)
	ll_scores = np.zeros(len(x), 1)
	
	
	for num in x: 
		model = KMeans(n_clusters=num, max_iter=500, init='k-means++')
		labels = model.fit_predict(X_train)
		
	

	
	
if (0):
	print (' --------  PART 2  DIMENSION REDUCTION and CLUSTERING  ----------')

	
	
if (0):
	print (' --------  PART 3  NEURAL NETWORK PERFORMANCE----------')