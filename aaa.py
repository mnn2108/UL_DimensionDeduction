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
from sklearn.neural_network import MLPClassifier

from scipy.stats import kurtosis


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

print ('HELLO Assignment 3 - UL Dim Reduction')
col_names  = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']
pima = pd.read_csv("PhishingData.csv", header=None, names=col_names)

feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
X = pima[feature_cols].to_numpy() # Features
X = X[1:]
y = (pima.Result).to_numpy() # Target variable
y = y[1:]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)




if (0):
	print (' --------  PART 1  CLUSTERING ----------')
	
#	poker = pd.read_csv("poker-hand-training.csv")
#	X = poker.iloc[:,0:-1]
#	y = poker.iloc[:,-1]
#	print (poker.head())
#	print (X.shape)
#	print (y.shape)

	sick = pd.read_csv("dataset_38_sick_cleanup.csv")
	sick_nonan = sick.dropna()
	X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
	X = X_pre.to_numpy() # Features
	y = (sick_nonan.Class).to_numpy() # Target variable

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#	y_train = y_train.to_numpy()
	
	model = KMeans(n_clusters=3, max_iter=500, init='k-means++')
	labels = model.fit_predict(X_train)
	print (labels[0:10])
	print (y_train[0:10])
	print ('\n CLUSTER = 3 ')
	print ('homogeneity_score = ', metrics.homogeneity_score(y_train, labels))
	print ('completeness_score = ', metrics.completeness_score(y_train, labels))
	print ('adjusted_rand_score = ', metrics.adjusted_rand_score(y_train, labels))
	print ('silhouette_score = ', metrics.silhouette_score(y_train.reshape(-1, 1), labels.reshape(-1, 1)))
	
	model = KMeans(n_clusters=5, max_iter=500, init='k-means++')
	labels = model.fit_predict(X_train)
	print ('\n CLUSTER = 5 ')
	print ('homogeneity_score = ', metrics.homogeneity_score(y_train, labels))
	print ('completeness_score = ', metrics.completeness_score(y_train, labels))
	print ('adjusted_rand_score = ', metrics.adjusted_rand_score(y_train, labels))
	print ('silhouette_score = ', metrics.silhouette_score(y_train.reshape(-1, 1), labels.reshape(-1, 1)))

	model = KMeans(n_clusters=9, max_iter=500, init='k-means++')
	labels = model.fit_predict(X_train)
	print ('\n  CLUSTER = 9 ')
	print ('homogeneity_score = ', metrics.homogeneity_score(y_train, labels))
	print ('completeness_score = ', metrics.completeness_score(y_train, labels))
	print ('adjusted_rand_score = ', metrics.adjusted_rand_score(y_train, labels))		
	print ('silhouette_score = ', metrics.silhouette_score(y_train.reshape(-1, 1), labels.reshape(-1, 1)))
	
#	print ('AAA *** labels.inertia_ = ', (labels.inertia_))
	print ('adjusted_mutual_info_score = ', metrics.adjusted_mutual_info_score(y_train, labels))	 

	
#	print ('TESTING: ')
#	print ('homogeneity_score = ', metrics.homogeneity_score([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))
#	print ('completeness_score = ', metrics.completeness_score([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))
#	print ('adjusted_rand_score = ', metrics.adjusted_rand_score([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))	
#	print ('accuracy_score  = ', metrics.accuracy_score ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))	
#	print ('f1_score  = ', metrics.f1_score ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))
	
#	print ('silhouette_score   = ', metrics.silhouette_score  ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))	
#	print ('silhouette_samples    = ', metrics.silhouette_samples   ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))	


	model = GMM(covariance_type = 'diag')
	model.set_params(n_components=3)
	model.fit(X_train)
	labels = model.predict(X_train)	
	print ('\n EM n_component = 3 ')
	print ('homogeneity_score = ', metrics.homogeneity_score(y_train, labels))
	print ('completeness_score = ', metrics.completeness_score(y_train, labels))
	print ('adjusted_rand_score = ', metrics.adjusted_rand_score(y_train, labels))
	
if (1):
	poker = pd.read_csv("poker-hand-training.csv")
	X = poker.iloc[:,0:-1]
	y = poker.iloc[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
	y_train = y_train.to_numpy()
	model = GMM(covariance_type = 'full')
	model.set_params(n_components=3)
	model.fit(X_train, y_train)
	labels = model.predict(X_train)	

	print (max(labels))
	print (model.aic(X_train))
	print (model.bic(X_train))
	print (model.score(X_train, y_train))
	
	
if (0):
	print (' --------  PART 2  DIMENSION REDUCTION and CLUSTERING  ----------')
	sick = pd.read_csv("dataset_38_sick_cleanup.csv")
	sick_nonan = sick.dropna()
	X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
	X = X_pre.to_numpy() # Features
	y = (sick_nonan.Class).to_numpy() # Target variable

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
	
	# PCA
	print (X_train.shape)
	num = 9
	pca = PCA(n_components=num, random_state=0)
	pca_X_tr = pca.fit_transform(X_train)
	print (pca_X_tr.shape)
	ica = FastICA(n_components=num, random_state=0)
	ica_X_tr = ica.fit_transform(X_train)
	print (ica_X_tr.shape)
	rp = SparseRandomProjection(n_components=num, random_state=0, eps=None)
	rp_X_tr = rp.fit_transform(X_train)
	print (rp_X_tr.shape)
	lda = LDA(n_components=None)
	lda_X_tr = lda.fit_transform(X_train, y_train)	
	print (lda_X_tr.shape)
	
	model = KMeans(n_clusters=9, max_iter=500, init='k-means++')
	labels = model.fit_predict(pca_X_tr)
	print (labels.shape)
	print ('\n  CLUSTER = 9 ')
	print ('homogeneity_score = ', metrics.homogeneity_score(y_train, labels))
	print ('completeness_score = ', metrics.completeness_score(y_train, labels))
	print ('adjusted_rand_score = ', metrics.adjusted_rand_score(y_train, labels))		
	print ('silhouette_score = ', metrics.silhouette_score(y_train.reshape(-1, 1), labels.reshape(-1, 1)))	
	labels2 = model.fit(pca_X_tr)
	print ('distortion = ', labels2.inertia_)	
	
	kurt = kurtosis(pca_X_tr, axis=0)
	print (sum(kurt))
	
if (0):
	print (' --------  PART 3  NEURAL NETWORK PERFORMANCE----------')
	
	
	
	
if (0):
	poker = pd.read_csv("poker-hand-training.csv")
	X = poker.iloc[:,0:-1]
	y = poker.iloc[:,-1]
	print (poker.head())
	print (X.shape)
	print (y.shape)
	y = y.to_numpy()
	X = X.to_numpy()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


	# Method 1: DT
	print ('\nMethod 1: Decision Tree')
	clf = DecisionTreeClassifier(max_depth=3)
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



	# Method 2: NN
	print ('\nMethod 2: Neural Network')
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=40)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# Method 3: Boosting
	print ('\nMethod 3: Boosting')
	clf = AdaBoostClassifier(n_estimators=100, random_state=0)
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# Method 4: Support Vector Machine SVM
	print ('\nMethod 4: Support Vector Machine SVM')
	clf = svm.SVC()
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# MMethod 5: k-nearest neighbor
	print ('\nMethod 5: k-nearest neighbor')
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh = neigh.fit(X_train,y_train)
	y_pred = neigh.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))