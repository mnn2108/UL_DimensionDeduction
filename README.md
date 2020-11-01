# UL_DimensionDeduction

README.txt
Author: MinhTrang (Mindy) Nguyen
Date: Oct 30, 2020


ASSIGNMENT 3: UNSUPERVISED LEARNING and DIMENSION REDUCTION


REQUIREMENTS:

    - pandas 1.0.5
    - numpy 1.17.0
    - sklearn 0.23.2
    - matplotlib 3.2.2


DATASETS:
Dataset 1: dataset_38_sick_cleanup.csv 
Dataset 2: poker-hand-training.csv


CODE DESCRIPTIONS:

Source code is in:
https://github.com/mnn2108/UL_DimensionReduction

The project was divided into 5 parts. All 5 parts are in main.py

Part 1 is to run 2 clustering algorithms on 2 datasets. The two algorithms are:
	1. K-mean clustering        (KM)
	2. Expectation Maximization (EM)

Part 2 is to apply the dimesion reduction algorithms on 2 datasets. Here are the 4 dimesion reduction algorithms:
	1. (PCA)
	2. (ICA)
	3. (RP)
	4. (LDA)


Part 3 is rerun the cluster algorithm on the data after applying the dimension reduction for both datasets.
	1a. PCA - KM
	1b. PCA - EM
	2a. ICA - KM
	2b. ICA - EM
	3a. RP - KM
	3b. RP - EM
	4a. LDA - KM
	4b. LDA - EM

Part 4 is to apply dimension reduction to one of my dataset which is would use the dataset_38_sick_cleanup.csv since I used that one in my Assignment 1. I then rerun my neural network learner on the new projected data


Part 5 is to apply the cluster algorithms to the data above. That would generate a new set of features, and use those features combined with the original data to rerun the neural network learner again.


HOW TO RUN:
- Dataset 1 - Sickness :
  https://github.com/mnn2108/UL_DimensionReduction/blob/main/UL_DimReduction.ipynb 
- Dataset 2 - Poker hand : 
  https://github.com/mnn2108/UL_DimensionReduction/blob/main/UL_DimReduction-%20Dat2-Poker.ipynb



NOTES:
I only show the source code, input file, and a README in this repository.
All the output data and charts can be regenerate from the codes above.



REFERENCE WEBSITES:

-	https://www.cc.gatech.edu/~isbell/tutorials/mimic-tutorial.pdf
-	https://classroom.udacity.com/
- 	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- 	https://scikit-learn.org/stable/modules/mixture.html
