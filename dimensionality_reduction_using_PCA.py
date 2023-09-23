#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:03:43 2023

@author: miguelangelsuevispacheco
"""

#import the necessary libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns

#load the MNIST dataset.
raw_data, raw_labels = fetch_openml('mnist_784', version=1, return_X_y=True)
print(raw_data.shape)
print(np.unique(raw_labels)) # we can see all the labels not repeted

# Display each digit
    # For what we want to do, 70 000 is way too much, 
    # so we're going to start by selecting a subset of the dataset
nsamples = 5000
data = raw_data[:nsamples]
labels = raw_labels[:nsamples]



# Then, we do a bit of preprocessing:
for i in range(10):
    print(i)
    digit = data.iloc[i]
    digit_pixels = np.array(digit).reshape(28, 28)
    plt.imshow(digit_pixels)
    plt.title('Number: {}'.format(labels[i]))
    plt.axis('off')
    plt.show()
    

# Perform PCA with 2 components
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(raw_data)

# Get the first and second principal components
first_pc = pca.components_[0]
second_pc = pca.components_[1]

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Output the first and second principal components
print("First Principal Component:")
print("")
print(first_pc)

print("---------------------------------------------------------------------")

print("\nSecond Principal Component:")
print("")
print(second_pc)

print("/--------------------------------------------------------------------")

# Output explained variance ratio
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# Create a 1D hyperplane using the first principal component
projection_1d_first = np.dot(raw_data, first_pc)

# Create a 1D hyperplane using the second principal component
projection_1d_second = np.dot(raw_data, second_pc)

# Plot the projections
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(projection_1d_first, np.zeros_like(projection_1d_first), c=raw_labels.astype(int), cmap='viridis', s=1)
plt.title("Projection onto 1D Hyperplane (1st Principal Component)")
plt.xlabel("Projection Value")
plt.ylabel("")

plt.subplot(1, 2, 2)
plt.scatter(projection_1d_second, np.zeros_like(projection_1d_second), c=raw_labels.astype(int), cmap='viridis', s=1)
plt.title("Projection onto 1D Hyperplane (2nd Principal Component)")
plt.xlabel("Projection Value")
plt.ylabel("")

plt.tight_layout()
plt.show()

print("/--------------------------------------------------------------------")

# Define the number of dimensions you want to reduce to
n_dimensions = 154

# Create an IncrementalPCA instance
incremental_pca = IncrementalPCA(n_components=n_dimensions, batch_size=500)
X_ipca = incremental_pca.fit_transform(raw_data)

# Display original and compressed digits
n = 10  # Number of digits to display
original_images = raw_data[:n]
compressed_images = incremental_pca.inverse_transform(X_ipca[:n])

plt.figure(figsize=(12, 4))
for i in range(n):
    
    digit_original = original_images.iloc[i]
    digit_original_pixels = np.array(digit_original).reshape(28, 28)
    plt.subplot(2, n, i + 1)
    plt.imshow(digit_original_pixels, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.show()

    digit_compressed = compressed_images[i]
    digit__compressed_pixels = np.array(digit_compressed).reshape(28, 28)
    plt.subplot(2, n, i + 1)
    plt.imshow(digit__compressed_pixels, cmap='gray')
    plt.title('Compressed')
    plt.axis('off')
    plt.show()
    
print("--------------------------------------------------------------------")
print(":::::::::::::::::::::: QUESTION 2 ::::::::::::::::::::::::::::::::::")
print("--------------------------------------------------------------------")

from sklearn.decomposition import KernelPCA


# Generate a Swiss roll dataset with 1000 samples
n_samples = 1000
X, color = make_swiss_roll(n_samples=n_samples, random_state=0)


# Define class labels based on color ranges (you can adjust these ranges)
color_classes = np.digitize(color, bins=[-10, 0, 10, 20, 30])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, color_classes, test_size=0.2, random_state=42)

# Plot the Swiss roll
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Dataset")
plt.show()

# Perform Kernel PCA with a linear kernel
kpca = KernelPCA(kernel='linear', n_components=2)
Linear_kpca = kpca.fit_transform(X)

# Perform Kernel PCA with an RBF (Gaussian) kernel
kpca = KernelPCA(kernel='rbf', gamma=0.04, n_components=2)
rbf_kpca = kpca.fit_transform(X)

# Perform Kernel PCA with a Sigmoid kernel
kpca = KernelPCA(kernel='sigmoid', gamma=0.001, coef0=1, n_components=2)
sigmoid_kpca = kpca.fit_transform(X)


# Plot the result of kPCA Linear in 2D
plt.figure(figsize=(8, 6))
plt.scatter(Linear_kpca[:, 0], Linear_kpca[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Kernel PCA with Linear Kernel")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


# Plot the result of kPCA RBF in 2D
plt.figure(figsize=(8, 6))
plt.scatter(rbf_kpca[:, 0], rbf_kpca[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Kernel PCA with RBF Kernel")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# Plot the result of kPCA Sigmoid in 2D
plt.figure(figsize=(8, 6))
plt.scatter(sigmoid_kpca[:, 0], sigmoid_kpca[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Kernel PCA with Sigmoid Kernel")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


# Create a pipeline with Kernel PCA and Logistic Regression
clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

param_grid = [{
       "kpca__gamma": np.linspace(0.03, 0.05, 10),
       "kpca__kernel": ["linear","rbf","sigmoid"]
    }]


grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Fit the model with the best parameters to the full training set
best_pipe = grid_search.best_estimator_
best_pipe.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = best_pipe.score(X_test, y_test)
print("Test Accuracy:", accuracy)

# Get the results of GridSearchCV
results = grid_search.cv_results_
param_grid = grid_search.param_grid

# Get the results of GridSearchCV
results = grid_search.cv_results_

# Extract the mean test scores
mean_test_scores = np.array(results['mean_test_score'])

# Reshape the scores to match the parameter grid
scores = mean_test_scores.reshape(len(param_grid[0]['kpca__kernel']), len(param_grid[0]['kpca__gamma']))

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.set()
sns.heatmap(scores, annot=True, fmt='.3f', xticklabels=param_grid[0]['kpca__gamma'], yticklabels=param_grid[0]['kpca__kernel'], cmap='viridis')
plt.xlabel('Gamma')
plt.ylabel('Kernel')
plt.title('Grid Search Results (Accuracy)')
plt.show()

# Print the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)