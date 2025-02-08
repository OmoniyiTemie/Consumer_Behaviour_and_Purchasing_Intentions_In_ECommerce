# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:02:52 2024

@author: dell
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




os.chdir('C:/Users/dell/Desktop/WIU_ASDA/FALL_2024/DS_485/Project/online_shoppers_intention')


df = pd.read_csv('online_shoppers_intention.csv')        
df



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           DATA PREP
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Checking for null values in the dataset:

null_values = df.isnull().sum()
null_values

# NO NULL VALUES


# DATA TYPE TRANSFORMATION
# Converting 'Weekend' and 'Revenue' to boolean
df['Weekend'] = df['Weekend'].astype(bool)
df['Revenue'] = df['Revenue'].astype(bool)







#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           FIRST RESEARCH QUESTION - What factors influence a shopper's likelihood to make a purchase?
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Feature Variable - Others - x
# Target Variable - "Revenue" - y

X = df.drop(columns=['Revenue', 'Month', 'VisitorType'])
y = df['Revenue']


# 80% training set AND 20% testing set, random seed specification to ensure reproducibility

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#----------------------------------------------------------------------------------------------------------

# To determine the type of Regression to be used, inspecting the possible values/members of the response variable, "Revenue"
unique_values = df['Revenue'].unique()
unique_values

# We have two possible 'categorical' values in the column 'Revenue' - 'True', 'False'. We can use Logistic Regression.

# Using LOGISTIC REGRESSION 

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)

coefficients = model.coef_[0]
coefficients

coeff_df = pd.DataFrame(coefficients, X.columns, columns=['Coefficient'])
print(coeff_df)

# Print classification report and accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Inspecting why we have relatively low precision and recall for the "True" values

true_count = (df['Revenue'] == True).sum()
print(f"Number of 'True' values in Revenue column: {true_count}")

# The model has less TPs because we have a class imbalance, where 'TRUE' values have way lesser representation in the data (1909 observations, out of a total 12331 observations in the whole dataset)




#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           VISUALIZATION
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Assuming 'coeff_df' is the DataFrame with features and their coefficients
coeff_df = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])

# Sort coefficients by absolute value for better visualization
coeff_df = coeff_df.reindex(coeff_df['Coefficient'].abs().sort_values(ascending=False).index)

# Create the bar chart
plt.figure(figsize=(10, 6))
colors = ['green' if coeff > 0 else 'red' for coeff in coeff_df['Coefficient']]  # Green for positive, red for negative
plt.barh(coeff_df.index, coeff_df['Coefficient'], color=colors)

# Add titles and labels
plt.title('Feature Coefficients in Logistic Regression', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add a grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           SECOND RESEARCH QUESTION - Can we identify distinct groups of shoppers based on their browsing behavior?
#------------------------------------------------------------------------------------------------------------------------------------------------------------------


# PRE-PROCESSING THE DATA FROR CLUSTERING

# Features related to browsing behavior include:
features = ['Administrative', 'Administrative_Duration', 'Informational', 
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

# Extract features for clustering
X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#-------------------------------------------


# DETERMINING THE OPTIMAL NUMBER OF CLUSTERS USING THE ELBOW METHOD

# Calculate the within-cluster sum of squares (WCSS) for different cluster numbers
wcss = []

for i in range(1, 11):  # Test clusters from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Clusters', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.grid()
plt.show()

# The elbow point is where the curve flattens out
# The optimal number of clusters here appears to be 3, as that's where the curve starts to flatten.


# Apply K-Means with the optimal number of clusters (e.g., k=3 based on the Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = clusters

# View the first few rows with cluster labels
print(df[['Cluster'] + features].head())


#-------------------------------------------

# VISUALIZING CLUSTERS: Using Principal Component Analysis to reduce the Dimensionality 

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
for cluster in range(3):  # Adjust based on the number of clusters
    plt.scatter(X_pca[df['Cluster'] == cluster, 0], 
                X_pca[df['Cluster'] == cluster, 1], 
                label=f'Cluster {cluster}')

# Add plot details
plt.title('Visualization of Clusters', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# To get the loadings of the each pca component, for better interpretation:
loadings = pd.DataFrame(pca.components_, columns=features, index=['PCA1', 'PCA2'])
print(loadings)

#This analysis shows that PCA2 captures behaviors tied to disengagement rather than purchase intent and PCA1 captures behaviors tied to purchase intent	


# Get the mean feature values for each cluster
cluster_summary = df.groupby('Cluster')[features].mean()                                                                                                                                                                                        
print(cluster_summary)
























