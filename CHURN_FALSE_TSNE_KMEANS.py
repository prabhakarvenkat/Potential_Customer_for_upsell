#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# - Pandas is used for data manipulation and analysis.
# - Standard Scalar is used to standardize features by removing the mean and scaling to unit variance.
# - Kmeans is used to perform kmeans clustering in the features , which is popular algorithm that partitions data into a predefined number of clusters.
# - TSNE is used for dimentionality reduction.
# - Matplotlib is used for visualization.
# - Seaborn provides informative statistical graphics.

# In[3]:


# Load the dataset
file_path = r"C:\Users\manid\OneDrive\Desktop\churn_false.csv"
df = pd.read_csv(file_path)


# - Loading the CSV file into a pandas Dataframe.

# In[4]:


# Ensure 'Churn' column is properly formatted
df['Churn'] = df['Churn'].astype(int)


# - This Converts datatype of the churn column into the integer type.
# After Conversion if the value is 1 then it represents that the customers has churned(i.e left the service)
# if the value is 0 then it represents that the customers has not churned(i.e Retained)

# In[5]:


# Select relevant numerical columns for clustering
numerical_cols = [
    'Account_Length', 'VMail_Message', 'Day_Mins', 'Day_Calls', 'Day_Charge', 
    'Eve_Mins', 'Eve_Calls', 'Eve_Charge', 'Night_Mins', 'Night_Calls', 
    'Night_Charge', 'Intl_Mins', 'Intl_Calls', 'Intl_Charge', 'CustServ_Calls'
]
X = df[numerical_cols]


# - It retrieves the numerical columns in the dataset to perform clustering.

# In[6]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# - StandardScaler creates helps in standardizing the feature by removing the mean and scaling to unit variance.
# - It Centers the data around the mean and standard deviation of 1.(for better performance)
# - fit is used to calculate the mean and standard deviation for each features in the dataset.
# - Transform uses the calculated values to standardize the features in dataset.

# In[7]:


# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# - Here the Elbow Method is used to determine the optimal number of Clusters by plotting the WCSS(Within-cluster sum of squares) for different numbers of clusters.
# - First the WCSS list is created to store the wcss for different number of clusters. The loop here iterates over a range of cluster numbers from 1 to 10 helps to test the number of clusters.
# - Kmeans++ is a method for initializing the centroids that helps in speed up the convergence of the algorithm.
# - Max_iter is set as 300 that the maximum number of times the algorithm run if does not converge before.
# - n_init is set as 10 as the number of times the algorithm will run with different centroids.
# - random_state is set as 42 to show the same output while executing the code for multiple times.

# In[8]:


# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# - By plotting the Elbow Method we can conclude the optimal number of clusters as 3 because the elbow appeared at the point 3 in x-axis

# In[9]:


optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# - Kmeans clustering is applied to the standardized data with optimal number of clusters identified from the elbow method.

# In[10]:


# Perform t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]


# - t-SNE(t-Distributed Stochastic Neighbor Embedding) is a dimentionality reduction technique to visualize the clusters in low dimentions.
# - Here the data is reduced to 2 dimensions to facilate visualization in a 2D plot.
# - standardizing the features before applying the t-SNE helps to improve the result.
# - TSNE1 and TSNE2 is the first and second dimensions of the transformed data.(added as a new column in the dataframe)

# In[11]:


# Define cluster names and colors
cluster_names = {
    0: 'low value Customers',
    1: 'Medium value Customers',
    2: 'high value customers'
}
cluster_colors = {
    'low value Customers': '#2ca02c',   # Blue
    'Medium value Customers': '#ff7f0e',      # Orange
    'high value Customers': '##1f77b4'             # Green
}


# The clusters are named as -> cluster 0 as Potential low value customers and indicated by the colour Green. cluster 1 as Potential Medium value customer indicated by the Blue Colour. cluster 2 as Potential High value customer indicated by the Red colour.

# In[13]:


# Verify and correct mapping from cluster names to colors

# Print existing keys in cluster_colors to debug
print("Available colors:", cluster_colors.keys())

# Ensure that all cluster names are present in cluster_colors
missing_colors = df[~df['Cluster_Name'].isin(cluster_colors.keys())]['Cluster_Name'].unique()
if missing_colors.size > 0:
    print("Missing color entries:", missing_colors)
    # Optionally handle missing colors (e.g., set a default color)
    default_color = 'grey'  # or any other default color
    df['Cluster_Color'] = df['Cluster_Name'].map(lambda x: cluster_colors.get(x, default_color))
else:
    # Map clusters to descriptive names and colors if all keys are present
    df['Cluster_Color'] = df['Cluster_Name'].map(cluster_colors)

# Create a color palette matching the Cluster_Color values
palette = [cluster_colors.get(name, default_color) for name in cluster_names.values()]

# Optional: Display the DataFrame to check results
print(df.head())


# - Maps each customer to a specific name based on their cluster It allows us to see which segment each customer belongs to.
# And then it assigns a color to each customer based on their cluster name It helps in visualizing the data, when we create charts or plots.
# 

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure all cluster names are in the cluster_colors dictionary
missing_colors = df['Cluster_Name'].unique()
for color in missing_colors:
    if color not in cluster_colors:
        # Add missing colors to the cluster_colors dictionary with a default color
        cluster_colors[color] = 'grey'  # Use any default color you prefer

# Plot the clusters using t-SNE
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='TSNE1', y='TSNE2', hue='Cluster_Name', data=df,
    palette=cluster_colors, s=100
)
plt.title('Cluster Visualization using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster')
plt.show()


# - The grey colour indicates potential high value customers(cluster 2)-customers who are most valuable to the business
# - green colour indicates potential low value customers(cluster 0)-customers are likely the least engaged or have the lowest transaction values
# - orange colour indicates potential medium value customers(cluster 1)-customers may have moderate engagement with the business

# In[16]:


# Display cluster statistics
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    print(f"\n📈 {cluster_name} Statistics:")
    print(df[df['Cluster'] == i][numerical_cols].describe())


# In[ ]:





# In[17]:


# Churn rate analysis for each cluster
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    churn_rate = df[df['Cluster'] == i]['Churn'].mean() * 100  # Convert to percentage
    print(f"⚠ Churn Rate for {cluster_name}: {churn_rate:.2f}%")


# - As the dataset is splitted as true and false and it the false part the churn rate is 0 for all the clusters.

# In[18]:


# Count the number of records in each cluster
print('Number of Records in Each Cluster:')
cluster_counts = df['Cluster_Name'].value_counts()
print(cluster_counts)


# - It shows the count of customers in each clusters.

# In[ ]:




