#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

# In[4]:


file_path = r"C:\Users\manid\OneDrive\Documents\cleaned_churn dataset cdr.csv"
df = pd.read_csv(file_path)


# - Loading the CSV file into a pandas Dataframe.

# In[5]:


numerical_cols = [
    'Account_Length', 'VMail_Message', 'Day_Mins', 'Day_Calls', 'Day_Charge', 
    'Eve_Mins', 'Eve_Calls', 'Eve_Charge', 'Night_Mins', 'Night_Calls', 
    'Night_Charge', 'Intl_Mins', 'Intl_Calls', 'Intl_Charge', 'CustServ_Calls'
]
X = df[numerical_cols]


# - It retrieves the numerical columns in the dataset to perform clustering.

# In[6]:


df['Churn'] = df['Churn'].astype(int)


# - This Converts datatype of the churn column into the integer type.
# - After Conversion if the value is 1 then it represents that the customers has churned(i.e left the service)
# - if the value is 0 then it represents that the customers has not churned(i.e Retained)

# In[7]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# - StandardScaler creates helps in standardizing the feature by removing the mean and scaling to unit variance.
# - It Centers the data around the mean and standard deviation of 1.(for better performance)
# - fit is used to calculate the mean and standard deviation for each features in the dataset.
# - Transform uses the calculated values to standardize the features in dataset.

# In[8]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# - Here the Elbow Method is used to determine the optimal number of Clusters by plotting the WCSS(Within-cluster sum of squares) for different numbers of clusters.
# - First the WCSS list is created to store the wcss for different number of clusters. The loop here iterates over a range of cluster numbers from 1 to 10 helps to test the number of clusters.
# - Kmeans++ is a method for initializing the centroids that helps in speed up the convergence of the algorithm.
# - Max_iter is set as 300 that the maximum number of times the algorithm run if does not converge before.
# - n_init is set as 10 as the number of times the algorithm will run with different centroids.
# - random_state is set as 42 to show the same output while executing the code for multiple times.
# - By plotting the Elbow Method we can conclude the optimal number of clusters as 3 because the elbow appeared at the point 3 in x-axis.

# In[9]:


optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# - Kmeans clustering is applied to the standardized data with optimal number of clusters identified from the elbow method.

# In[10]:


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0] 
df['TSNE2'] = X_tsne[:, 1]


# - t-SNE(t-Distributed Stochastic Neighbor Embedding) is a dimentionality reduction technique to visualize the clusters in low dimentions.
# - Here the data is reduced to 2 dimensions to facilate visualization in a 2D plot.
# - standardizing the features before applying the t-SNE helps to improve the result.
# - TSNE1 and TSNE2 is the first and second dimensions of the transformed data.(added as a new column in the dataframe)

# In[11]:


cluster_names = {
    0: 'Potential Low Value Customers',
    1: 'Potential Medium Value Customers',
    2: 'Potential High Value Customers'
}
cluster_colors = {
    'Potential Low Value Customers': '#32CD32',   # Green
    'Potential Medium Value Customers': '#4682B4', # Blue
    'Potential High Value Customers': '#FF6347'   # Red
}


# The clusters are named as -> cluster 0 as Potential low value customers and indicated by the colour Green.
# cluster 1 as Potential Medium value customer indicated by the Blue Colour.
# cluster 2 as Potential High value customer indicated by the Red colour.

# In[12]:


df['Cluster_Name'] = df['Cluster'].map(cluster_names)
df['Cluster_Color'] = df['Cluster_Name'].map(lambda x: cluster_colors[x])


# - Maps each customer to a specific name based on their cluster
# It allows us to see which segment each customer belongs to.
# - And then it assigns a color to each customer based on their cluster name
# It helps in visualizing the data, when we create charts or plots. 

# In[13]:


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


# - The red colour indicates potential high value customers(cluster 2)-customers who are most valuable to the business
# - green colour indicates potential low value customers(cluster 0)-customers are likely the least engaged or have the lowest transaction values
# - blue colour indicates potential medium value customers(cluster 1)-customers may have moderate engagement with the business

# In[15]:


# Display cluster statistics
for i in range(optimal_clusters):
   cluster_name = cluster_names.get(i, f'Cluster {i}')
   print(f"\n📈 {cluster_name} Statistics:")
   print(df[df['Cluster'] == i][numerical_cols].describe())


# - 

# In[16]:


# Churn rate analysis for each cluster
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    churn_rate = df[df['Cluster'] == i]['Churn'].mean() * 100  
    print(f"⚠ Churn Rate for {cluster_name}: {churn_rate:.2f}%")


# - Clusters with high churn rates show where customers are unhappy, indicating they have issues needing attention.
# - Low churn rates indicates stable and possibly satisfied customers.
# - Varying churn rates suggest that different customer groups have different needs and behaviors.
# - For medium level churn rate customers we should identify the areas of improvement and develop specific strategies to keep these customers engaged
# - Cluster 1 is the best choice for upselling because these customers are likely to stay and are already engaged with your product.
# Cluster 2 could be a good option if they show interest for increased spending, but be cautious since they have a moderate risk of leaving.
# Cluster 3 is not ideal for upselling because these customers are more likely to leave soon. It's better to focus on keeping them rather than trying to sell more to them..

# In[17]:


# Count the number of records in each cluster
print('Number of Records in Each Cluster:')
cluster_counts = df['Cluster_Name'].value_counts()
print(cluster_counts)


# - From this we can identify the number of customers in each clusters.
# - Potential High Value Customers      19939
# Potential Low Value Customers       - 9638
# Potential Medium Value Customers    - 6270

# In[ ]:




