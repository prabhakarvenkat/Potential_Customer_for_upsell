#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\manid\OneDrive\Documents\cleaned_churn dataset cdr.csv"
df = pd.read_csv(file_path)

# Ensure 'Churn' column is properly formatted
df['Churn'] = df['Churn'].astype(int)

# Select relevant numerical columns for clustering
numerical_cols = [
    'Account_Length', 'VMail_Message', 'Day_Mins', 'Day_Calls', 'Day_Charge', 
    'Eve_Mins', 'Eve_Calls', 'Eve_Charge', 'Night_Mins', 'Night_Calls', 
    'Night_Charge', 'Intl_Mins', 'Intl_Calls', 'Intl_Charge', 'CustServ_Calls'
]

X = df[numerical_cols]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Choose the optimal number of clusters (can be determined from the Elbow plot)
optimal_clusters = 3  # Assume 3 clusters based on the Elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters using t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

# Calculate Customer Lifetime Value (CLV)
df['CLV'] = df['Day_Charge'] + df['Eve_Charge'] + df['Night_Charge'] + df['Intl_Charge']

# Define cluster names and colors based on your analysis
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

# Map clusters to descriptive names and colors
df['Cluster_Name'] = df['Cluster'].map(cluster_names)
df['Cluster_Color'] = df['Cluster_Name'].map(lambda x: cluster_colors[x])

# Create a color palette matching the Cluster_Color values
palette = [cluster_colors[name] for name in cluster_names.values()]

# Plotting the t-SNE visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='TSNE1', y='TSNE2', hue='Cluster_Name', data=df, 
    palette=cluster_colors, s=100
)
plt.title('Cluster Visualization using t-SNE')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.legend(title='Cluster')
plt.show()

# Display cluster statistics
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    print(f"\n📈 {cluster_name} Statistics")
    print(df[df['Cluster'] == i][numerical_cols].describe())

# Cluster-wise CLV analysis
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    cluster_clv_value = df[df['Cluster'] == i]['CLV'].mean()
    print(f"💰 Average CLV for {cluster_name}: {cluster_clv_value:.2f}")

# Churn rate analysis for each cluster
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    churn_rate = df[df['Cluster'] == i]['Churn'].mean() * 100  # Convert to percentage
    print(f"⚠️ Churn Rate for {cluster_name}: {churn_rate:.2f}%")

# Count the number of records in each cluster
cluster_counts = df['Cluster_Name'].value_counts()
print("\nNumber of Records in Each Cluster")
print(cluster_counts)

# Save the updated dataframe with clusters and CLV
df.to_csv('clustered_data.csv', index=False)


# In[ ]:




