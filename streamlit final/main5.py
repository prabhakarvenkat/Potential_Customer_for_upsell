import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Streamlit app setup
st.set_page_config(page_title="Customer Clustering & Churn Analysis", layout="wide")
st.title("Customer Clustering & Churn Analysis")

# Load your dataset
@st.cache_data
def load_data():
    # Replace with your actual data loading code
    df = pd.read_csv(r"C:\Users\manid\OneDrive\Desktop\cleaned_churn_cdr_ds.csv")
    return df

df = load_data()

# Standardize the dataset
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Function for PCA and KMeans clustering
def perform_clustering(df_standardized, n_clusters):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df_standardized)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(pca_results)
    
    return pca_results, cluster_labels, kmeans.cluster_centers_

# Sidebar for user input
with st.sidebar:
    st.header("Clustering Configuration")
    num_clusters = st.slider("Select the number of clusters:", 2, 10, 7)

# Perform clustering on the entire dataset
pca_results, cluster_labels, centroids = perform_clustering(df_standardized, num_clusters)
df['Cluster'] = cluster_labels

# Plotting the clusters
st.subheader("KMeans Clustering on Entire Dataset")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, cmap='viridis')
ax.set_title("KMeans Clustering on Entire Dataset", fontsize=16)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.colorbar(scatter, label="Cluster", ax=ax)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
ax.legend()
st.pyplot(fig)

# Churn rate analysis for each cluster
st.subheader("Churn Rate Analysis per Cluster")
churn_rate_analysis = df.groupby('Cluster')['Churn'].mean().reset_index()
churn_rate_analysis.columns = ['Cluster', 'Churn Rate']
st.dataframe(churn_rate_analysis)

# Additional clusters for churn_false
churn_false_std = df[df['Churn'] == False]
pca_results_false, cluster_labels_false, centroids_false = perform_clustering(churn_false_std[numeric_cols], num_clusters)
churn_false_std['Cluster'] = cluster_labels_false

# Plotting the clusters for churn_false
st.subheader("KMeans Clustering on churn_false")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(pca_results_false[:, 0], pca_results_false[:, 1], c=cluster_labels_false, cmap='viridis')
ax.set_title("KMeans Clustering on churn_false", fontsize=16)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.colorbar(scatter, label="Cluster", ax=ax)
ax.scatter(centroids_false[:, 0], centroids_false[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
ax.legend()
st.pyplot(fig)

# Additional clusters for churn_true
churn_true_std = df[df['Churn'] == True]
pca_results_true, cluster_labels_true, centroids_true = perform_clustering(churn_true_std[numeric_cols], num_clusters)
churn_true_std['Cluster'] = cluster_labels_true

# Plotting the clusters for churn_true
st.subheader("KMeans Clustering on churn_true")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(pca_results_true[:, 0], pca_results_true[:, 1], c=cluster_labels_true, cmap='viridis')
ax.set_title("KMeans Clustering on churn_true", fontsize=16)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
plt.colorbar(scatter, label="Cluster", ax=ax)
ax.scatter(centroids_true[:, 0], centroids_true[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
ax.legend()
st.pyplot(fig)

# Display descriptive statistics for each cluster
st.subheader("Descriptive Statistics by Cluster")
grouped_df = df.groupby('Cluster')
st.dataframe(grouped_df.describe())

# Phone number selection and cluster identification
st.subheader("Phone Number Cluster Identification")
phone_number = st.selectbox("Select a Phone Number", df['Phone_Number'].unique())

if phone_number:
    customer_cluster = df[df['Phone_Number'] == phone_number]['Cluster'].values[0]
    st.write(f"The selected phone number belongs to Cluster {customer_cluster}.")
    
    # Visualization of the selected phone number's cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, cmap='viridis')
    plt.title(f'Cluster Visualization for Selected Phone Number: {phone_number}')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.colorbar(scatter, label='Cluster')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.scatter(pca_results[df['Phone_Number'] == phone_number, 0], pca_results[df['Phone_Number'] == phone_number, 1], c='black', marker='o', s=100, label='Selected Phone Number')
    plt.legend()
    st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("**Developed by [Your Name]**")
