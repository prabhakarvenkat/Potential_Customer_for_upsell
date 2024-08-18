import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app setup
st.set_page_config(page_title="Customer Analysis", layout="wide")
st.title("Customer Analysis")

# Manually specify file paths for the datasets
path_eda = r"C:\Users\manid\OneDrive\Desktop\PRODUCT UPSELL\DATASET CDR\CDR-Call-Details.csv"  # Replace with your EDA dataset path
path_clustering = r"C:\Users\manid\OneDrive\Desktop\cleaned_churn_cdr_ds.csv" # Replace with your clustering dataset path

# Load datasets
def load_data(file_path):
    return pd.read_csv(file_path)

df_eda = load_data(path_eda)
df_clustering = load_data(path_clustering)

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    option = st.radio("Select a section:", ["Exploratory Data Analysis (EDA)", "Clustering"])

# EDA Section
if option == "Exploratory Data Analysis (EDA)" and df_eda is not None:
    st.title('Exploratory Data Analysis (EDA)')

    # Dataset Information
    st.write('*Dataset Shape:*', df_eda.shape)
    st.write('There are', df_eda.shape[0], 'rows and', df_eda.shape[1], 'columns/features.')

    # Display dataset columns
    st.write('*Columns in the Dataset:*')
    st.write(df_eda.columns)

    # Replace spaces with underscores in column names
    df_eda.columns = df_eda.columns.str.replace(' ', '_')
    st.write('*Columns After Replacing Spaces with Underscores:*')
    st.write(df_eda.columns)

    # Display the first few rows of the dataset
    st.write('*First Few Rows of the Dataset:*')
    st.write(df_eda.head())

    # Assumptions
    st.write('*Assumptions:*')
    st.write('- Assuming Account_Length is measured in days.')
    st.write('- Assuming charges are measured in USD ($) because there is no dataset description provided.')

    # Display dataset information
    st.write('*Dataset Information:*')
    st.write(df_eda.info())

    st.write('There are 15 numeric columns - 8 float and 7 int, 1 boolean, and 1 object column.')

    # Descriptive statistics
    st.write('*Descriptive Statistics:*')
    st.write(df_eda.describe())

    st.write('- Account_Length has a maximum value of 21111, which is highly unlikely (57 years).')
    st.write('- Similar to Account_Length, other features also have questionable maximum values indicating the presence of outliers skewing the dataset.')
    st.write('- Every column has a mean << 75th percentile, indicating a right-skewed distribution.')
    st.write('- High standard deviation values suggest a wide range of values.')
    st.write('- VMail_Message, CustServ, Intl_Mins, Intl_Calls, and Intl_Charge aside, the rest of the columns need outlier detection and removal due to impossible values.')

    # Numeric columns for analysis
    numeric_cols = ['Account_Length', 'Day_Mins', 'Day_Calls', 'Day_Charge', 'Eve_Mins', 'Eve_Calls', 'Eve_Charge', 
                    'Night_Mins', 'Night_Calls', 'Night_Charge', 'Intl_Mins', 'Intl_Calls', 'Intl_Charge', 'VMail_Message']

    # Histograms
    st.write('*Histograms:*')
    fig, ax = plt.subplots(figsize=(12, 12))
    df_eda[numeric_cols].hist(ax=ax)
    st.pyplot(fig)
    st.write('The histogram clearly indicates skewness.')

    # Check for missing values
    st.write('*Missing Values:*')
    st.write(df_eda.isna().sum())

    # Check for duplicates
    st.write('*Duplicate Rows:*')
    st.write('Total Duplicate Rows:', df_eda.duplicated().sum())
    st.write('- Around 40k rows are duplicates.')
    st.write('- The df.duplicated() function compares all the feature values for an exact match, looking for a 1-1 copy.')

    # Remove duplicates
    df_eda = df_eda.loc[~df_eda.duplicated()].reset_index(drop=True).copy()
    st.write('*Dataset After Removing Duplicates:*')
    st.write(df_eda.shape)

    # Unique counts in features
    st.write('*Unique Counts in Each Feature:*')
    st.write(df_eda.nunique())
    st.write('Checking for unusual unique counts in hopes of finding anything unusual.')

    # Outlier detection and removal
    st.write('*Outlier Detection and Removal:*')
    def remove_outliers_iqr(df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    outlier_cols = ['Account_Length', 'Day_Mins', 'Day_Calls', 'Eve_Mins', 'Eve_Calls', 'Night_Mins', 'Night_Calls']
    for col in outlier_cols:
        df_eda = remove_outliers_iqr(df_eda, col)

    df_eda.reset_index(drop=True, inplace=True)
    st.write('*Dataset After Removing Outliers:*')
    st.write(df_eda.shape)

    # Histograms after outlier removal
    st.write('*Histograms After Outlier Removal:*')
    fig, ax = plt.subplots(figsize=(12, 12))
    df_eda[numeric_cols].hist(ax=ax)
    st.pyplot(fig)
    st.write('After addressing outliers, the histogram looks less skewed.')

    # Box plot after outlier removal
    st.write('*Box Plot After Outlier Removal:*')
    fig, ax = plt.subplots(figsize=(12, 8))
    df_eda.boxplot(column=numeric_cols, ax=ax)
    ax.set_title('Distribution of Call Minutes by Type', fontsize=16)
    ax.set_xlabel('Call Type', fontsize=14)
    ax.set_ylabel('Freq', fontsize=14)
    ax.set_xticklabels(numeric_cols, rotation=45)
    st.pyplot(fig)
    st.write('By visualizing the box plot, we can clearly identify outliers.')

    # Correlation heatmap
    st.write('*Correlation Heatmap:*')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_eda[numeric_cols].corr(), annot=True, fmt='.2f', cmap='magma')
    st.pyplot(plt)
    st.write('Shows somewhat linear relationships indicating that as minutes increase, so does the price (charge) across day, eve, night & intl.')

    # Scatter plots for selected feature pairs
    feature_pairs = [
        ('Day_Mins', 'Day_Charge'),
        ('Eve_Mins', 'Eve_Charge'),
        ('Night_Mins', 'Night_Charge'),
        ('Intl_Mins', 'Intl_Charge'),
    ]
    for x, y in feature_pairs:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_eda[x], df_eda[y], alpha=0.5)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Scatter Plot of {x} vs {y}')
        st.pyplot(plt)

    # VMail_Message histogram
    st.write('*Histogram for VMail_Message Feature:*')
    plt.hist(df_eda['VMail_Message'], bins=20)
    st.pyplot(plt)
    st.write('*Inference:*')
    st.write('Just checking the VMail_Message feature for anything unusual.')
    st.write(df_eda.describe())

# Clustering Section
elif option == "Clustering" and df_clustering is not None:
    st.title('Customer Clustering Analysis')

    # Standardize the dataset
    numeric_cols_clustering = df_clustering.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_clustering[numeric_cols_clustering]), columns=numeric_cols_clustering)

    # Function for PCA and KMeans clustering
    def perform_clustering(df_standardized, n_clusters):
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(df_standardized)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(pca_results)
        
        return pca_results, cluster_labels, kmeans.cluster_centers_

    # Sidebar for clustering configuration
    with st.sidebar:
        st.header("Clustering Configuration")
        num_clusters = st.slider("Select the number of clusters:", 2, 10, 7)

    # Perform clustering on the entire dataset
    pca_results, cluster_labels, centroids = perform_clustering(df_standardized, num_clusters)
    df_clustering['Cluster'] = cluster_labels

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
    churn_rate_analysis = df_clustering.groupby('Cluster')['Churn'].mean().reset_index()
    churn_rate_analysis.columns = ['Cluster', 'Churn Rate']
    st.dataframe(churn_rate_analysis)

    # Additional clusters for churn_false
    churn_false_std = df_clustering[df_clustering['Churn'] == False]
    pca_results_false, cluster_labels_false, centroids_false = perform_clustering(churn_false_std[numeric_cols_clustering], num_clusters)
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
    churn_true_std = df_clustering[df_clustering['Churn'] == True]
    pca_results_true, cluster_labels_true, centroids_true = perform_clustering(churn_true_std[numeric_cols_clustering], num_clusters)
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
    grouped_df = df_clustering.groupby('Cluster')
    st.dataframe(grouped_df.describe())

    # Phone number selection and cluster identification
    st.subheader("Phone Number Cluster Identification")
    phone_number = st.selectbox("Select a Phone Number", df_clustering['Phone_Number'].unique())

    if phone_number:
        customer_cluster = df_clustering[df_clustering['Phone_Number'] == phone_number]['Cluster'].values[0]
        st.write(f"The selected phone number belongs to Cluster {customer_cluster}.")
        
        # Visualization of the selected phone number's cluster
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, cmap='viridis')
        plt.title(f'Cluster Visualization for Selected Phone Number: {phone_number}')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.colorbar(scatter, label='Cluster')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.scatter(pca_results[df_clustering['Phone_Number'] == phone_number, 0], pca_results[df_clustering['Phone_Number'] == phone_number, 1], c='black', marker='o', s=100, label='Selected Phone Number')
        plt.legend()
        st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Developed by COGNIZANT BATCH 4</h3>", unsafe_allow_html=True)
