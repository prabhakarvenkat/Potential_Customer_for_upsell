import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\ACER\Documents\cognizant\streamlit\cleaned_churn.csv"
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

# Streamlit app with background styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00008B;  /* Changed to dark blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('📊 Customer Segmentation Analysis')

# Display the Elbow Method plot with a dark blue title
st.subheader('Elbow Method for Optimal Clusters')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title('Elbow Method for Optimal Clusters', color='#00008B')  # Changed to dark blue
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Choose the optimal number of clusters with slider
st.subheader('Select Number of Clusters')
optimal_clusters = st.slider('How many clusters?', min_value=1, max_value=3, value=3)
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Calculate Customer Lifetime Value (CLV)
df['CLV'] = df['Day_Charge'] + df['Eve_Charge'] + df['Night_Charge']

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

st.subheader('Cluster Visualization using PCA')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster_Name', data=df, 
    palette=cluster_colors, s=100, ax=ax
)
ax.set_title('Cluster Visualization using PCA', color='#00008B')  # Changed to dark blue
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.legend(title='Cluster')
st.pyplot(fig)

# Display cluster statistics and CLV analysis with collapsible sections
st.subheader('Cluster Analysis')
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    with st.expander(f"📈 {cluster_name} Statistics"):
        st.write(df[df['Cluster'] == i][numerical_cols].describe())

st.subheader('Cluster-wise CLV Analysis')
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    cluster_clv_value = df[df['Cluster'] == i]['CLV'].mean()
    st.write(f"💰 Average CLV for {cluster_name}: {cluster_clv_value:.2f}")

# Churn rate analysis for each cluster
st.subheader('Churn Rate Analysis by Cluster')
for i in range(optimal_clusters):
    cluster_name = cluster_names.get(i, f'Cluster {i}')
    churn_rate = df[df['Cluster'] == i]['Churn'].mean() * 100  # Convert to percentage
    st.write(f"⚠ Churn Rate for {cluster_name}: {churn_rate:.2f}%")

# Count the number of records in each cluster
st.subheader('Number of Records in Each Cluster')
cluster_counts = df['Cluster_Name'].value_counts()
st.write(cluster_counts)

# GUI to find the cluster of a particular customer by phone number
st.subheader('Find Customer Cluster by Phone Number')

# Choose a customer phone number from a dropdown menu
customer_phone_number = st.selectbox('Select Customer Phone Number:', df['Phone_Number'].unique())

# Find the cluster of the selected customer
customer_cluster = df[df['Phone_Number'] == customer_phone_number]['Cluster'].values[0]
customer_cluster_name = df[df['Phone_Number'] == customer_phone_number]['Cluster_Name'].values[0]

st.write(f"📞 Customer with phone number *{customer_phone_number}* belongs to *{customer_cluster_name}* (Cluster {customer_cluster}).")

# Display detailed cluster information
st.write(f"### {customer_cluster_name} Details")
st.write(df[df['Cluster'] == customer_cluster][numerical_cols].describe())

# Save the updated dataframe with clusters and CLV
csv = df.to_csv(index=False)
st.download_button(
    label="💾 Download Clustered Data",
    data=csv,
    file_name='clustered_data.csv',
    mime='text/csv'
)