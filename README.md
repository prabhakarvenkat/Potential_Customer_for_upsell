# Potential_Customer_for_upsell

## Abstract
This project develops a customer clustering and churn analysis system utilizing call detail records (CDR). By leveraging Exploratory Data Analysis (EDA) and clustering techniques, it categorizes customers into distinct clusters to identify patterns related to churn behavior. The implementation, deployed through a Streamlit app, features an interactive interface where users can explore EDA results, visualize cluster distributions for both churned and non-churned customers, and examine specific customer assignments within these clusters. The integration of PCA aids in the clear visualization of clustering outcomes.
## Acknowledgements

 - [KAGGLE](https://www.kaggle.com/datasets/anshulmehtakaggl/cdrcall-details-record-predict-telco-churn)
 - [EDA](https://www.ibm.com/topics/exploratory-data-analysis)
 - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
 - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## Appendix

#### A. Code Structure and Key Modules
1. *Exploratory Data Analysis (EDA) Module*:
   - *Purpose*: Gain insights into the dataset by identifying patterns, trends, and anomalies.
   - *Key Functions*:
     - load_data(): Loads the dataset for analysis.
     - summarize_data(): Provides statistical summaries and basic data profiling.
     - visualize_distributions(): Creates distribution plots for key variables.
     - correlation_analysis(): Visualizes correlations among features to identify relationships.

2. *Clustering Module*:
   - *Purpose*: Cluster customers into distinct groups and analyze these clusters, particularly focusing on churn prediction.
   - *Key Functions*:
     - perform_clustering(): Executes clustering algorithms (e.g., K-Means) on the dataset.
     - plot_clusters(): Visualizes the resulting clusters using PCA for dimensionality reduction.
     - analyze_churn_clusters(): Separates and visualizes clusters for churn_true and churn_false subsets.

3. *Streamlit Application*:
   - *Purpose*: Provide an interactive interface for users to explore the EDA and clustering results.
   - *Key Components*:
     - *Sidebar Navigation*: Allows users to switch between EDA and clustering visualizations.
     - *Customer Selection*: Users can input a phone number to view its specific cluster assignment.
     - *Dynamic Visualizations*: Presents cluster analysis, distribution plots, and PCA projections interactively.

#### B. Data Handling and Preprocessing
1. *Data Import*:
   - The code includes flexible data loading mechanisms, allowing for the import of different datasets as needed.
   - Functions handle data cleaning, ensuring the datasets are prepared for both EDA and clustering analysis.

2. *Preprocessing Steps*:
   - Includes normalization, handling missing values, and feature selection to optimize the clustering process.
   - Special focus on handling categorical variables and ensuring data is in a suitable format for analysis.

#### C. Visualization Techniques
1. *PCA (Principal Component Analysis)*:
   - *Purpose*: Reduces the dimensionality of the data to simplify visualization of clusters.
   - *Usage*: Helps in representing multi-dimensional data in 2D or 3D plots for better interpretability of cluster structures.

2. *Cluster Visualization*:
   - Visual tools like scatter plots, heatmaps, and bar charts are employed to present cluster compositions and characteristics.
   - Separate visualizations for churn_true and churn_false help in contrasting behaviors within different customer segments.

#### D. User Interaction Features
1. *Customer Phone Number Input*:
   - Users can input a specific phone number to see which cluster it belongs to and visualize the cluster's characteristics.
   - This feature enhances the app's utility by providing personalized insights.

2. *EDA and Clustering Toggles*:
   - The app's sidebar allows users to seamlessly switch between different analyses, making it easy to navigate through the results.
   - The interface is designed to be user-friendly, with clear labels and instructions.

#### E. Analysis Methodologies
1. *Clustering*:
   - The project focuses on unsupervised learning techniques, primarily K-Means, to discover natural groupings within the data.
   - Special emphasis on understanding how these clusters relate to customer churn behavior.

2. *EDA*:
   - The exploratory phase involves understanding the data’s structure, checking for outliers, and identifying key trends that inform the clustering process.

#### F. Conclusion and Future Work
- The application successfully combines EDA and clustering to provide deep insights into customer behavior, with a particular focus on churn analysis.
- Future work could explore integrating more sophisticated clustering algorithms or expanding the dataset to include additional customer features.
## Developers

- [PRABHAKAR V](https://github.com/prabhakarvenkat)
- [DHANUSHIYA S M](https://github.com/Dhanushiyasm11)
- [JAGADEESH M](https://github.com/Jagadeeshm11)
- [KANISHKA B](https://github.com/kanishkabalakrishnan)
- [PRITHI PRASANNA P](https://github.com/PrithiPrasanna)
- [SHREERAM T](https://github.com/tshreeram)


## OUTPUT SCREENSHOTS

![OUTPUT 1:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/01.jpeg)
![OUTPUT 2:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/02.jpeg)
![OUTPUT 3:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/11.jpeg)
![OUTPUT 4:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/12.jpeg)
![OUTPUT 5:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/13.jpeg)
![OUTPUT 6:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/17.jpeg)
![OUTPUT 7:](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/18.jpeg)


## Used By

This project is used by the following company:

 ### COGNIZANT
![](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell/blob/4f1ba9a3b0e251d4a482cff767e11512e7c07e6d/assets/cognizant.jpg)

## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Submit a pull request with a detailed description of your changes

## License

This project is licensed under the [Team](https://github.com/prabhakarvenkat/Potential_Customer_for_upsell?tab=readme-ov-file#developers). See the [LICENSE](LICENSE) file for more information.

