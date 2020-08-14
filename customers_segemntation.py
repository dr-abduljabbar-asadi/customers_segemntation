##################################################################
#                      Import packages                           #
##################################################################
import pandas as pd
import numpy as np
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots


##################################################################
#                      Functions Area                            #
##################################################################

# ***************************************************
# ******            normalizer Function        ******
# ***************************************************
def normalizer(df, norm_type='StandardScaler'):
    if norm_type == 'StandardScaler':
        normalizer = StandardScaler()
    else:
        normalizer = MinMaxScaler()
    # fit and transform the data table df
    normalizer.fit(df)
    df_normalized = pd.DataFrame(normalizer.transform(df))
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized


# ***************************************************
# ******          elbow_method Function        ******
# ***************************************************
def elbow_method(df, test_clusters=10):
    distortions = []
    classes = range(1, test_clusters)
    for num_class in classes:
        kmeans_model = KMeans(n_clusters=num_class, random_state=100)
        kmeans_model.fit(df)
        distortions.append(kmeans_model.inertia_)

    plt.title('The elbow method for optimal number of classes')
    plt.xlabel('num_class')
    plt.ylabel('Distortion')
    plt.plot(classes, distortions, 'bo-')
    plt.show()

    return distortions

##################################################################
#                      Main Area                                 #
##################################################################
# ***************************************************
# ******       Instant Variables Setting       ******
# ***************************************************
data_path = 'data/'

# ***************************************************
# ******            Load Data                  ******
# ***************************************************
# -------------------------------
# 1. Load data
# https://archive.ics.uci.edu/ml/datasets/online+retail#
data_table = pd.read_excel(data_path + 'Online Retail.xlsx')

# -------------------------------
# 2. Sample of the data table
data_table.head()

# -------------------------------
# 3. Summary of the numeric data_table's columns
data_table.describe()

# ***************************************************
# ******  Data Cleaning & Feature Engineering  ******
# ***************************************************
# -------------------------------
# 1. Convert type
data_table["InvoiceDate"] = data_table["InvoiceDate"].dt.date

# -------------------------------
# 2. Create new feature: total_price
data_table["total_price"] = data_table["Quantity"] * data_table["UnitPrice"]

# -------------------------------
# 3. Create date variable that records recency
snapshot_date = max(data_table.InvoiceDate) + datetime.timedelta(days=1)
# 3.A. Aggregate data by each customer
customers_data = data_table.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'total_price': 'sum'})
# 3.B. Rename columns
customers_data.rename(columns={'InvoiceDate': 'day_to_invoice',
                               'InvoiceNo': 'count_products',
                               'total_price': 'monetary_value'}, inplace=True)

# ***************************************************
# ******  Data Preparation For Modeling        ******
# ***************************************************
# -------------------------------
# 1. Skewness check
customers_data_cleaned = pd.DataFrame()
customers_data_cleaned["day_to_invoice"] = stats.boxcox(customers_data['day_to_invoice'])[0]
customers_data_cleaned["count_products"] = stats.boxcox(customers_data['count_products'])[0]
customers_data_cleaned["monetary_value"] = pd.Series(np.cbrt(customers_data['monetary_value'])).values
customers_data_cleaned.head()

# -------------------------------
# 2. Normalization
customers_data_norm = normalizer(df=customers_data_cleaned, norm_type='StandardScaler')

# ***************************************************
# ******            Modelling                  ******
# ***************************************************
# -------------------------------
# 1. Extract optimize number of clusters based on the Elbow Method
distortions = elbow_method(df=customers_data_norm, test_clusters=40)

# -------------------------------
# 2. Apply clustering model
clustering_model = KMeans(n_clusters=3, random_state=42)
clustering_model.fit(customers_data_norm)

customers_data_norm["cluster"] = clustering_model.labels_
customers_data_norm["cluster_str"] = "cluster" + customers_data_norm["cluster"].astype(str)

customers_data["cluster"] = clustering_model.labels_
customers_data["cluster_str"] = "cluster" + customers_data["cluster"].astype(str)

# ***************************************************
# ******    Post-processing & Visualization    ******
# ***************************************************
# -------------------------------
# 1. Count per class/ rate of class
cluster_count = customers_data_norm.cluster.value_counts()
cluster_count = cluster_count.sort_index()

cluster_rate = cluster_count.sort_index() / customers_data_norm.shape[0]
cluster_rate = cluster_rate.sort_index()

fig_bar_plot = make_subplots(rows=1, cols=2)
fig_bar_plot.add_trace(
    go.Bar(name='clusters_count', x=cluster_count.index, y=cluster_count),
    row=1, col=1
)
fig_bar_plot.add_trace(
    go.Bar(name='cluster_rate', x=cluster_rate.index, y=cluster_rate),
    row=1, col=2
)
plot(fig_bar_plot)

# -------------------------------
# 2. Scatter plot of clusters and features, 2D and 3D
# 2.A. 2D scatter plot
fig_2D = go.Figure(data=go.Scatter(
    x=customers_data_norm.day_to_invoice,
    y=customers_data_norm.monetary_value,
    mode='markers',
    marker=dict(color=customers_data_norm.cluster, opacity=0.7, size=10)))
plot(fig_2D)

# 2.B. 3D scatter plot
fig_3D = go.Figure(data=[go.Scatter3d(
    x=customers_data_norm.count_products,
    y=customers_data_norm.day_to_invoice,
    z=customers_data_norm.monetary_value,
    mode='markers',
    marker=dict(
        size=12,
        color=customers_data_norm.cluster,
        colorscale='Jet',
        opacity=0.6
    )
)])
fig_3D.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                     scene=dict(xaxis_title="norm count_products",
                                yaxis_title="norm day_to_invoice",
                                zaxis_title="norm monetary_value"))
plot(fig_3D)

# -------------------------------
# 1. Calculating average of each feature of the classes in normalized data and original data
# 1.A. Calculation average from normalized data
features_avg_per_class_norm = customers_data_norm.groupby('cluster').agg({
    'day_to_invoice': 'mean',
    'count_products': 'mean',
    'monetary_value': 'mean'}).round(2)
# 1.B. Calculation average from original data
features_avg_per_class_orig = customers_data.groupby('cluster').agg({
    'day_to_invoice': 'mean',
    'count_products': 'mean',
    'monetary_value': 'mean'}).round(2)

# 1.C. Plot features_avg_per_class_norm
fig_bar_norm = go.Figure(data=[
    go.Bar(name='day_to_invoice', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.day_to_invoice),
    go.Bar(name='count_products', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.count_products),
    go.Bar(name='monetary_value', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.monetary_value)
])
fig_bar_norm.update_layout(barmode='group')
plot(fig_bar_norm)

# 1.D. Plot features_avg_per_class_orig
fig_bar_orig = go.Figure(data=[
    go.Bar(name='day_to_invoice', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.day_to_invoice),
    go.Bar(name='count_products', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.count_products),
    go.Bar(name='monetary_value', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.monetary_value)
])
fig_bar_orig.update_layout(barmode='group')
plot(fig_bar_orig)

# -------------------------------
# 2. Calculating average of each feature of the classes in normalized data and original data
# 2.A. Melting data
customers_data_norm_melt = pd.melt(customers_data_norm.reset_index(),
                                   id_vars=['index', 'cluster'],
                                   value_vars=['day_to_invoice', 'count_products', 'monetary_value'],
                                   var_name='feature',
                                   value_name='value')
# 2.B. Visualize melted data
sns.lineplot('feature', 'value', hue='cluster', data=customers_data_norm_melt)


