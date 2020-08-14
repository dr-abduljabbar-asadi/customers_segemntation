##################################################################
#                      Import packages                           #
##################################################################
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go

##################################################################
#                      Functions Area                            #
##################################################################
# <editor-fold desc="Functions Area">
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
    df_normalized=pd.DataFrame( normalizer.transform(df))
    df_normalized.columns=df.columns
    df_normalized.index=df.index
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


# </editor-fold>

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
# <editor-fold desc="Load Data">
# -------------------------------
# 1. Load  Data
# https://archive.ics.uci.edu/ml/datasets/online+retail#
data_table = pd.read_excel(data_path + 'Online Retail.xlsx')

# -------------------------------
# 2. Sample of the data table
data_table.head()

# -------------------------------
# 3. Summary of the numeric data_table's columns
data_table.describe()

# </editor-fold>

# ***************************************************
# ******  Data Cleaning & Feature Engineering  ******
# ***************************************************
# <editor-fold desc="Data Cleaning & Feature Engineering">
# -------------------------------
# 1. Convert Type
data_table.dtypes
data_table["InvoiceDate"] = data_table["InvoiceDate"].dt.date

# -------------------------------
# 2. Create new feature: total_price
data_table["total_price"] = data_table["Quantity"] * data_table["UnitPrice"]

# -------------------------------
# 3. Create date variable that records recency
snapshot_date = max(data_table.InvoiceDate) + datetime.timedelta(days=1)

# Aggregate data by each customer
customers_data = data_table.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'total_price': 'sum'})

# Rename columns
customers_data.rename(columns={'InvoiceDate': 'day_to_invoice',
                               'InvoiceNo': 'count_products',
                               'total_price': 'monetary_value'}, inplace=True)
# ***************************************************
# ******            Data Visualization         ******
# ***************************************************


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
# ******            Data Visualization         ******
# ***************************************************

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

# ***************************************************
# ******    Post-processing & Visualization    ******
# ***************************************************
# 1. Calculating average of each feature of the classes
features_avg_per_class=customers_data_norm.groupby('cluster').agg({
    'day_to_invoice': 'mean',
    'count_products': 'mean',
    'monetary_value': 'mean'}).round(2)

fig = go.Figure(data=[
    go.Bar(name='day_to_invoice', x=features_avg_per_class.index, y=features_avg_per_class.day_to_invoice),
    go.Bar(name='count_products', x=features_avg_per_class.index, y=features_avg_per_class.count_products),
    go.Bar(name='monetary_value', x=features_avg_per_class.index, y=features_avg_per_class.monetary_value)
])
fig.update_layout(barmode='group')
plot(fig)


features_avg_per_class=customers_data.groupby('cluster').agg({
    'day_to_invoice': 'mean',
    'count_products': 'mean',
    'monetary_value': ['mean', 'count']}).round(2)



# Create the dataframe
df_normalized = pd.DataFrame(customers_data_norm, columns=['day_to_invoice', 'count_products', 'monetary_value'])
df_normalized['customer_id'] = customers_data.index
df_normalized['cluster'] = clustering_model.labels_
# Melt The Data
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['customer_id', 'cluster'],
                      value_vars=['day_to_invoice', 'count_products', 'monetary_value'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()
# Visualize it
sns.lineplot('Attribute', 'Value', hue='cluster', data=df_nor_melt)

# </editor-fold>
