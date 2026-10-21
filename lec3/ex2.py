import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,StandardScaler


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import plotly.express as px
import kaleido


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


from scipy.cluster.hierarchy import dendrogram, linkage



from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('Country-data.csv')
compressed_data = pd.read_pickle('compressed_data.pkl')


print(data.head())
print(compressed_data.head())



col = list(data.columns)
col.remove('country')
categorical_features = ['country']
numerical_features = [*col]   #creates a list called numerical_features and assigns it 
# the remaining column names from the col list. The * operator is used to unpack the elements 
# in the col list into the numerical_features list. 

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

compressed_data['Health'] = mms.fit_transform(compressed_data[['Health']])
compressed_data['Trade'] = mms.fit_transform(compressed_data[['Trade']])
compressed_data['Finance'] = mms.fit_transform(compressed_data[['Finance']])
compressed_data.insert(loc = 0, value = list(data['country']), column = 'Country')
print(compressed_data.head())








df_no_country = compressed_data.select_dtypes(include='number')

sse = {}
sil = []
kmax = 10
fig = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))

# Elbow Method :
plt.subplot(1,2,1)
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_no_country)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
sns.lineplot(x = list(sse.keys()), y = list(sse.values()))
plt.title('Elbow Method')
plt.xlabel("k : Number of cluster")
plt.ylabel("Sum of Squared Error")
plt.grid()

# Silhouette Score Method
plt.subplot(1,2,2)
for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters = k).fit(df_no_country)
    labels = kmeans.labels_
    sil.append(silhouette_score(df_no_country, labels, metric = 'euclidean'))
sns.lineplot(x = range(2,kmax + 1), y = sil)
plt.title('Silhouette Score Method')
plt.xlabel("k : Number of cluster")
plt.ylabel("Silhouette Score")
plt.grid()

plt.show()



model = KMeans(n_clusters = 4,max_iter = 1000)
model.fit(df_no_country)
cluster = model.cluster_centers_
centroids = np.array(cluster)
labels = model.labels_


compressed_data['Class'] = labels; data['Class'] = labels



fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

##   WHICH FEATURE WE COULD CHECK TO SEE THE CLASSES? Add a feature in the y

plt.subplot(1,2,1)
sns.boxplot(x = 'Class', y = 'child_mort', data  = data, color = '#FF781F')
plt.title('Child Mort vs Class')

plt.subplot(1,2,2)
sns.boxplot(x = 'Class', y = 'income', data  = data, color = '#FF781F')
plt.title('Income vs Class')

plt.show()





compressed_data['Class'] = compressed_data['Class'].astype(object)
compressed_data.loc[compressed_data['Class'] == 3, 'Class'] = 'NO HELP'
compressed_data.loc[compressed_data['Class'] == 0, 'Class'] = 'Might Need Help'
compressed_data.loc[compressed_data['Class'] == 1, 'Class'] = 'Help Needed'
compressed_data.loc[compressed_data['Class'] == 2, 'Class'] ='Probably no help' 

fig = px.choropleth(compressed_data[['Country','Class']],
                    locationmode = 'country names',
                    locations = 'Country',
                    title = 'Needed Help Per Country (World)',
                    color = compressed_data['Class'],  
                    color_discrete_map={'NO HELP' : 'Black',
                                        'Might Need Help':'Yellow',
                                        'Help Needed':'Red',
                                        'No Help Needed':'Green'})
fig.update_geos(fitbounds = "locations", visible = True)
fig.update_layout(legend_title_text = 'Labels',legend_title_side = 'top',title_pad_l = 260,title_y = 0.86)
fig.show(engine = 'kaleido')






knn = NearestNeighbors(n_neighbors = 7)
model = knn.fit(df_no_country)
distances, indices = knn.kneighbors(df_no_country)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.xlabel('Points Sorted by Distance')
plt.ylabel('7-NN Distance')
plt.title('K-Distance Graph')
plt.show()




db = DBSCAN(eps = 0.080, min_samples = 8).fit(df_no_country)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Number of Clusters : ', n_clusters_)
print('Number of Outliers : ', n_noise_)

data['Class'] = labels; compressed_data['Class'] = labels




# Plot the map with the classes
compressed_data['Class'] = compressed_data['Class'].astype(object)
compressed_data.loc[compressed_data['Class'] == -1, 'Class'] = 'Noise / Outliers'
compressed_data.loc[compressed_data['Class'] == 0, 'Class'] = 'Might Need Help'
compressed_data.loc[compressed_data['Class'] == 1, 'Class'] = 'No Help Needed'
compressed_data.loc[compressed_data['Class'] == 2, 'Class'] ='Help Needed'

fig = px.choropleth(compressed_data[['Country','Class']],
                    locationmode = 'country names',
                    locations = 'Country',
                    title = 'Needed Help Per Country (World)',
                    color = compressed_data['Class'],  
                    color_discrete_map={'Noise / Outliers' : 'Black',
                                        'Help Needed':'Red',
                                        'Might Need Help':'Yellow',
                                        'No Help Needed':'Green'})
fig.update_geos(fitbounds = "locations", visible = True)
fig.update_layout(legend_title_text = 'Labels',legend_title_side = 'top',title_pad_l = 260,title_y = 0.86)
fig.show(engine = 'kaleido')






linkage_data = linkage(df_no_country, method = 'ward', metric = 'euclidean')
dendrogram(linkage_data)
plt.grid()
plt.tight_layout()
plt.show()





hierarchical_cluster = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')
labels = hierarchical_cluster.fit(df_no_country)

pred_agc = pd.Series(hierarchical_cluster.labels_)
data['Class'] = pred_agc; compressed_data['Class'] = pred_agc


#Plot the distribution of the classes for the defined features (decided previously)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

plt.subplot(1,2,1)
sns.boxplot(x = 'Class', y = 'child_mort', data  = data, color = '#FF781F')
plt.title('child_mort vs Class')

plt.subplot(1,2,2)
sns.boxplot(x = 'Class', y = 'income', data  = data, color = '#FF781F')
plt.title('income vs Class')

plt.show()


compressed_data['Class'] = compressed_data['Class'].astype(object)
compressed_data.loc[compressed_data['Class'] == 0, 'Class'] = 'No Help Needed'
compressed_data.loc[compressed_data['Class'] == 1, 'Class'] = 'Help Needed'
compressed_data.loc[compressed_data['Class'] == 2, 'Class'] = 'Might Need Help'

fig = px.choropleth(compressed_data[['Country','Class']],
                    locationmode = 'country names',
                    locations = 'Country',
                    title = 'Needed Help Per Country (World)',
                    color = compressed_data['Class'],  
                    color_discrete_map = {'Help Needed' : 'Red',
                                          'Might Need Help' : 'Yellow',
                                          'No Help Needed' : 'Green'})
fig.update_geos(fitbounds = "locations", visible = True)
fig.update_layout(legend_title_text = 'Labels',legend_title_side = 'top',title_pad_l = 260,title_y = 0.86)
fig.show(engine = 'kaleido')