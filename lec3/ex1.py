import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('Country-data.csv')

data.info()



col = list(data.columns)
col.remove('country')
categorical_features = ['country']
numerical_features = [*col]   #creates a list called numerical_features and assigns it 
# the remaining column names from the col list. The * operator is used to unpack the elements 
# in the col list into the numerical_features list. 

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


# Add what is needed (y, by and the title)

groups = {
    'Health': ['child_mort', 'health', 'life_expec', 'total_fer'],
    'Trade': ['imports', 'exports'],
    'Finance': ['income', 'inflation', 'gdpp'],
}


compressed_data = pd.DataFrame()
compressed_data['country'] = data['country']
compressed_data['Health'] = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(data[groups['Health']]))[:, 0]
compressed_data['Trade'] = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(data[groups['Trade']]))[:, 0]
compressed_data['Finance'] = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(data[groups['Finance']]))[:, 0]
compressed_data.head()


compressed_data.to_pickle('compressed_data.pkl')


