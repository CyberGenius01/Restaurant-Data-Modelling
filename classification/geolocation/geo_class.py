""" RESTAURANT GEOGRAPHY CLASSIFICATION MODEL """

# Importing Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
dataset = dataset.fillna('Unknown')
X = dataset.iloc[:,[7,8]].values

# Handling the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
numeric_cols = [0,1]
X[:,numeric_cols] = imputer.fit_transform(X[:,numeric_cols])

# Using the K-means to find the optimal number of clusters
"""
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()
"""

# Using the dendrogram to find the optimal number of clusters
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Location')
plt.ylabel('Euclidean Distance')
plt.show()
"""


# Trainnig the model
from sklearn.cluster import KMeans
hc = KMeans(n_clusters = 2)
y_hc = hc.fit_predict(X)

# Visualization of result
def classes(lat, long, class_of_param):
    color = ['red', 'limegreen', 'cyan', 'yellow']
    plt.figure(figsize=(8,5))
    plt.style.use('ggplot')
    ax = plt.subplot()
    for i in range(2):
        plt.scatter(X[y_hc == i,0], X[y_hc == i,1], s = 20, color = color[i], label = f'class-{i+1}')
    ax.scatter(lat, long, s=25, color='black', label=class_of_param)
    ax.annotate('({}, {})'.format(lat, long) ,xy=(lat, long))
    plt.title('Lattitude-Longitude Restaurant Classification')
    plt.xlabel('Longitude')
    plt.ylabel('Lattitude')
    plt.legend()
    plt.show()

def geo_classification(lat,long):
    param = np.array([[lat,long]])
    pred = hc.predict(param)
    if pred[0] == 0:     
        class_of_param = 'class-1'
    else:    
        class_of_param = 'class-2'
    print(class_of_param)
    classes(lat, long, class_of_param)

if __name__ == '__main__':
   geo_classification(36.45,20.15)
