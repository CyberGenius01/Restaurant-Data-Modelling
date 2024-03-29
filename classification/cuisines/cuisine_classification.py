""" RESTAURANT RECOMMENDATION PREDICTION MODEL """

# Importing Basic Libraries
from collections import OrderedDict
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer    

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
dataset = dataset.iloc[:,[1,2,3,4,5,7,8,9,10,11,16,17,18]]
dataset = dataset.fillna('Unknown')

corpus = []
cuisines_hist = []
for _ in range(len(dataset['Cuisines'])):
    cuisine = re.sub('[^a-zA-Z]', ' ', str(dataset['Cuisines']))
    cuisine = cuisine.lower()
    cuisine = cuisine.split()
    wnl= WordNetLemmatizer()
    unwanted = ['cuisine', 'dtype', 'cafe', 'world', 'object', 'name', 'length', 'cuisines']
    cuisine = [wnl.lemmatize(word) for word in cuisine if word not in unwanted]
    for x in cuisine:
        cuisines_hist.append(x)
    cuisine = ' '.join(cuisine)
    corpus.append(cuisine)

#DISTRIBUTION OF CUISINES OVER THE RESTAURANTS   
"""
cuisines_hist = np.asarray(cuisines_hist)
labels = np.unique(cuisines_hist, return_counts=True)[0]
values = np.unique(cuisines_hist, return_counts=True)[1]

cuisines_bar = {}
for cuisine, frequency in zip(labels, values):
    cuisines_bar[cuisine] = frequency

cuisines_bar = OrderedDict(sorted(cuisines_bar.items(), reverse=False, key=lambda lam:lam[1]))

plt.figure(figsize=(10,5))
plt.bar(x = cuisines_bar.keys(), height=cuisines_bar.values(), color='chocolate')
plt.title('Frequency of Cuisines prefered')
plt.xlabel('Cuisines')
plt.ylabel('Frequency')
plt.yticks([2000, 4000, 6000, 8000])
plt.xticks(rotation=-90, )
plt.show()
"""

# Implementation of Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cVizer = CountVectorizer()
vector = cVizer.fit_transform(corpus)

# Implementation of Similarity Transformation
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

class Restaurant:    
    def __init__(self, cuisine, dataset=dataset):
        self.cuisine = cuisine
        self.dataset = dataset

    def classify(self):
        restaurants = []
        try: 
            index = self.dataset[self.dataset['Cuisines'].str.contains(str(self.cuisine))].index[0]
            distance = sorted(list(enumerate(similarity[index])), reverse=True, key= lambda vector:vector[1])
            for i in distance[:10]:
                try:
                    restaurants.append([self.dataset.iloc[i[0]].Restaurant_Name, self.dataset.iloc[i[0]].City ,self.dataset.iloc[i[0]].Address ,self.dataset.iloc[i[0]].Locality,
                                    self.dataset.iloc[i[0]].Cuisines, self.dataset.iloc[i[0]].Currency, self.dataset.iloc[i[0]].Aggregate_rating, self.dataset.iloc[i[0]].Rating_color]) 
                except IndexError:
                    continue
            return np.array(restaurants)
        except IndexError:
            return np.array(restaurants)

if __name__ == '__main__':
    restaurants = Restaurant(cuisine='Brazilian').classify()
    print(restaurants)