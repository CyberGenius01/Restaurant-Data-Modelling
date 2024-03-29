""" RESTAURANT RECOMMENDATION PREDICTION MODEL """

# Importing Basic Libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
dataset = dataset.iloc[:,[1,2,3,4,5,7,8,9,10,11,16,17,18]]
dataset = dataset.fillna('Unknown')

# Modifying rating color parameter
import re
for i in range(9551):
    dataset.iloc[i,-1] = re.sub(' ', '', dataset.iloc[i,-1])
    dataset.iloc[i,-1] = dataset.iloc[i,-1].lower()

# Implementation of Natural Language Processing
from nltk.stem.porter import PorterStemmer    

corpus = []
for i in range(9551):
    cuisine = re.sub('[^a-zA-Z]', ' ', str(dataset.iloc[i,7]))
    cuisine = cuisine.lower()
    cuisine = cuisine.split()
    ps = PorterStemmer()
    cuisine = [ps.stem(word) for word in cuisine]
    cuisine = ' '.join(cuisine)
    corpus.append(cuisine)

# Implementation of Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cVizer = CountVectorizer(max_features=150)
vector = cVizer.fit_transform(corpus)

# Implementation of Similarity Transformation
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

class FilterRestaurant:    
    def __init__(self, cuisine, price=None, rating=None, dataset=dataset):
        self.cuisine = cuisine
        self.rating = rating
        self.price = price
        if self.price != None and self.rating == None:
            self.dataset = dataset[dataset['Price_range'] <= self.price]
        elif self.price == None and self.rating != None:
            self.dataset = dataset[dataset['Aggregate_rating'] >= self.rating]
        elif self.price != None and self.rating != None:
            self.dataset = dataset[dataset['Price_range'] <= self.price]
            self.dataset = self.dataset[self.dataset['Aggregate_rating'] >= self.rating]
        else:
            self.dataset = dataset

    def recommend(self):
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
    restaurants = FilterRestaurant(cuisine='Brazilian').recommend()
    print(restaurants)