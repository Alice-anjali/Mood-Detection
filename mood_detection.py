import pandas as pd
import pickle
import numpy as np

word2vec_model = pickle.load(open('Word2Vec_Model', 'rb'))
train = pd.read_csv("lyrics1_500.csv")
