import pickle
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

def lyrics_to_words(raw_lyric):
    lyric_text = BeautifulSoup(raw_lyric, "html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", lyric_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join(meaningful_words))

forest_pickle = open('song_model.pickle','rb')
forest = pickle.load(forest_pickle)
vectorizer_pickle = open('song_vectorizer.pickle','rb')
vectorizer = pickle.load(vectorizer_pickle)

# with open ("lyrics.txt", "r") as myfile:
#     lyrics = myfile.readlines()

# with open('lyrics.txt', 'r') as myfile:
#     lyrics = myfile.read().replace('\n', '')

def magic_funct(lyrics):
    print(lyrics)

    clean_test_lyrics = []
    clean_lyrics = lyrics_to_words(lyrics)
    clean_test_lyrics.append( clean_lyrics )

    test_data_features = vectorizer.transform(clean_test_lyrics)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)

    print(result)
    return result
