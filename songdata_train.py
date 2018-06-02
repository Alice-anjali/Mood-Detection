import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle


def traindata():
    # train = pd.read_csv("songs_train_set_data.csv")
    train = pd.read_csv("lyrics1_500.csv")


    def lyrics_to_words(raw_lyric):
        lyric_text = BeautifulSoup(raw_lyric, "html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", lyric_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return(" ".join(meaningful_words))

    # print("Cleaning and parsing the training set song dataset...\n")
    num_lyrics = train["Lyrics"].size
    clean_train_lyrics = []
    for i in range(0,num_lyrics):
        if((i+1)%10 == 0):
            print("Lyric %d of %d\n" % ( i+1, num_lyrics ))
        clean_train_lyrics.append(lyrics_to_words(train["Lyrics"][i]))


    # print("Creating the bag of words...\n")
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = 5000)


    train_data_features = vectorizer.fit_transform(clean_train_lyrics)
    train_data_features = train_data_features.toarray()

    # print("Training the random forest...")
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["Mood"] )


    pickle_model = open('song_model.pickle','wb')
    pickle.dump(forest,pickle_model)
    pickle_model.close()

    vectorizer_file = open('song_vectorizer.pickle','wb')
    pickle.dump(vectorizer,vectorizer_file)
    vectorizer_file.close()

    # print("Data trained")
traindata()
