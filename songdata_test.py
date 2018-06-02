import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import pickle


def change_to_index(str):
    if str == 'happy':
        return 0
    if str == 'sad':
        return 1
    if str == 'angry':
        return 2
    if str == 'relaxed':
        return 3
def testdata():
    test = pd.read_csv("songs_test_set_data.csv")
    test1 = pd.read_csv("songs_train_set_data.csv") #trainset


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


    num_lyrics = len(test["Lyrics"])
    clean_test_lyrics = []
    # print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_lyrics):
        # if( (i+1) % 10 == 0 ):
            # print("Lyric %d of %d\n" % (i+1, num_lyrics))
        clean_lyrics = lyrics_to_words( test["Lyrics"][i] )
        clean_test_lyrics.append( clean_lyrics )

    #start trainset
    num_lyrics1 = len(test1["Lyrics"]) #trainset
    clean_test_lyrics1 = []
    # print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_lyrics1):
        # if( (i+1) % 10 == 0 ):
            # print("Lyric %d of %d\n" % (i+1, num_lyrics))
        clean_lyrics = lyrics_to_words( test1["Lyrics"][i] )
        clean_test_lyrics1.append( clean_lyrics )

    #end trainset test

    # print(len(clean_test_lyrics))
    test_data_features = vectorizer.transform(clean_test_lyrics)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame( data={"Title":test["Title"], "Actual_sentiment":test["Mood"], "Predicted_sentiment":result} )

    output.to_csv( "songdata_BOW_model.csv")

    compare_data = pd.read_csv("songdata_BOW_model.csv")

    num_data = compare_data["Title"].size
    count = 0
    confusion_matrix = [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]]
    print(num_data)
    for i in range(0,num_data):
        confusion_matrix[change_to_index(compare_data["Actual_sentiment"][i])][change_to_index(compare_data["Predicted_sentiment"][i])] += 1
        if compare_data["Actual_sentiment"][i] == compare_data["Predicted_sentiment"][i]:
            count += 1

    # print("Count = "+str(count))
    accuracy = (count/num_data)*100
    print("TEST accuracy :- "+str(accuracy))
    print(confusion_matrix)
    # print("Accuracy = "+str(accuracy)+"%")
    # return accuracy
    # print("Data Tested")


    #start trainset
    test_data_features1 = vectorizer.transform(clean_test_lyrics1)
    test_data_features1 = test_data_features1.toarray()
    result1 = forest.predict(test_data_features1)
    output1 = pd.DataFrame( data={"Title":test1["Title"], "Actual_sentiment":test1["Mood"], "Predicted_sentiment":result1} )

    output1.to_csv( "songdata_BOW_model1.csv")

    compare_data1 = pd.read_csv("songdata_BOW_model1.csv")

    num_data1 = compare_data1["Title"].size
    count1 = 0


    for i in range(0,num_data1):
        if compare_data1["Actual_sentiment"][i] == compare_data1["Predicted_sentiment"][i]:
            count1 += 1
    # print("Count = "+str(count))
    accuracy1 = (count1/num_data1)*100
    print("TRAIN accuracy :- "+str(accuracy1))
    print(confusion_matrix)
    # print("Accuracy = "+str(accuracy)+"%")
    return accuracy-accuracy1

    #end trainset
testdata()
