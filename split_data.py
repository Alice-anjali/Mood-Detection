import pandas as pd
from sklearn.model_selection import train_test_split

# songs = pd.read_csv("songdata.csv",index_col=False)
# train_data, test_data = train_test_split(train, test_size=0.5)
# train_data.to_csv( "training_set_data.csv", index=False, sep='\t', quoting=3 )
# test_data.to_csv( "testing_set_data.csv", index=False, sep='\t', quoting=3 )
# print(songs.size)
# songs1, songs2 = train_test_split(songs, test_size=0.99825)
# print(songs1.size)
# print("rows ="+str(len(songs1)))
# songs1.to_csv( "songs1_data.csv", index=False)


#MERGE FILES
def splitme():
    songdata = pd.read_csv("lyrics_testdata.csv")
    songdata = pd.read_csv("lyricstrain.csv")
    # print(songdata.size)
    songs1, songs2 = train_test_split(songdata, test_size=0.1, shuffle=True)
    # print(songs1.shape)
    # print(songs2.shape)
    songs1.to_csv("songs_train_set_data.csv")
    songs2.to_csv("songs_test_set_data.csv")
    # print("split file was called")
