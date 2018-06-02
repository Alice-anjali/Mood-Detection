import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import re

word2vec_model = pickle.load(open('Word2Vec_Model', 'rb'))
dictionary = pickle.load(open('Dictionary', 'rb'))
train = pd.read_csv("lyrics1_500.csv")
numWords = []
for pf in train['Lyrics']:
    counter = len(pf.split())
    numWords.append(counter)

maxSeqLength = max(numWords)
row = len(train)
numFiles = len(train)
# idsMatrix = []

for i in range(row):
    fname = train['Lyrics'][i]
    # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    def cleanSentences(string):
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    firstFile = np.zeros((maxSeqLength), dtype='int32')
    indexCounter = 0
    cleanedLine = cleanSentences(fname)
    split = cleanedLine.split()
    for word in split:
        if word in dictionary:
            firstFile[indexCounter] = dictionary[word]
        else:
            firstFile[indexCounter] = 399999 #Vector for unknown words

        indexCounter = indexCounter + 1

    # idsMatrix.append(firstFile)


# idsMatrix_numpy = np.array(idsMatrix)
# print(idsMatrix_numpy.shape)

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for pf in train['Lyrics']:
    indexCounter = 0
    cleanedLine = cleanSentences(pf)
    split = cleanedLine.split()
    for word in split:
        if word in dictionary:
            ids[fileCounter][indexCounter] = dictionary[word]
        else:
            ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
                    break

    fileCounter = fileCounter + 1

#Pass into embedding function and see if it evaluates.

np.save('idsMatrix', ids)

batchSize = 24
lstmUnits = 64
numClasses = 4 #happy, sad, angry, relaxed
iterations = 100000

ids = np.load('idsMatrix.npy')


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
    reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
    counter = counter + 1.
    return reviewFeatureVecs



#
# tf.reset_default_graph()
#
# labels = tf.placeholder(tf.float32, [batchSize, numClasses])
# input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#
# print("shape = "+str(word2vec_model.shape[1]))
# numDimensions = word2vec_model.shape[1]
# data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.nn.embedding_lookup(word2vec_model,input_data)
#
# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
# value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#
# weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
# bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# value = tf.transpose(value, [1, 0, 2])
# last = tf.gather(value, int(value.get_shape()[0]) - 1)
# print(last)
# prediction = (tf.matmul(last, weight) + bias)
#
# correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
# print(correctPred)
