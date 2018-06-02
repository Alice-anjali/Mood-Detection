import tensorflow as tf
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import collections
import numpy as np
import random
import math
import datetime as dt
import pickle

train = pd.read_csv("lyrics1_500.csv")


def lyrics_to_words(raw_lyric):
        lyric_text = BeautifulSoup(raw_lyric, "html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", lyric_text)
        words = letters_only.lower().split()
        # stops = set(stopwords.words("english"))
        # meaningful_words = [w for w in words if not w in stops]
        meaningful_words = [w for w in words]
        return(" ".join(meaningful_words))

num_lyrics = train["Lyrics"].size
clean_train_lyrics = []
words = []
sentences = []
for i in range(0,num_lyrics):
    if((i+1)%10 == 0):
        print("Lyric %d of %d\n" % ( i+1, num_lyrics ))
    get_lyrics = lyrics_to_words(train["Lyrics"][i])
    clean_train_lyrics.append(get_lyrics)
    for word in get_lyrics.split():
        if word != ' ':
            words.append(word)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.  """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


max_words = 10000
data, count, dictionary, reversed_dictionary = build_dataset(words, max_words)
vocabulary_size = len(count)

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a context.



data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


graph = tf.Graph()

with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the softmax
    weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))
    hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

    # convert train_context to a one-hot format
    train_one_hot = tf.one_hot(train_context, vocabulary_size)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,labels=train_one_hot))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()



def run(graph, num_steps):
    print(graph)
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val


        if step % 10 == 0:
          if step > 0:
            average_loss /= 10
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reversed_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reversed_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()
      return final_embeddings
      # print(final_embeddings)
      # print(final_embeddings.shape)
      # print(final_embeddings[dictionary['sad']])


num_steps = 500
softmax_start_time = dt.datetime.now()
word2vec_model = run(graph, num_steps=num_steps)
filename = 'Word2Vec_Model'
pickle.dump(word2vec_model, open(filename, 'wb'))
softmax_end_time = dt.datetime.now()
print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))
file_name = 'Dictionary'
pickle.dump(dictionary, open(file_name, 'wb'))
