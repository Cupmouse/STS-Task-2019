from gensim.models import KeyedVectors
from gensim.models import FastText
import numpy as np
np.seterr('raise')
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
import nltk
import scipy

w2v = FastText.load_fasttext_format('cc.en.300')
# w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

TRAIN_FILES = [
              'MSRpar',
              'MSRvid',
              'OnWN',
              'SMTeuroparl',
              'deft-forum',
              'deft-news',
              'headlines',
              'tweet-news'
              ]
def load_data(name):
  y = np.loadtxt('data/STS.gs.%s.txt' % name)

  data = []
  with open('data/STS.input.%s.txt' % name, mode='rt', encoding='utf-8-sig') as fin:
    lines = fin.readlines()
  for i, line in enumerate(lines):
    pair = [ doc[:-1].split() for doc in line.split('\t') ]
    pair.append(y[i])
    data.append(pair)
  
  return data

def load():
  trains = [ load_data(file) for file in TRAIN_FILES ]
  test = load_data('images')
  return trains, test

def doc_to_vector(doc):
  word_vecs = []
  for word in doc:
    if word in w2v.wv:
      word_vecs.append(w2v.wv[word])
    else:
      print('Warning: Vector for word ', word, 'does not exist')
  if not np.any(word_vecs):
    return np.zeros((300,))
  else:
    return np.mean(word_vecs, axis=0)

def data_to_vectors(data):
  vecs = np.zeros(shape=(len(data), 2, 300), dtype=np.float64)
  for i, pair in enumerate(data):
    vecs[i][0] = doc_to_vector(pair[0])
    vecs[i][1] = doc_to_vector(pair[1])
    
  return vecs

CHAR_REMOVE_TARGETS = list(',.()#"$?!&%:\'')
with open('stop_words.txt', 'rt') as file:
	STOP_WORDS = [ line[:-1] for line in file.readlines() ]
ps = nltk.stem.PorterStemmer()
def preprocess_doc(sent_array):
  # Convert all words to lower case
  sent_array = map(lambda word: word.lower(), sent_array)
  # Remove a specific character if it exists
  sent_array = map(lambda word: ''.join([ ch for ch in word if ch not in CHAR_REMOVE_TARGETS ]), sent_array)
  # Stem
  sent_array = map(lambda word: ps.stem(word), sent_array)
  # Remove stop-words
  sent_array = filter(lambda word: word not in STOP_WORDS, sent_array)
	# Remove numbers
  # sent_array = filter(lambda word: not word.isdigit(), sent_array)
  # Remove words which is less than 4
  sent_array = filter(lambda word: len(word) > 4, sent_array)

  return list(sent_array)

def preprocess(data, discard_empty=True):
  new_data = []
  for pair in data:
    ls, rs = preprocess_doc(pair[0]), preprocess_doc(pair[1])
    if len(ls) == 0 or len(rs) == 0:
      print('Warning: Sentence is empty after pre-processing.', pair[0], pair[1])
      if not discard_empty:
        new_data.append([ls, rs, pair[2]])
    else:
      new_data.append([ls, rs, pair[2]])
  return new_data


trains, test = load()
trains = [ preprocess(train, discard_empty=True) for train in trains ]
train_concat = np.concatenate(trains, axis=0)

print('Preprocessing test data')
test = preprocess(test, discard_empty=False)

pca = PCA(n_components=2)
pca.fit(np.reshape(data_to_vectors(train_concat), newshape=(-1, 300)))
# pca.fit(np.concatenate((all_vecs, test_vecs), axis=0))

plt.figure(num=None, figsize=(32, 16), dpi=80, facecolor='w', edgecolor='k')
for i, data in enumerate(trains):
  p_pca = pca.transform(np.reshape(data_to_vectors(data), newshape=(-1, 300)))
  plt.subplot(3, 3, i+1)
  plt.scatter(p_pca[:, 0], p_pca[:, 1], s=[4]*len(p_pca))
  plt.xticks(np.arange(21)/10 - 1)
  plt.yticks(np.arange(21)/10 - 1)
  plt.title(TRAIN_FILES[i])

p_pca = pca.transform(np.reshape(data_to_vectors(test), newshape=(-1, 300)))
plt.subplot(3, 3, 9)
plt.scatter(p_pca[:, 0], p_pca[:, 1], s=[4]*len(p_pca))
plt.xticks(np.arange(21)/10 - 1)
plt.yticks(np.arange(21)/10 - 1)
plt.title('TEST: images')

plt.show()

def show_clusters_losses(data):
  hkmeans = []
  hloss = []

  for n in range(2,50):
    kmeans = KMeans(n)
    vecs_in = np.reshape(data_to_vectors(data), newshape=(-1, 300))
    labels = kmeans.fit_predict(vecs_in)
    dist_matrix = kmeans.transform(vecs_in)

    loss_vec = np.zeros(shape=(len(labels),))
    for i in range(len(labels)):
      loss_vec[i] = dist_matrix[i][labels[i]]
    loss = np.sum(loss_vec)
    hloss.append(loss)

  plt.plot(np.arange(2, 50), hloss)
  plt.title('Loss')
  plt.show()

def filter_small_group(data, n_clusters, size):
  list_of_vectors = np.reshape(data_to_vectors(data), (-1, 300))
  kmeans = KMeans(n_clusters)
  labels = kmeans.fit_predict(list_of_vectors)

  plt.figure(num=None, figsize=(32, 16), dpi=80, facecolor='w', edgecolor='k')
  for i in range(n_clusters):
    d = pca.transform([ vec for j, vec in enumerate(list_of_vectors) if labels[j] == i ])
    plt.scatter(d[:, 0], d[:, 1], s=[2])
  plt.show()

  uniq, counts = np.unique(labels, return_counts=True)
  plt.bar(uniq, counts)
  plt.show()
  print(counts)

  acceptable_groups = uniq[np.where(counts >= size)]
  new_data = []
  for pair in data:
    docs_group = kmeans.predict([ doc_to_vector(pair[0]), doc_to_vector(pair[1]) ])
    if docs_group[0] in acceptable_groups and docs_group[1] in acceptable_groups:
      new_data.append(pair)

  return new_data

def limit_samples_per_score(data, merge, limit):
  # Limit number of samples per class
  factor = 1/merge
  print('merging scores into ', factor, ' per 1.0')
  y_rounded = np.round(np.array([ pair[2] for pair in data ])*factor, decimals=0)/factor

  uniq = np.unique(y_rounded)

  new_data = []
  for y in uniq:
    target_indexes = np.where(y_rounded == y)[0]
    choosen = np.random.choice(target_indexes, np.min((limit, target_indexes.shape[0])), replace=False)
    new_data += [ data[i] for i in choosen ]

  return new_data

def show_bar(data):
  # Show bar graph of y distribution
  uniq, counts = np.unique([ pair[2] for pair in data ], return_counts=True)
  plt.bar(uniq, counts, width=0.05)
  plt.show()
  print(counts)

def merge_score(data, merge):
  factor = 1/merge
  print('merging scores into ', factor, ' per 1.0')

  new_data = []
  for i, pair in enumerate(data):
    new_data.append([pair[0], pair[1], round(pair[2]*factor)/factor])

  return new_data

def filter_far_group(data, n_clusters, limit):
  list_of_vectors = np.reshape(data_to_vectors(data), (-1, 300))
  kmeans = KMeans(n_clusters)
  distances = kmeans.fit_transform(list_of_vectors)
  labels = np.argmin(distances, axis=1)
  
  # Sum of distance for each group, from center of group to point 
  distances = np.sum(distances, axis=0)

  plt.bar(np.arange(distances.shape[0]), distances)
  plt.show()

  acceptable_groups = np.where(distances <= limit)[0]
  new_data = []
  for i, pair in enumerate(data):
    if labels[2*i] in acceptable_groups and labels[2*i+1] in acceptable_groups:
      new_data.append(pair)

  return new_data

def test_normality(data):
  chi2, p = scipy.stats.normaltest(data[:, 0: 300])
  print(chi2, p)
  chi2, p = scipy.stats.normaltest(data[:, 1: 300])
  print(chi2, p)

train = train_concat
# show_bar(train)
# # train = filter_small_group(train, 15, 400)
# # train = filter_far_group(train, 15, 6000)
# # show_bar(train)
# # train = limit_samples_per_score(train, 0.5, 400)
# show_bar(train)
# # train = merge_score(train, 1)
# show_bar(train)
# print(len(train))
# test_normality(train)


# Output to file
with open('STS.input.preprocessed.train.txt', 'wt', encoding='utf-8') as file:
  file.writelines([ ' '.join(pair[0]) + '\t' + ' '.join(pair[1]) + '\n' for pair in train ])
with open('STS.input.preprocessed.images.txt', 'wt', encoding='utf-8') as file:
  file.writelines([ ' '.join(pair[0]) + '\t' + ' '.join(pair[1]) + '\n' for pair in test ])

np.savetxt('STS.gs.preprocessed.train.txt', [ pair[2] for pair in train ], encoding='utf-8')
np.savetxt('STS.gs.preprocessed.test.txt', [ pair[2] for pair in test ], encoding='utf-8')

plt.figure(num=None, figsize=(32, 16), dpi=80, facecolor='w', edgecolor='k')
p_pca = pca.transform(np.reshape(data_to_vectors(train_concat), (-1, 300)))
plt.scatter(p_pca[:, 0], p_pca[:, 1], s=[4]*len(p_pca))
# p_pca = pca.transform(to_vectors(x_test))
# plt.scatter(p_pca[:, 0], p_pca[:, 1], s=[4]*len(p_pca))

plt.xticks(np.arange(21)/10 - 1)
plt.yticks(np.arange(21)/10 - 1)
plt.show()