import sys
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import scipy.stats as ss
import numpy.polynomial.polynomial as poly

import common

# Using TFIDF to vectorize documents and taking cosine of them to find the similarity between each doc.

		
class TFIDF():
	def __init__(self):
		self.vocabulary = None

	def fit(self, train: list):
		voc = dict()
		df = []

		for doc in train:
			# Word appeared in a document 
			lvoc = dict()
			for word in doc:
				# If found a word not yet seen before in a document
				if word not in lvoc:
					# If it was appeared in a process
					if word in voc:
						# Add DF score by 1
						df[voc[word]] += 1
					else:
						# Not in the vocabulary, add it
						voc[word] = len(df)
						df.append(1)

		# Set new vocabulary
		self.vocabulary = voc
		# Set learnt IDF
		df = np.array(df)
		# IDF = log(1/(DF/N)) = log(N/DF) ~ log(N/(DF+1)) = logN - log(DF+1)
		self.n = len(train)
		self.idf = np.log(self.n) - np.log(df + 1)
		# IDF for not-known word
		self.nidf = np.log(self.n)

	def tf(self, doc) -> (np.array, dict):
		"""
		Calculates and returns TF vector according to its learnt vocabulary and IDF values.
		It also returns TF values of words that was not appeared in train documents, therefore not leant,
		in dict form which has word as key, TF as value.
		"""
		# Array for storing TF vector index is according to self.vocabulary
		tfv = [0] * len(self.vocabulary)
		# Short for non-appearing term frequency
		ntfd = dict()
		for word in doc:
			if word in self.vocabulary:
				tfv[self.vocabulary[word]] += 1
			else:
				if word in ntfd:
					ntfd[word] += 1
				else:
					ntfd[word] = 1
		
		tfv = np.array(tfv)

		return tfv, ntfd

	def tfidf(self, doc) -> (np.array, dict):
		tfv, ntfd = self.tf(doc)
		tfidfv = tfv * self.idf

		ntfidf = dict()
		for word, tf in ntfd.items():
			# Use non-appearing IDF for unknown words
			ntfidf[word] = tf * self.nidf

		return tfidfv, ntfidf

	def produce_score(self, doc_pair):
		# Calculate TFIDF vector and dict
		lv, ld = self.tfidf(doc_pair[0])
		rv, rd = self.tfidf(doc_pair[1])

		# Calculate cosine of two vectors and dicts
		# cos(l, r) = inner(l, r) / (norm(l) * norm(r))
		# <=> cos(lv, ld, rv, rd) = (inner(lv, rv) + sum_k(k is in ld and rd)( ld(k)*rd(k) )) / sqrt(( lv*lv + sum_i(ld(i)^2) ) * ( rv*rv + sum_i(rd(i)^2) ))
		# = ip / lnorm * rnorm
		ip = np.inner(lv, rv)
		lnorm = np.inner(lv, lv)
		rnorm = np.inner(rv, rv)

		for word in ld.keys():
			if word in rd:
				k = word
				ip += ld[k] * rd[k]

		for v in ld.values():
			lnorm += v**2

		for v in rd.values():
			rnorm += v**2


		if lnorm == 0 or rnorm == 0:
			print(doc_pair)
			print(lnorm, rnorm)

		cos_sim = ip / np.sqrt(lnorm * rnorm)

		return cos_sim * 5

def read_docs(fname: str) -> (list, list):
	docs = []
	with open(fname, 'rt', encoding='utf_8_sig') as fin:
		for line in fin:
			splitted = line[:-1].split('\t')
			docs.append([ splitted[0].split(' '), splitted[1].split(' ') ])

	return docs

CHAR_REMOVE_TARGETS = list(',.()')
with open(common.FILE_STOP_WORDS, 'rt') as file:
	STOP_WORDS = [ line[:-1] for line in file.readlines() ]

def pre_process_doc(doc: list):
	# Convert all words to lower case
	doc = map(lambda word: word.lower(), doc)
	# Remove a specific character if it exists
	doc = map(lambda word: ''.join([ ch for ch in word if ch not in CHAR_REMOVE_TARGETS ]), doc)
	# Remove stop-words
	doc = filter(lambda word: word not in STOP_WORDS, doc)

	return list(doc)

def pre_process(pairs: list):
	for pair in pairs:
		for i in range(len(pair)):
			pair[i] = pre_process_doc(pair[i])

def train_and_score(x_train, x_test, y_train, y_test, poly_num):
	# Initialize and train
	transf = TFIDF()
	transf.fit(np.concatenate(np.array(x_train)))

	# Calculate consine similarity between a right and left sentence of the train data.
	a_train = [ transf.produce_score(train) for train in x_train ]
	# And test data.
	a_test = [ transf.produce_score(test) for test in x_test ]

	if poly_num == 0:
		s_train = a_train
		s_test = a_test

		return s_train, s_test, transf, np.array([])
	else:
		# Fit polynomial model to output score
		poly_coeff = poly.polyfit(a_train, y_train, poly_num)
		# Apply it to train data
		s_train = poly.polyval(a_train, poly_coeff)
		s_test = poly.polyval(a_test, poly_coeff)
	
		return s_train, s_test, transf, poly_coeff

def kfold(grand_x_train: list, grand_y_train: list, poly_num: int):
	print('# KFold in train data')

	# Needed to be wrapped by numpy array to easily split train data
	grand_x_train = np.array(grand_x_train)
	grand_y_train = np.array(grand_y_train)

	# Set up kfold
	kfold = KFold(n_splits=10, shuffle=True)

	# Train and test
	peas_train_history = []
	peas_test_history = []
	for train_index, test_index in kfold.split(grand_x_train):
		print('## Split')
		x_train = grand_x_train[train_index]
		x_test = grand_x_train[test_index]
		y_train = grand_y_train[train_index]
		y_test = grand_y_train[test_index]

		# Fit and test for this split
		s_train, s_test, _, _ = train_and_score(x_train, x_test, y_train, y_test, poly_num)

		# Record pearson coefficient
		peas_train = ss.pearsonr(s_train, y_train)
		peas_test = ss.pearsonr(s_test, y_test)
		peas_train_history.append(peas_train)
		peas_test_history.append(peas_test)

		print('For train data:\t%f (p: %f)' % peas_train)
		print('For test data:\t%f (p: %f)' % peas_test)

	print('### KFold Mean')
	peas_train_mean = np.mean([ c for c, p in peas_train_history ])
	peas_test_mean = np.mean([ c for c, p in peas_test_history ])
	print('For train data:\t%f' % peas_train_mean)
	print('For test data:\t%f' % peas_test_mean)

	return peas_train_history, peas_test_history

def whole(x_train: list, x_test: list, y_train: list, y_test, poly_num: int):
	print('# Whole')
	s_train, s_test, transf, poly_coeff = train_and_score(x_train, x_test, y_train, y_test, poly_num)
	
	# Save vocabulary for future improvement
	with open(common.FILE_VOCABULARY, 'wt') as file:
		file.write(str(transf.vocabulary))

	# Also save polynomial coefficent
	np.savetxt(common.FILE_POLY_COEFF, poly_coeff)

	# Save answers for plot use
	np.savetxt(common.FILE_TRAIN_RESULT, s_train)
	np.savetxt(common.FILE_TEST_RESULT, s_test)

	# Calculate peason coefficent
	peas_train = ss.pearsonr(s_train, y_train)
	peas_test = ss.pearsonr(s_test, y_test)
	print('For train data:\t%f (p: %f)' % peas_train)
	print('For test data:\t%f (p: %f)' % peas_test)
	return peas_train, peas_test

def main():
	# Read the train data. the train data have to be concatenated for training.
	x_train = read_docs(common.FILE_TRAIN_INPUT)
	x_test = read_docs(common.FILE_TEST_INPUT)
	y_train = np.loadtxt(common.FILE_TRAIN_SCORE)
	y_test = np.loadtxt(common.FILE_TEST_SCORE)

	# Preprocess
	if len(sys.argv) < 2 or sys.argv[1].lower() == 'true':
		print('! Preprocess Enabled')
		pre_process(x_train)
		pre_process(x_test)

	# Polynomial coefficent number
	if len(sys.argv) < 3:
		poly_num = 5
	else:
		poly_num = int(sys.argv[2])

	# Do kfold
	peas_train_history, peas_test_history = kfold(x_train, y_train, poly_num)
	# Do whole
	peas_train, peas_test = whole(x_train, x_test, y_train, y_test, poly_num)
	
if __name__ == '__main__':
	main()
