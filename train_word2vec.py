import sqlite3
from bow import *
from random import shuffle
from nltk.classify.scikitlearn import SklearnClassifier
import nltk.classify
from sklearn.naive_bayes import BernoulliNB
from gensim.models.word2vec import *
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk.data
import re

#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_wordlist(review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = str(review)
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words



    # Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    try:
    	sentlist = [s for d in docs for s in d]
    	# the log likelihood of each sentence in this review under each w2v representation
    	llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    	# now exponentiate to get likelihoods, 
    	lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    	# normalize across models (stars) to get sentence-star probabilities
    	prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    	# and finally average the sentence probabilities to get the review probability
    	prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    	prob = prob.groupby("doc").mean()
    	return prob
    except RuntimeError:
    	print("small error - moving on")
    	#return None
   

conn = sqlite3.connect('database.sqlite')
c = conn.cursor()
aww_pos_total = [] # contains all of the useful posts
aww_pos_total_raw_text = [] # to be passed to Word2Vec model



aww_neg_total = [] # contains all of the negative posts
aww_neg_total_raw_text = []


useful_threshold = int(sys.argv[1]) 
troll_threshold = int(sys.argv[2])
min_num_words = 120
if (len(sys.argv) > 3):
	min_num_words = int(sys.argv[3])

all_pos = open('positives.txt', 'w+') # 
all_neg = open('negatives.txt', 'w+')

i = 0
j = 0

print('starting')

result_aww_pos = c.execute('SELECT * FROM May2015 WHERE subreddit="aww" AND LENGTH(body) > %i AND score > %i' % (min_num_words, useful_threshold))
for row in c:
	if i < 3000:
		all_pos.write(row[-5])
		all_pos.write('\n')
	i += 1
	#aww_pos_total_raw_text += review_to_sentences(row[-5], tokenizer)
	aww_pos_total_raw_text.append(row[-5])

result_aww_neg = c.execute('SELECT * FROM May2015 WHERE subreddit="aww" AND LENGTH(body) > %i AND score < %i' % (min_num_words, troll_threshold))
for row in c:
	if j < 3000:
		all_neg.write(row[-5])
		all_neg.write('\n')
	j += 1
	#aww_neg_total_raw_text += review_to_sentences(row[-5], tokenizer)
	aww_neg_total_raw_text.append(row[-5])
print("read shit in")

minimum = min(len(aww_pos_total_raw_text), len(aww_neg_total_raw_text))
testing = int(minimum/4)
training = int(3 * minimum/4)

print(str(testing))
print(str(training))

training_sentences_pos = aww_pos_total_raw_text[0:training]
training_sentences_neg = aww_neg_total_raw_text[0:training]

testing_sentences_pos = aww_pos_total_raw_text[-testing:]
testing_sentences_neg = aww_neg_total_raw_text[-testing:]


model_pos = Word2Vec(size=300, window=5, min_count=5, workers=4, alpha=.025)
model_neg = Word2Vec(size=300, window=5, min_count=5, workers=4)


model_pos.build_vocab(training_sentences_pos)
for epoch in range(15):
	model_pos.train(training_sentences_pos)
	model_pos.alpha -= 0.007 # decrease the learning rate
	model_pos.min_alpha = model_pos.alpha # fix the learning rate, no deca
	model_pos.train(training_sentences_pos)

model_neg.build_vocab(training_sentences_neg)
for epoch in range(15):
	model_neg.train(training_sentences_neg)
	model_neg.alpha -= 0.007 # decrease the learning rate
	model_neg.min_alpha = model_neg.alpha # fix the learning rate, no deca
	model_neg.train(training_sentences_neg)

print(model_pos)
print(model_neg)
models = [model_pos, model_neg]

results_1 = docprob(testing_sentences_pos, models)
results_2 = docprob(testing_sentences_neg, models)


if results_1 != None:
	print(str(len(results_1)))
	print("results")
	results_1 = [result for result in results_1 if result != None]
	file_1 = open('positive_tests.txt', 'w+')
	for index, row in results_1.iterrows():
		file_1.write(str(row))
		file_1.write('\n')
	file_1.close()

if results_2 != None:
	print(str(len(results_2)))
	print("results2")
	results_2 = [result for result in results_2 if result != None]
	file_2 = open('negative_tests.txt', 'w+')
	for index, row in results_2.iterrows():
		file_2.write(str(row))
		file_2.write('\n')
	file_2.close()





