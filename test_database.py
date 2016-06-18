import sqlite3
from bow import *
from random import shuffle
from nltk.classify.scikitlearn import SklearnClassifier
import nltk.classify
from sklearn.naive_bayes import MultinomialNB
import sys

# python test_database.py 100 0

def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
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

conn = sqlite3.connect('database.sqlite')
c = conn.cursor()
aww_pos_total = [] # contains all of the useful posts
aww_pos_total_raw_text = [] # to be passed to Word2Vec model


aww_neg_total = [] # contains all of the negative posts
aww_pos_total_raw_text = [] # to be passed to Word2Vec model

useful_threshold = int(sys.argv[1]) 
troll_threshold = int(sys.argv[2])
min_num_words = 120
if (len(sys.argv) > 3):
	min_num_words = sys.argv[3]
subreddit = "aww"
if (len(sys.argv) > 4):
	subreddit = sys.argv[4]

all_pos = open('positives.txt', 'w+') # 
all_neg = open('negatives.txt', 'w+')

i = 0
j = 0

result_aww_pos = c.execute('SELECT * FROM May2015 WHERE subreddit="videos" AND LENGTH(body) > %i AND score > %i' % (min_num_words, useful_threshold))
for row in c:
	doc = Document()
	if i < 3000:
		all_pos.write(row[-5])
		all_pos.write('\n')
	i += 1
	doc.read_text(row[-5])
#	aww_pos_total_raw_text.append()
	aww_pos_total.append(dict(doc))

result_aww_neg = c.execute('SELECT * FROM May2015 WHERE subreddit="videos" AND LENGTH(body) > %i AND score < %i' % (min_num_words, troll_threshold))
for row in c:
	if j < 3000:
		all_neg.write(row[-5])
		all_neg.write('\n')
	j += 1
	doc = Document()
	doc.read_text(row[-5])
	aww_neg_total.append(dict(doc))
print("read shit in")
all_pos.close()
all_neg.close()

print(str(len(aww_pos_total)))
print(str(len(aww_neg_total)))
minimum = min(len(aww_pos_total), len(aww_neg_total))
testing = int(minimum/4)
training = int(3 * minimum/4)
print(str(minimum))
aww_pos_training = aww_pos_total[0:training] # get first 2000 positive posts
aww_neg_training = aww_neg_total[0:training] # get first 2000 negative posts

aww_pos_testing = aww_pos_total[-testing:] # get last 100 positive posts
aww_neg_testing = aww_neg_total[-testing:]
total_1 = len(aww_pos_testing)
total_2 = len(aww_neg_testing)
training_data = []
for dictionary in aww_pos_training:
	training_data.append((dictionary, "pos"))
for dictionary2 in aww_neg_training:
	training_data.append((dictionary2, "neg"))
shuffle(training_data)

print("classifying and training")
classif = SklearnClassifier(MultinomialNB()).train(training_data)
results_pos = classif.classify_many(aww_pos_testing)
results_neg = classif.classify_many(aww_neg_testing)

correct_1 = 0
correct_2 = 0

print("testing")

for string in results_pos:
	if "pos" in string:
		correct_1 += 1
for string in results_neg:
	if "neg" in string:
		correct_2 += 1

print("For positive: %i correct out of %i for a percentage of %f" % (correct_1, total_1, correct_1/total_1))
print("For negative: %i correct out of %i for a percentage of %f" % (correct_2, total_2, correct_2/total_2))
print("total classification percentage is %f" % ((correct_1 + correct_2)/(total_1 + total_2)))

