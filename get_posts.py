import sqlite3
import sys


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

all_pos = open('aww_pos_50.txt', 'w+') # 
all_neg = open('aww_neg_50.txt', 'w+')

i = 0
j = 0

print('starting')

result_aww_pos = c.execute('SELECT * FROM May2015 WHERE subreddit="politics" AND LENGTH(body) > %i AND score > %i LIMIT 100' % (min_num_words, useful_threshold))
for row in c:
	if i < 3000:
		all_pos.write(row[-5])
		all_pos.write('\n')
	i += 1
	#aww_pos_total_raw_text += review_to_sentences(row[-5], tokenizer)
	aww_pos_total_raw_text.append(row[-5])
print('did half')

result_aww_neg = c.execute('SELECT * FROM May2015 WHERE subreddit="politics" AND LENGTH(body) > %i AND score < %i LIMIT 100' % (min_num_words, troll_threshold))
for row in c:
	if j < 3000:
		all_neg.write(row[-5])
		all_neg.write('\n')
	j += 1
	#aww_neg_total_raw_text += review_to_sentences(row[-5], tokenizer)
	aww_neg_total_raw_text.append(row[-5])
print("read shit in")
all_neg.close()
all_pos.close()
