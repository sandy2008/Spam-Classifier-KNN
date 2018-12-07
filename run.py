import numpy as np
import nltk
import csv
import re

def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)

dataset = []
allsentences = []
hamdicter = dict()
spamdicter = dict()
with open("sms_train.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        dataset.append(row)
        allsentences.append(row[1])
        if row[0] == 'ham':
            for i in extract_words(row[1]):
                if i not in hamdicter:
                    hamdicter[i] = 1
                else:
                    hamdicter[i] += 1	
        else:
            for i in extract_words(row[1]):
                if i not in spamdicter:
                    spamdicter[i] = 1
                else:
                    spamdicter[i] += 1	
                    
# lister = sorted(hamdicter.keys(), key = lambda x: hamdicter[x])
# print(lister[-6:-1])
# lister = sorted(spamdicter.keys(), key = lambda x: spamdicter[x])
# print(lister[-6:-1])
words = tokenize_sentences(allsentences)
spam = []
ham = []
for i in dataset:
	if i[0] == 'spam':
		spam.append(bagofwords(i[1], words))
	else:
		ham.append(bagofwords(i[1], words))

testset = []
with open("sms_test.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        testset.append(bagofwords(row[1], words))
        
for i in testset:
	distspam = 0
	distham = 0
	for j in spam:
		distspam += np.linalg.norm(i-j)
	distspam = distspam / len(spam)
	for j in ham:
		distham += np.linalg.norm(i-j)
	distham = distham / len(ham)
	if distspam > distham:
		print("SPAM", i)
	else:
		print("HAM", i)
		
# print(proc)
