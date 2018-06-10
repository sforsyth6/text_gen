import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import string 

model = keras.models.load_model('models/lstm_50_20-2.2759')

raw_file = open('hhgttg.txt', 'r')

text = ''.join(raw_file.readlines()).lower().replace('\n', ' ')
#sentences = re.split('[.?!:();]', text)

all_words = re.split('(\W+)', text)

word_list = []

special = set(string.punctuation.replace("_", ""))

for val in all_words:
        new_val = val.split(' ')
        for word in new_val:
                if word != '' and ' ' not in word:
                        if any(char in special for char in word):
                                spec_val = list(word)
                                for new_spec in spec_val:
                                        word_list.append(new_spec)
                        else:
                                word_list.append(word)

t = Tokenizer(filters='')
t.fit_on_texts(list(set(word_list)))

x = []
seq = 16

rand = np.random.randint(0,len(word_list)-2*seq)

count = 0
for word in word_list[rand:]:
        entry = t.word_index.get(word)
        if entry is not None:
                x.append(entry)
                count += 1
        if count == seq:
                break

x = np.asarray(x).reshape(1,seq)
#x = x/float(len(t.word_index))
y = model.predict(x)

rev_index = dict()
for val in t.word_index.keys():
        rev_index[t.word_index.get(val)] = val

test = 1000
#norm = len(t.word_index)
words = []
rand = np.random.randint(1,200)
perm = np.random.random_integers(0,1000,rand)
for i in range(test):
        if i == 0:
                temp = ''.join([rev_index[word] for word in x[0]])
                print temp

        y = model.predict(x)

        pred = np.argmax(y)
	
	if i in perm:
		y = np.delete(y,pred)
		pred = np.argmax(y)

	try:
	        words.append(rev_index[pred])
	except:
		words.append('butt-stuff')
	if i % 250 == 0:
		words.append('\n')
        x = np.append(x,pred)
	x = x[1:].reshape(1,seq)

print ' '.join(words)
