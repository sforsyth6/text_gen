import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import cPickle as pickle
import re
import string

def write_embed(words):
	f = open('embed.txt', 'r')
	total_embed = dict()

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		total_embed[word] = coefs
	f.close()
	
	word_embed = dict()
	for word in words:
		try:
			word_embed[word] = total_embed[word]
		except:
			print word

	embed_file = open('hhgttg_embed_words.txt', 'w')
	embed_file.write(pickle.dumps(word_embed))


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

#write_embed(list(set(word_list)))

embed = open('hhgttg_embed_words.txt', 'r')
word_embed = pickle.load(embed)

#for i in range(len(sentences)):
#	sentences[i] = sentences[i].replace('\n', ' ').lower().replace(',', '').replace('*', '').replace('_', '').replace('[', '').replace(']', '').replace('-', ' ')


t = Tokenizer(filters='')
t.fit_on_texts(list(set(word_list)))

embedding_matrix = np.zeros((len(t.word_index) + 1, 300))
for word, i in t.word_index.items():
	embedding_vector = word_embed.get(word)
    	if embedding_vector is not None:
        	embedding_matrix[i] = embedding_vector

seq = 16
x,y = [[] for i in range(2)]


for i in range(len(word_list) - seq):
	temp = []
	for word in word_list[i:i+seq]:
		ent = t.word_index.get(word)
		if ent is not None:
			temp.append(ent)
		else:
			temp.append(0)
	x.append(temp)
	ent = t.word_index.get(word_list[i+seq])
	if ent is not None:
		y.append(ent)
	else:
		y.append(0)

x = np.asarray(x)
y = np_utils.to_categorical(y)

'''
for i in range(len(raw_file) - seq):
	text = raw_file[i:i+seq]
	text_list = []
	for entry in text:
		try:
			text_list.append(t.word_index[entry])
		except:
			text_list.append(0)
	y_list = []
	try:
		y_list.append(t.word_index[raw_file[i+seq]])
	except:
		y_list.append(0)

        x.append(text_list)
        y.append(y_list)
'''

#LSTM input shape [samples, time_steps,features]
model = Sequential()
model.add(Embedding(len(t.word_index)+1,300,weights = [embedding_matrix],input_length=seq, trainable=False))
model.add(Bidirectional(LSTM(300,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(300)))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model = keras.models.load_model('models/lstm_50_50-2.0141')
model.summary()

filepath = '/workspace/models/lstm_50_{epoch:02d}-{loss:.4f}'

checkpoint = ModelCheckpoint(filepath, verbose=1,monitor='loss', save_best_only=True, mode='min')

model.fit(x,y,epochs=50,batch_size=128, callbacks = [checkpoint,TensorBoard(log_dir='./logss')])

