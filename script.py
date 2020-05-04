# Please comment your code
import json
import re

import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Word2Vec model
from gensim.models import KeyedVectors
model_movie = KeyedVectors.load_word2vec_format("word2vec_movies_format.txt", binary=False)

def movieDF():
	# Open movie file
	with open('../data.txt', 'r', errors='replace') as f:
	    lines = (f.readlines())[:800]

	sents=[]
	for i in range(0,len(lines)-1,2):
		q = lines[i][11:-1]
		a = lines[i+1][9:-1]
		sents.append([q,a])

	return pd.DataFrame(sents, columns=['Question','Answer'])

df_movie = movieDF()

# Please comment your code
import re
import string
import pickle

def build_seq(df, model):
	seq_data = []
	whole_words = []
	max_input_words_amount = 0
	max_output_words_amount = 1
	for index, row in df.iterrows():
		question = row.Question.lower()
		answer = row.Answer
		seq_data.append([question,answer])
		# we need to tokenise question 
		for i in question:
				if i in string.punctuation: 
						question = question.replace(i," ")
		for i in question:
				if i in string.digits:
						question = question.replace(i," ")
		tokenized_q = question.split()
		# we do not need to tokenise answer (because we implement N to One model)
		# make a list with only one element (whole sentence)
		tokenized_a =[answer]
		
		# add question list and answer list (one element)
		whole_words += tokenized_q
		whole_words += tokenized_a
		
		# we need to decide the maximum size of input word tokens
		max_input_words_amount = max(len(tokenized_q), max_input_words_amount)

	# we now have a vacabulary list
	unique_words = list(set(whole_words))

	# adding special tokens in the vocabulary list
	# _B_: Beginning of Sequence
	# _E_: Ending of Sequence
	# _P_: Padding of Sequence - for different size input
	# _U_: Unknown element of Sequence - for different size input
	unique_words.append('_B_')
	unique_words.append('_E_')
	unique_words.append('_P_')
	unique_words.append('_U_')

	num_dic = {n: i for i, n in enumerate(unique_words)}

	dic_len = len(num_dic)
		
	return num_dic,dic_len,seq_data,max_input_words_amount,unique_words

def get_target_a(sentence, num_dic):
	tokenized_sentence = [sentence]
	ids = []
	for token in tokenized_sentence:
		if token in num_dic:
 			ids.append(num_dic[token])
		else:
			ids.append(num_dic['_U_'])
	return ids

def get_vectors_q(sentence, model, num_dic, max_):
	# tokenise the sentence
	max_input_words_amount = max_
	sen = sentence.lower()
	for i in sen:
		if i in string.punctuation: 
			sen = sen.replace(i," ")
	for i in sen:
		if i in string.digits:
			sen = sen.replace(i," ")
			
	tokens =	sen.split()
	tokenized_sentence = [w for w in tokens if not w in stopwords.words()]
	 
	diff = max_input_words_amount - len(tokenized_sentence)
	
	# add paddings if the word is shorter than the maximum number of words	
	for x in range(diff):
		tokenized_sentence.append('_P_')
	data = tokens_to_ids(tokenized_sentence,model,num_dic)
	return data

def get_vectors_a(sentence, model, num_dic):	
	tokenized_sentence = [sentence]
	data = tokens_to_ids(tokenized_sentence,model,num_dic)
	return data

# convert tokens to index
def tokens_to_ids(tokenized_sentence,model,num_dic):
	ids = []
	for token in tokenized_sentence:
		try:
			ids.append(model[token])
		except KeyError as e:
			ids.append(model['_U_'])
	return ids

seq_movie = build_seq(df_movie, model_movie)
movie_dic,movie_len,movie_seq,movie_max,movie_unique = seq_movie
movie_key = []
for key in movie_dic:
	movie_key.append(key)

# generate a batch data for training/testing
def make_batch(seq_data, model, num_dic, dic, max_):
	input_batch = []
	output_batch = []
	target_batch = []

	for seq in seq_data:	
		# Input for encoder cell, convert question to vector
		input_data = get_vectors_q(seq[0],model,num_dic,max_) 
		# Input for decoder cell, Add '_B_' at the beginning of the sequence data
		output_data = output_data = [model['_B_']] #[np.random.rand(100)]
		output_data += get_vectors_a(seq[1],model,num_dic)
		# Output of decoder cell (Actual result), Add '_E_' at the end of the sequence data
		target = get_target_a(seq[1],dic)
		target.append(dic['_E_'])
		array_input=np.array(input_data)
		array_output=np.array(output_data)
		# Convert each token vector to one-hot encode data
		input_batch.append(array_input)
		output_batch.append(array_output)
		target_batch.append(target)
	return input_batch, output_batch, target_batch

movie_key = []
for key in movie_dic:
	movie_key.append(key)

def build(com_len):
	learning_rate = 0.002
	n_hidden = 128
	n_class = com_len
	n_input = 100

	### Neural Network Model
	tf.reset_default_graph()

	# encoder/decoder shape = [batch size, time steps, input size]
	tf.disable_eager_execution()
	enc_input = tf.placeholder(tf.float32, [None, None, n_input])
	dec_input = tf.placeholder(tf.float32, [None, None, n_input])

	# target shape = [batch size, time steps]
	targets = tf.placeholder(tf.int64, [None, None])

	# Encoder Cell
	with tf.variable_scope('encode'):
		enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
		enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
		outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
												dtype=tf.float32)
	# Decoder Cell
	with tf.variable_scope('decode'):
		dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
		dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
		# [IMPORTANT] Setting enc_states as inital_state of decoder cell
		outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
													initial_state=enc_states,
													dtype=tf.float32)

	model = tf.layers.dense(outputs, n_class, activation=None)
	cost = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=model, labels=targets))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	return sess,model,cost,optimizer,enc_input,dec_input,targets


'''
sess,model,cost,optimizer,enc_input,dec_input,targets = build(movie_len)
input_batch, output_batch, target_batch = make_batch(movie_seq, model_movie, movie_key, movie_dic, movie_max)

saver_movie = tf.train.Saver()
total_epoch = 5000

for epoch in range(total_epoch):
		_, loss = sess.run([optimizer, cost],
											 feed_dict={enc_input: input_batch,
																	dec_input: output_batch,
																	targets: target_batch})
		if epoch % 100 == 0:
				print('Epoch:', '%04d' % (epoch + 1),
							'cost =', '{:.6f}'.format(loss))

print('Epoch:', '%04d' % (epoch + 1),
			'cost =', '{:.6f}'.format(loss))
print('Training completed')
save_path_movie = saver_movie.save(sess,"save/model_movie.ckpt")
'''

'''
import string
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def prepro(que):
	q = ""
	que = que.lower()
	for i in que:
		if i in string.punctuation: 
			que = que.replace(i," ")
	for i in que:
		if i in string.digits:
			que = que.replace(i," ")
	tokens = word_tokenize(que)
	filtered_q = [w for w in tokens if not w in stopwords.words()]
	for i in filtered_q:
		q = q + i + " "
	return q.strip()

# Please comment your code
def answer_com(sentence,max_output_words_amount):
	
	tf.reset_default_graph()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		learning_rate = 0.002
		n_hidden = 128
		n_class = movie_len
		n_input = 100
		enc_input = tf.placeholder(tf.float32, [None, None, n_input])
		dec_input = tf.placeholder(tf.float32, [None, None, n_input])
		targets = tf.placeholder(tf.int64, [None, None])
		with tf.variable_scope('encode'):
		  enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
		  enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
		  outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
												dtype=tf.float32)
		with tf.variable_scope('decode'):
		  dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
		  dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
		  outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
											  initial_state=enc_states,
											  dtype=tf.float32)
		model = tf.layers.dense(outputs, n_class, activation=None)
		cost = tf.reduce_mean(
				  tf.nn.sparse_softmax_cross_entropy_with_logits(
					  logits=model, labels=targets))
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		saver_movie = tf.train.Saver()
		saver_movie.restore(sess, "save/model_movie.ckpt")
		
		seq_data = [sentence, '_U_' * max_output_words_amount]
		input_batch, output_batch, target_batch = make_batch([seq_data],model_movie,movie_key,movie_dic,movie_max)
		prediction = tf.argmax(model, 2)
		result = sess.run(prediction,
						  feed_dict={enc_input: input_batch,
									 dec_input: output_batch,
									 targets: target_batch})

		# convert index number to actual token 
		decoded = [movie_unique[i] for i in result[0]]
		# Remove anything after '_E_'		
		if "_E_" in decoded:
			end = decoded.index('_E_')
			translated = ' '.join(decoded[:end])
		else :
			translated = ' '.join(decoded[:])
	return translated

questions = ["Hi","Hello","I am so lonely", "Can you sleep?", "What is your age?", "I hate you", "Do you like me?", "You're so mean", "Can you drive?", "That's so bad", "what do you mean?", "oh my god"]
for q in questions:
	print(q , ' ->', answer_com(prepro(q),movie_max))
'''