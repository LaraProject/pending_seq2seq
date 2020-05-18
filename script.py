import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers, activations, models, preprocessing
from tensorflow.keras import preprocessing, utils
import os
import yaml
import requests
import zipfile
import io
import re
from gensim.models import Word2Vec
import re
from gensim.models import KeyedVectors
import argparse

# Global variables

# Make a TensorFlow Tokenizer
tokenizer = preprocessing.text.Tokenizer()
questions = []
answers = []
vocab = []
model_w2v = None
embedding_matrix = None
maxlen_questions = 0
maxlen_answers = 0
VOCAB_SIZE = 0

# Import data
def import_data():
	# Download data set
	r = requests.get('https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip?raw=true')
	z = zipfile.ZipFile(io.BytesIO(r.content))
	z.extractall()

# Preprocess the data
def preprocess_data():
	global questions
	global answers

	dir_path = 'chatbot_nlp/data'
	files_list = os.listdir(dir_path + os.sep)

	# Get questions and answers
	for filepath in files_list:
		stream = open(dir_path + os.sep + filepath, 'rb')
		docs = yaml.safe_load(stream)
		conversations = docs['conversations']
		for con in conversations:
			if len(con) > 2:
				questions.append(con[0])
				replies = con[1:]
				ans = ''
				for rep in replies:
					ans += ' ' + rep
				answers.append(ans)
			elif len(con) > 1:
				questions.append(con[0])
				answers.append(con[1])

	# Filter out non-string questions
	answers_with_tags = list()
	for i in range(len(answers)):
		if type(answers[i]) == str:
			answers_with_tags.append(answers[i])
		else:
			questions.pop(i)

	# Tokenize answers
	answers = list()
	for i in range(len(answers_with_tags)):
		answers.append('<start> ' + answers_with_tags[i] + ' <end>')

# Prefilter before tokenizer
def clean_text(text):
	text = text.lower()
	text = re.sub(r"i'm", 'i am', text)
	text = re.sub(r"he's", 'he is', text)
	text = re.sub(r"it's", 'it is', text)
	text = re.sub(r"she's", 'she is', text)
	text = re.sub(r"that's", 'that is', text)
	text = re.sub(r"what's", 'what is', text)
	text = re.sub(r"where's", 'where is', text)
	text = re.sub(r"how's", 'how is', text)
	text = re.sub(r"\'ll", ' will', text)
	text = re.sub(r"\'ve", ' have', text)
	text = re.sub(r"\'re", ' are', text)
	text = re.sub(r"\'d", ' would', text)
	text = re.sub(r"n't", ' not', text)
	text = re.sub(r"won't", 'will not', text)
	text = re.sub(r"can't", 'cannot', text)
	return text

def clean_everything():
	global questions
	global answers
	questions = [clean_text(s) for s in questions]
	answers = [clean_text(s) for s in answers]

def load_word2vec(model_path):
	global model_w2v
	model_w2v = KeyedVectors.load_word2vec_format(model_path, binary=False)

def fit_tokenizer():
	global tokenizer
	global VOCAB_SIZE
	tokenizer.fit_on_texts(questions + answers + [["<unk>","<start>","<end>"]])
	VOCAB_SIZE = len(tokenizer.word_index) + 1
	#print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

def fill_vocab():
	global vocab
	vocab = []
	for word in tokenizer.word_index:
		vocab.append(word)

# Add a token for words which aren't in the model
def replace_unknown_words():
	unknown_words = []
	for word in tokenizer.word_index:
		if word not in model_w2v.vocab:
			unknown_words.append(word)
	for q in questions:
		for unk in unknown_words:
			q.replace(" " + unk + " ", ' <unk> ')
	for a in answers:
		for unk in unknown_words:
			a.replace(" " + unk + " ", ' <unk> ')
	return unknown_words

# Create the embedding matrix
def create_embedding_matrix(unknown_words):
	global questions
	global answers
	global embedding_matrix
	embedding_matrix = np.zeros((VOCAB_SIZE, 300))
	for i in range(len(tokenizer.word_index)):
		if vocab[i] in unknown_words:
			embedding_matrix[i] = model_w2v['<unk>']
		else:
			embedding_matrix[i] = model_w2v[vocab[i]]

# Create input and output datasets
# encoder_input_data
def create_input_output():
	tokenized_questions = tokenizer.texts_to_sequences(questions)
	maxlen_questions = max([len(x) for x in tokenized_questions])
	padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions,
			maxlen=maxlen_questions, padding='post')
	encoder_input_data = np.array(padded_questions)
	#print((encoder_input_data.shape, maxlen_questions))

	# decoder_input_data
	tokenized_answers = tokenizer.texts_to_sequences(answers)
	maxlen_answers = max([len(x) for x in tokenized_answers])
	padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers,
			maxlen=maxlen_answers, padding='post')
	decoder_input_data = np.array(padded_answers)
	#print((decoder_input_data.shape, maxlen_answers))

	# decoder_output_data
	tokenized_answers = tokenizer.texts_to_sequences(answers)
	for i in range(len(tokenized_answers)):
		tokenized_answers[i] = (tokenized_answers[i])[1:]
	padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers,
			maxlen=maxlen_answers, padding='post')
	onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
	decoder_output_data = np.array(onehot_answers)
	#print(decoder_output_data.shape)

	return encoder_input_data, decoder_input_data, decoder_output_data

# Defining the Encoder-Decoder model
def create_model(encoder_input_data, decoder_input_data, decoder_output_data):
	encoder_inputs = tf.keras.layers.Input(shape=(None, ))
	encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 300,
			mask_zero=False, weights=[embedding_matrix], trainable=False, input_length=maxlen_questions)(encoder_inputs)
	encoder_embedding = tf.keras.layers.SpatialDropout1D(0.2)(encoder_embedding)
	(encoder_outputs, state_h, state_c) = tf.keras.layers.LSTM(300,
			return_state=True)(encoder_embedding)
	encoder_states = [state_h, state_c]

	decoder_inputs = tf.keras.layers.Input(shape=(None, ))
	decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 300,
			mask_zero=False, weights=[embedding_matrix], trainable=False, input_length=maxlen_answers)(decoder_inputs)
	decoder_embedding = tf.keras.layers.SpatialDropout1D(0.2)(decoder_embedding)
	decoder_lstm = tf.keras.layers.LSTM(300, return_state=True,
										return_sequences=True, recurrent_dropout=0.2)
	(decoder_outputs, _, _) = decoder_lstm(decoder_embedding,
			initial_state=encoder_states)
	decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE,
			activation=tf.keras.activations.softmax)
	output = decoder_dense(decoder_outputs)

	model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
	model.compile(optimizer=tf.keras.optimizers.Adam(),
				  loss='categorical_crossentropy')
	model.summary()
	return model, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs

# Training the model
def train():
	encoder_input_data, decoder_input_data, decoder_output_data = create_input_output()
	model, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs = create_model(encoder_input_data, decoder_input_data, decoder_output_data)
	model.fit([encoder_input_data, decoder_input_data],
			  decoder_output_data, batch_size=32, epochs=200)
	return encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs

# Defining inference models
def make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs):
	encoder_model = tf.keras.models.Model(encoder_inputs,
			encoder_states)
	decoder_state_input_h = tf.keras.layers.Input(shape=(300, ))
	decoder_state_input_c = tf.keras.layers.Input(shape=(300, ))
	decoder_states_inputs = [decoder_state_input_h,
							 decoder_state_input_c]
	(decoder_outputs, state_h, state_c) = decoder_lstm(decoder_embedding,
					 initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = tf.keras.models.Model([decoder_inputs]
			+ decoder_states_inputs, [decoder_outputs] + decoder_states)
	return (encoder_model, decoder_model)

# Save the inference model
def save_inference_model(path, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs):
	(encoder_model, decoder_model) = make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)
	encoder_model.save(path + '/model_enc.h5')
	decoder_model.save(path + '/model_dec.h5')

# Save the tokenizer
def save_tokenizer(path):
	with open(path + '/tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the inference model
def load_inference_model(enc_file, dec_file):
	encoder_model = tf.keras.models.load_model(enc_file)
	decoder_model = tf.keras.models.load_model(dec_file)
	return (encoder_model, decoder_model)

# Load the tokenizer
def load_tokenizer(tokenizer_file):
	with open(tokenizer_file, 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer

# Talking with our Chatbot
def str_to_tokens(sentence : str ):
	words = sentence.lower().split()
	tokens_list = list()
	for word in words:
		if word in tokenizer.word_index:
			tokens_list.append(tokenizer.word_index[word])
		else:
			tokens_list.append(tokenizer.word_index["<unk>"])
	return preprocessing.sequence.pad_sequences([tokens_list],
			maxlen=maxlen_questions, padding='post')

# Ask multiple questions
def ask_questions(enc_model, dec_model):
	for _ in range(10):
		states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
		empty_target_seq = np.zeros((1, 1))
		empty_target_seq[0, 0] = tokenizer.word_index['<start>']
		stop_condition = False
		decoded_translation = ''
		while not stop_condition:
			(dec_outputs, h, c) = dec_model.predict([empty_target_seq]
					+ states_values)
			sampled_word_index = np.argmax(dec_outputs[0, -1, :])
			sampled_word = None
			for (word, index) in tokenizer.word_index.items():
				if sampled_word_index == index:
					decoded_translation += ' {}'.format(word)
					sampled_word = word

			if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
				stop_condition = True

			empty_target_seq = np.zeros((1, 1))
			empty_target_seq[0, 0] = sampled_word_index
			states_values = [h, c]

		print(decoded_translation[:-4].replace("<unk>",""))  # remove end w

# Argument management
argslist = argparse.ArgumentParser(description="Seq2Seq Neural Network")

argslist.add_argument('word2vec_model', metavar='word2vec_model', type=str,
		help='Path to the word2vec model')
argslist.add_argument('--downloadData', metavar='[True/False]', type=bool,
        help='Specify whether the dataset should be downloaded', default=False, required=False)
argslist.add_argument('--speak', metavar='[True/False]', type=bool,
        help='Specify whether to speak with the Network', default=False, required=False)
argslist.add_argument('--saveModel', metavar='path', type=str,
        help='Specify the path where to save the model', default='', required=False)
argslist.add_argument('--loadModel', metavar='path', type=str,
        help='Specify the path to import the model', default='', required=False)
args = argslist.parse_args()

# Launch everything

maxlen_questions = 22
maxlen_answers = 74

if len(args.loadModel) > 0:
	print("Seq2Seq: Loading model from " + args.loadModel + "...")
	encoder_model, decoder_model = load_inference_model(args.loadModel + "/model_enc.h5", args.loadModel + "/model_dec.h5")
	tokenizer = load_tokenizer(args.loadModel + "/tokenizer.pickle")
else:
	if args.downloadData:
		print("Seq2Seq: Downloading data")
		import_data()
	print("Seq2Seq: Preprocessing the data...")
	preprocess_data()
	print("Seq2Seq: Cleaning the data...")
	clean_everything()
	print("Seq2Seq: Loading word2vec model...")
	load_word2vec(args.word2vec_model)
	print("Seq2Seq: Training the tokenizer")
	fit_tokenizer()
	print("Seq2Seq: Filling the vocab list...")
	fill_vocab()
	print("Seq2Seq: Creating the embedding matrix...")
	unk_words = replace_unknown_words()
	create_embedding_matrix(unk_words)
	print("Seq2Seq: Training model...")
	encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs = train()
	if len(args.saveModel) > 0:
		print("Seq2Seq: Saving model to " + args.saveModel)
		save_inference_model(args.saveModel, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)
		save_tokenizer(args.saveModel)
	if args.speak:
		encoder_model, decoder_model = make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)

if args.speak:
	print("Seq2Seq: Ready for questions")
	ask_questions(encoder_model, decoder_model)
