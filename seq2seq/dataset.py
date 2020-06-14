import numpy as np
from tensorflow.keras import preprocessing, utils
import os
import yaml
import requests
import zipfile
import io
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

# Import data
def import_data():
	# Download data set
	r = requests.get('https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip?raw=true')
	z = zipfile.ZipFile(io.BytesIO(r.content))
	z.extractall()

# Import custom data
def use_custom_data(path, size):
	global questions
	global answers
	# Open file
	f = open(path)
	lines = (f.readlines())
	lines = lines[:int(len(lines) * (size/100.))]
	f.close()
	non_tonkenized_answers = []
	for i in range(len(lines)):
		if i % 2 == 0:
			questions.append(lines[i][11:-1])
		else:
			non_tonkenized_answers.append(lines[i][9:-1])

	# Tokenize answers
	answers = []
	for i in range(len(non_tonkenized_answers)):
		answers.append('<start> ' + non_tonkenized_answers[i] + ' <end>')

	# Force length
	length_limit = 25
	new_questions = []
	new_answers = []
	for i in range(min(len(questions), len(answers))):
		if not((len(questions[i].split()) > length_limit) or (len(answers[i].split()) > length_limit+2)):
			new_questions.append(questions[i])
			new_answers.append(answers[i])
	questions = new_questions
	answers = new_answers

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

def load_word2vec(model_path, useFastText):
	global model_w2v
	if useFastText:
		model_w2v = FastText.load(model_path)
	else:
		model_w2v = KeyedVectors.load_word2vec_format(model_path, binary=False)

def fit_tokenizer():
	global tokenizer
	global VOCAB_SIZE
	tokenizer.fit_on_texts(questions + answers)
	VOCAB_SIZE = len(tokenizer.word_index) + 1
	#print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

# Get all words which are in both the model and the dataset
def get_known_words():
	known_words = []
	for word in tokenizer.word_index:
		if word in model_w2v:
			known_words.append(word)
	return known_words + ["<unk>"]

# Make a new tokenizer
def fit_new_tokenizer():
	global tokenizer
	global VOCAB_SIZE
	fit_tokenizer()
	known_words = get_known_words()
	tokenizer_new = preprocessing.text.Tokenizer(oov_token='<unk>', filters='')
	tokenizer_new.fit_on_texts([known_words])
	tokenizer = tokenizer_new
	VOCAB_SIZE = len(tokenizer.word_index) + 1 + 1
	#print('VOCAB SIZE : {}'.format(VOCAB_SIZE))


# Create the embedding matrix
def create_embedding_matrix():
	global embedding_matrix
	embedding_matrix = np.zeros((VOCAB_SIZE, vectors_size))
	for word, i in tokenizer.word_index.items():
		embedding_matrix[i] = model_w2v[word]

# Create input and output datasets
# encoder_input_data
def create_input_output():
	global maxlen_answers
	global maxlen_questions
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
	decoder_output_data = np.array(padded_answers)
	#print(decoder_output_data.shape)

	return encoder_input_data, decoder_input_data, decoder_output_data