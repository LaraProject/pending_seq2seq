import numpy as np
from tensorflow.keras import preprocessing, utils
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

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