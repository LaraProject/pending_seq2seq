import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

# Import custom data
def use_custom_data(path, size):
	# Open file
	f = open(path)
	lines = (f.readlines())
	lines = lines[:int(len(lines) * (size/100.))]
	f.close()

	# Get questions and answers
	questions = []
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

	return new_questions, new_answers

# Load the Word2Vec model
def load_word2vec(model_path, useFastText):
	if useFastText:
		return FastText.load(model_path)
	else:
		return KeyedVectors.load_word2vec_format(model_path, binary=False)

# Get all words which are in both the model and the dataset
def get_known_words(tokenizer, model_w2v):
	known_words = []
	for word in tokenizer.word_index:
		if word in model_w2v:
			known_words.append(word)
	return known_words + ["<unk>"]

def fit_tokenizer(data, model_w2v):
	# Fit first tokenizer
	tokenizer = preprocessing.text.Tokenizer(oov_token='<unk>', filters='')
	tokenizer.fit_on_texts(data)
	# Get the known words
	known_words = get_known_words(tokenizer, model_w2v)
	# Fit a new tokenizer
	tokenizer = preprocessing.text.Tokenizer(oov_token='<unk>', filters='')
	tokenizer.fit_on_texts([known_words])
	vocab_size = len(tokenizer.word_index) + 1 + 1
	return tokenizer, vocab_size

# Create the embedding matrix
def create_embedding_matrix(tokenizer, model_w2v, vocab_size, vectors_size):
	embedding_matrix = tf.zeros((vocab_size, vectors_size))
	for word, i in tokenizer.word_index.items():
		embedding_matrix[i] = model_w2v[word]
	return embedding_matrix

# Create tokenized data
def tokenize(data, model_w2v):
	tokenizer, vocab_size = fit_tokenizer(data, model_w2v)
	tensor = lang_tokenizer.texts_to_sequences(data)
	tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
	return tensor, tokenizer, vocab_size

# Load the dataset
def load_dataset(path, model_w2v, vectors_size, size=100):
	input_data, target_data = use_custom_data(path, size)
	input_tensor, input_tokenizer, input_vocab_size = tokenize(input_data)
	target_tensor, target_tokenizer, target_vocab_size = tokenize(target_data)
	input_embedding_matrix = create_embedding_matrix(input_tokenizer, model_w2v, input_vocab_size, vectors_size)
	output_embedding_matrix = create_embedding_matrix(output_tokenizer, model_w2v, output_vocab_size, vectors_size)
	return input_tensor, target_tensor, input_tokenizer, target_tokenizer, input_embedding_matrix, output_embedding_matrix

# Create a TF Dataset
def create_tf_dataset(input_tensor_train, target_tensor_train, batch_size):
	buffer_size = len(input_tensor_train)
	dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
	dataset = dataset.batch(batch_size, drop_remainder=True)
	return dataset