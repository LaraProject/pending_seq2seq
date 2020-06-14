import numpy as np
import tensorflow as tf
import pickle

# Save the inference model
def save_inference_model(path, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs):
	(encoder_model, decoder_model) = make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)
	encoder_model.save(path + '/model_enc.h5')
	decoder_model.save(path + '/model_dec.h5')

# Save the tokenizer
def save_tokenizer(path):
	with open(path + '/tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=3)

def save_length(path):
	data = str(maxlen_questions) + "," + str(maxlen_answers)
	with open(path + "/length.txt", "w") as f:
		f.write(data)

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

def load_length(length_file):
	with open(length_file, "r") as f:
		data = ((f.read()).split(","))
	return int(data[0]), int(data[1])