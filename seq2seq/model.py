import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, models, preprocessing
from tensorflow.keras import preprocessing, utils

# Defining the Encoder-Decoder model
def create_model(encoder_input_data, decoder_input_data, decoder_output_data, use_spatial_dropout=False, use_reccurent_dropout=False, use_batch_normalisation=False):
	encoder_inputs = tf.keras.layers.Input(shape=(None, ))
	encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, vectors_size,
			weights=[embedding_matrix], trainable=False)(encoder_inputs)
	if use_batch_normalisation:
		encoder_embedding = tf.keras.layers.BatchNormalization()(encoder_embedding)
	if use_spatial_dropout:
		encoder_embedding = tf.keras.layers.SpatialDropout1D(0.2)(encoder_embedding)
	(encoder_outputs, state_h, state_c) = tf.keras.layers.LSTM(vectors_size,
			return_state=True)(encoder_embedding)
	encoder_states = [state_h, state_c]

	decoder_inputs = tf.keras.layers.Input(shape=(None, ))
	decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, vectors_size,
			weights=[embedding_matrix], trainable=False)(decoder_inputs)
	if use_batch_normalisation:
		decoder_embedding = tf.keras.layers.BatchNormalization()(decoder_embedding)
	if use_spatial_dropout:
		decoder_embedding = tf.keras.layers.SpatialDropout1D(0.2)(decoder_embedding)
	if use_reccurent_dropout:
		decoder_lstm = tf.keras.layers.LSTM(vectors_size, return_state=True,
									return_sequences=True, recurrent_dropout=0.2)
	else:
		decoder_lstm = tf.keras.layers.LSTM(vectors_size, return_state=True,
											return_sequences=True)
	(decoder_outputs, _, _) = decoder_lstm(decoder_embedding,
			initial_state=encoder_states)
	decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE,
			activation=tf.keras.activations.softmax)
	output = decoder_dense(decoder_outputs)

	model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
	model.compile(optimizer=tf.keras.optimizers.Nadam(),
				  loss='sparse_categorical_crossentropy')
	model.summary()
	return model, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs

# Training the model
def train(use_spatial_dropout, use_reccurent_dropout, use_batch_normalisation):
	encoder_input_data, decoder_input_data, decoder_output_data = create_input_output()
	model, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs = create_model(encoder_input_data, decoder_input_data, decoder_output_data, use_spatial_dropout, use_reccurent_dropout, use_batch_normalisation)
	model.fit([encoder_input_data, decoder_input_data],
			  decoder_output_data, batch_size=512, epochs=500)
	return encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs

# Defining inference models
def make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs):
	encoder_model = tf.keras.models.Model(encoder_inputs,
			encoder_states)
	decoder_state_input_h = tf.keras.layers.Input(shape=(vectors_size, ))
	decoder_state_input_c = tf.keras.layers.Input(shape=(vectors_size, ))
	decoder_states_inputs = [decoder_state_input_h,
							 decoder_state_input_c]
	(decoder_outputs, state_h, state_c) = decoder_lstm(decoder_embedding,
					 initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = tf.keras.models.Model([decoder_inputs]
			+ decoder_states_inputs, [decoder_outputs] + decoder_states)
	return (encoder_model, decoder_model)

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

			if sampled_word == '<end>' or len(decoded_translation.split()) > maxlen_answers:
				stop_condition = True

			empty_target_seq = np.zeros((1, 1))
			empty_target_seq[0, 0] = sampled_word_index
			states_values = [h, c]

		print(decoded_translation[:-5])  # remove end w