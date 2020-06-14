import tensorflow as tf
import os
import time

# Encoder layer
class Encoder(tf.keras.Model):
	def __init__(self, embedding_matrix, enc_units, batch_sz,use_batch_normalisation, use_spatial_dropout):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
										weights=[embedding_matrix],
										trainable=False)
		self.gru = tf.keras.layers.GRU(self.enc_units,
										return_sequences=True,
										return_state=True,
										recurrent_initializer='glorot_uniform')
		self.use_batch_normalisation = use_batch_normalisation
		self.use_spatial_dropout = use_spatial_dropout

	def call(self, x, hidden):
		x = self.embedding(x)
		# Use BatchNormalisation
		if self.use_batch_normalisation:
			x = tf.keras.layers.BatchNormalization()(x)
		# Use spatial dropout
		if self.use_spatial_dropout:
			x = tf.keras.layers.SpatialDropout1D(0.2)(x)
		output, state = self.gru(x, initial_state = hidden)
		return output, state

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.enc_units))

# Attention Seq2Seq lay
class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, query, values):
		# query hidden state shape == (batch_size, hidden size)
		# query_with_time_axis shape == (batch_size, 1, hidden size)
		# values shape == (batch_size, max_len, hidden size)
		# we are doing this to broadcast addition along the time axis to calculate the score
		query_with_time_axis = tf.expand_dims(query, 1)

		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		# the shape of the tensor before applying self.V is (batch_size, max_length, units)
		score = self.V(tf.nn.tanh(
				self.W1(query_with_time_axis) + self.W2(values)))

		# attention_weights shape == (batch_size, max_length, 1)
		attention_weights = tf.nn.softmax(score, axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights

# Decoder layer
class Decoder(tf.keras.Model):
	def __init__(self, embedding_matrix, dec_units, batch_sz, use_batch_normalisation, use_spatial_dropout):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
										weights=[embedding_matrix],
										trainable=False)
		self.gru = tf.keras.layers.GRU(self.dec_units,
										return_sequences=True,
										return_state=True,
										recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(embedding_matrix.shape[0])
		self.use_batch_normalisation = use_batch_normalisation
		self.use_spatial_dropout = use_spatial_dropout

		# used for attention
		self.attention = BahdanauAttention(self.dec_units)

	def call(self, x, hidden, enc_output):
		# enc_output shape == (batch_size, max_length, hidden_size)
		context_vector, attention_weights = self.attention(hidden, enc_output)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# Use BatchNormalisation
		if self.use_batch_normalisation:
			x = tf.keras.layers.BatchNormalization()(x)
		if self.use_spatial_dropout:
			x = tf.keras.layers.SpatialDropout1D(0.2)(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# output shape == (batch_size * 1, hidden_size)
		output = tf.reshape(output, (-1, output.shape[2]))

		# output shape == (batch_size, vocab)
		x = self.fc(output)

		return x, state, attention_weights

# Create model
def create_model(embedding_matrix_input, embedding_matrix_output, units, batch_size, use_batch_normalisation, use_spatial_dropout):
	# Encoder
	encoder = Encoder(embedding_matrix_input, units, batch_size, use_batch_normalisation, use_spatial_dropout)
	# Decoder
	decoder = Decoder(embedding_matrix_output, units, batch_size, use_batch_normalisation, use_spatial_dropout)

	return encoder, decoder

# Custom loss function
def loss_function(real, pred, loss_object):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tf.reduce_mean(loss_)

# Checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
def get_checkpoint(encoder, decoder, optimizer):
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
									encoder=encoder,
									decoder=decoder)
	return checkpoint

# Training step
@tf.function
def train_step(encoder, decoder, inp, targ, targ_tokenizer, enc_hidden, batch_size, loss_object, optimizer):
	loss = 0

	with tf.GradientTape() as tape:
		enc_output, enc_hidden = encoder(inp, enc_hidden)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * batch_size, 1)

		# Teacher forcing - feeding the target as the next input
		for t in range(1, targ.shape[1]):
			# passing enc_output to the decoder
			predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
			loss += loss_function(targ[:, t], predictions, loss_object)

			# using teacher forcing
			dec_input = tf.expand_dims(targ[:, t], 1)

	batch_loss = (loss / int(targ.shape[1]))
	variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(gradients, variables))

	return batch_loss

# Main training fonction
@tf.function
def train(embedding_matrix_input, embedding_matrix_output, targ_tokenizer, epochs, units, batch_size, dataset, checkpoint, use_batch_normalisation, use_spatial_dropout):
	# Optimizer
	optimizer = tf.keras.optimizers.Adam()
	# Loss object
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits=True, reduction='none')
	# Encoder/Decoder
	encoder, decoder = create_model(embedding_matrix_input, embedding_matrix_output, units, batch_size, use_batch_normalisation, use_spatial_dropout)

	for epoch in range(epochs):
		start = time.time()

		enc_hidden = encoder.initialize_hidden_state()
		total_loss = 0

		for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
			batch_loss = train_step(inp, targ, targ_tokenizer, enc_hidden, batch_size, loss_object, optimizer)
			total_loss += batch_loss

			if batch % 100 == 0:
				print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
															batch,
															batch_loss.numpy()))
		# saving (checkpoint) the model every 2 epochs
		if (epoch + 1) % 2 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print('Epoch {} Loss {:.4f}'.format(epoch + 1,
											total_loss / steps_per_epoch))
		print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# Get the output of the neural network
@tf.function
def evaluate(sentence, encoder, decoder, units, max_length_inp, targ_tokenizer):
	sentence = "<start>" + sentence + "<end>"

	inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
															maxlen=max_length_inp,
															padding='post')
	inputs = tf.convert_to_tensor(inputs)

	result = ''

	hidden = [tf.zeros((1, units))]
	enc_out, enc_hidden = encoder(inputs, hidden)

	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)

	for t in range(max_length_targ):
		# Training step
		predictions, dec_hidden, attention_weights = decoder(dec_input,
															dec_hidden,
															enc_out)
		predicted_id = tf.argmax(predictions[0]).numpy()

		result += targ_tokenizer.index_word[predicted_id] + ' '

		if targ_tokenizer.index_word[predicted_id] == '<end>':
			return result

		# the predicted ID is fed back into the model
		dec_input = tf.expand_dims([predicted_id], 0)

	return result

# Get an answer from a question
def getAnswer(question, encoder, decoder, units, max_length_inp, targ_tokenizer):
  result = evaluate(question, encoder, decoder, units, max_length_inp, targ_tokenizer)
  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))
