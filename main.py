import argparse
import seq2seq.dataset
import seq2seq.model
#import seq2seq.save
from tensorflow.keras import preprocessing

'''
# Make a TensorFlow Tokenizer
tokenizer = preprocessing.text.Tokenizer(filters='')
questions = []
answers = []
model_w2v = None
embedding_matrix = None
maxlen_questions = 0
maxlen_answers = 0
VOCAB_SIZE = 0
vectors_size = 100

# Argument management
argslist = argparse.ArgumentParser(description="Seq2Seq Neural Network")

argslist.add_argument('word2vec_model', metavar='word2vec_model', type=str,
		help='Path to the word2vec model')
argslist.add_argument('--useFastText', metavar='[True/False]', type=bool,
        help='Specify whether to use Facebook FastText model', default=False, required=True)
argslist.add_argument('--downloadData', metavar='[True/False]', type=bool,
        help='Specify whether the dataset should be downloaded', default=False, required=False)
argslist.add_argument('--customData', metavar='path', type=str,
        help='Specify the path to the custom dataset', default='', required=True)
argslist.add_argument('--speak', metavar='[True/False]', type=bool,
        help='Specify whether to speak with the Network', default=False, required=False)
argslist.add_argument('--saveModel', metavar='path', type=str,
        help='Specify the path where to save the model', default='', required=False)
argslist.add_argument('--loadModel', metavar='path', type=str,
        help='Specify the path to import the model', default='', required=False)
argslist.add_argument('--useSpatialDropout', metavar='[True/False]', type=bool,
        help='Specify whether to use 1D spatial dropout after the embedding layers', default=False, required=False)
argslist.add_argument('--useReccurentDropout', metavar='[True/False]', type=bool,
        help='Specify whether to use a recurrent dropout in the LSTM', default=False, required=False)
argslist.add_argument('--useBatchNormalisation', metavar='[True/False]', type=bool,
        help='Specify whether to use batch normalisation', default=True, required=False)
argslist.add_argument('--vectorSize', metavar='size', type=int,
        help='Specify the size of the word vectors', default=100, required=False)
argslist.add_argument('--dataSize', metavar='size', type=int,
        help='Specify the percentage of data to use', default=100, required=False)
args = argslist.parse_args()

# Launch everything

vectors_size = args.vectorSize

if len(args.loadModel) > 0:
	print("Seq2Seq: Loading model from " + args.loadModel + "...")
	encoder_model, decoder_model = load_inference_model(args.loadModel + "/model_enc.h5", args.loadModel + "/model_dec.h5")
	tokenizer = load_tokenizer(args.loadModel + "/tokenizer.pickle")
	maxlen_questions, maxlen_answers = load_tokenizer(args.loadModel + "/length.txt")
else:
	print("Seq2Seq: Using custom dataset from " + args.customData)
	use_custom_data(args.customData, args.dataSize)
	print("Seq2Seq: Loading word2vec model...")
	load_word2vec(args.word2vec_model, args.useFastText)
	print("Seq2Seq: Training the tokenizer")
	fit_new_tokenizer()
	print("Seq2Seq: Creating the embedding matrix...")
	create_embedding_matrix()
	print("Seq2Seq: Training model...")
	encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs = train(args.useSpatialDropout, args.useSpatialDropout, args.useBatchNormalisation)
	if len(args.saveModel) > 0:
		print("Seq2Seq: Saving model to " + args.saveModel)
		save_inference_model(args.saveModel, encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)
		save_tokenizer(args.saveModel)
		save_length(args.saveModel)
	if args.speak:
		encoder_model, decoder_model = make_inference_models(encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense, decoder_inputs)

if args.speak:
	print("Seq2Seq: Ready for questions")
	ask_questions(encoder_model, decoder_model)
'''