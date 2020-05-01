### How to use ? ###
0. Clone this repository
```
git clone https://github.com/LaraProject/pending_seq2seq -b louis
cd pending_seq2seq
```
1. Setup Python 2.7 environnement and install dependencies
```
$ virtualenv -p /usr/bin/python2.7 venv
$ source venv/bin/activate
$ pip install numpy sklearn tensorflow==1.15 tensorflow-gpu==1.15
```
2. Download NLP
```
git clone https://github.com/LaraProject/nlp
```
3. Download the dataset
```
wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
$ unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_conversations.txt"
$ unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_lines.txt"
$ mv "cornell movie-dialogs corpus/movie_conversations.txt" .
$ mv "cornell movie-dialogs corpus/movie_lines.txt" .
$ rm -rf "cornell movie-dialogs corpus"
```
4. Setup the NLP
Create *src/main/java/org/lara/nlp/Main.java*
```
package org.lara.nlp;

import java.io.File;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.lara.nlp.context.Cornell;
import org.lara.nlp.word2vec.W2v;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

class Main {
	public static void main(String[] args) throws Exception {
		// Context
		Cornell context = new Cornell("movie_lines.txt", "movie_conversations.txt", 0, 20);
		context.init();
		context.cleaning();
		// Word2Vec
		ArrayList<String> allWords = new ArrayList<String>();
		allWords.addAll(context.questions);
		allWords.addAll(context.answers);
		W2v w2v = new W2v(allWords, 5, 1, 3, 100);
		// Export everything
		w2v.exportWords("wordList.txt");
		w2v.exportEmbedding("embeddingMatrix.npy");
		context.exportDictionnary("createDictionnary.py");
	}
}
```
5. Run the NLP
```
cd nlp/
mvn package
java -cp target/laraproject-0.0.1-SNAPSHOT.jar org.lara.nlp.Main
```
6. Create the dictionnary
```
python createDictionnary.py
```
7. Run the Seq2Seq netword
```  
python Seq2Seq.py
```