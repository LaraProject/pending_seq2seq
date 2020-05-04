### How to use ? ###

**0. Clone this repository**

```
$ git clone https://github.com/LaraProject/pending_seq2seq -b louis
$ cd pending_seq2seq
```

**1. Setup Python 3.7 environnement and install dependencies**

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install tensorflow pandas nltk gensim
```

**2. Download NLP**

```
git clone https://github.com/LaraProject/nlp
```

**3. Download the dataset**

```
$ wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
$ unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_conversations.txt"
$ unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_lines.txt"
$ mv "cornell movie-dialogs corpus/movie_conversations.txt" .
$ mv "cornell movie-dialogs corpus/movie_lines.txt" .
$ rm -rf "cornell movie-dialogs corpus"
```

**4. Setup the NLP**

Create *nlp/src/main/java/org/lara/nlp/Main.java*

```
package org.lara.nlp;

import java.util.ArrayList;
import org.lara.nlp.context.Cornell;
import org.lara.nlp.word2vec.W2v;

class Main {
	public static void main(String[] args) throws Exception {
		// Cornell data
		Cornell context = new Cornell("../movie_lines.txt", "../movie_conversations.txt", 2, 20);
		context.init();
		context.lengthFilter();
		// Export before cleaning
		context.exportData("../data.txt");
		context.cleaning();
		// Word2Vec
		ArrayList<String> allWords = new ArrayList<String>();
		allWords.addAll(context.questions);
		allWords.addAll(context.answers);
		allWords.add("_U_");
		allWords.add("_U_");
		allWords.add("_U_");
		allWords.add("_U_");
		allWords.add("_U_");
		W2v w2v = new W2v(allWords, 5, 1, 3, 100);
		w2v.write_vectors("../word2vec_movies_format.txt");
	}
}
```

**5. Run the NLP**

```
$ cd nlp/
$ mvn package
$ java -cp target/laraproject-0.0.1-SNAPSHOT.jar org.lara.nlp.Main
$ ./gensim_convert.sh ../word2vec_movies_format.txt
$ cd ..
```

**6. Run the RNN**

*6.1 Adjust the dataset length*

You can adjust the dataset length at line <u>22</u> of <u>script.py</u>

*6.2 Adjust the learning rate*

You can adjust the learning rate at line <u>167</u> and <u>263</u> of <u>script.py</u> (they must be the same).

*6.3 Train*

You can train the RNN by uncommenting lines <u>210</u> to <u>228</u>.

*6.4 Experiment*

You can experiment by uncommenting from line <u>231</u> to the end.




































