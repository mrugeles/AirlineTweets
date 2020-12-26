import re

import nltk
import stanza
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

class NLPUtils():

    def __init__(self):
        print('Init NLPUtils')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        stanza.download('es')
        self.nlp = stanza.Pipeline('es')
        self.pattern = re.compile(r'(\#\w+)')
        self.tweet_tokenizer = TweetTokenizer()

    def tokenize(self, text):
        """Text tokenization
        Parameters
        ----------
        text: string
            Text to tokenize
        Returns
        -------
        text: Tokenized text.
        """
        text = text.lower()
        tokens = self.tweet_tokenizer.tokenize(text)
        tokens = [token for token in tokens if 'http' not in token]
        text = re.sub(r"[^a-záéíóúÁÉÓÚÑñüÜ]", " ", ' '.join(tokens))

        doc = self.nlp(text)

        word_list = [[word.lemma for word in sentence.words] for sentence in doc.sentences]
        word_list = [item for sublist in word_list for item in sublist]
        word_list = [w for w in word_list if w not in stopwords.words("spanish")]
        return list(set(word_list))
