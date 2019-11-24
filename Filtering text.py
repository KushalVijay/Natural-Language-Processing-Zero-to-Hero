#Filtering a text: This program computes the vocabulary of a text, then removes all items
#that occur in an existing wordlist, leaving just the uncommon or misspelled words
import nltk
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    print(sorted(unusual))

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
