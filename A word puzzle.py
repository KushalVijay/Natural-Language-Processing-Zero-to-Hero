import nltk
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
print([w for w in wordlist if len(w)>=6 and obligatory in w and nltk.FreqDist(w) <=puzzle_letters])