import nltk

text = "Where are you going"

words = nltk.word_tokenize(text)

tags = nltk.pos_tag(words)
print(tags)

for word in tags:
    word = list(word)
    print(nltk.help.upenn_tagset(word[1]))

