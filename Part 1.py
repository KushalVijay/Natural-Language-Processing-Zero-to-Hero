import spacy

nlp  = spacy.load('en_core_web_lg')

# text to be examine
text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""
doc = nlp(text)

for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
