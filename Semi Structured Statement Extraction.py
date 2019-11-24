import spacy
import textacy.extract
nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = """London is the capital and most populous city of England and  the United Kingdom.  
Standing on the River Thames in the south east of the island of Great Britain, 
London has been a major settlement  for two millennia.  It was founded by the Romans, 
who named it Londinium.
"""
#parcing

doc = nlp(text)

#extracting
statements = textacy.extract.semistructured_statements(doc,"London")

print("Things I know about london are:")

for statement in statements:
    subject,verb,fact = statement
    print(f" - {fact}")