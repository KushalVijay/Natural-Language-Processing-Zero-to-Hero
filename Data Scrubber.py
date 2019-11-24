import spacy

nlp = spacy.load('en_core_web_lg')

print("here")
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDUCTED]"
    else:
        return token.string

def scrub(text):
    doc = nlp(text) 
    for ent in doc.ents:
        ent.merge()
    tokens = map(replace_name_with_placeholder,doc)

    return "".join(tokens)

s = """In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s 
Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
"""

print(scrub(s))