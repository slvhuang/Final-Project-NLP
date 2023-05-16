import spacy
import nltk

nlp = spacy.load('en_core_web_lg')

def process_text(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def process_text_sen(text):
    sents = nltk.sent_tokenize(text)
    ents = []
    for sent in sents:
        doc = nlp(sent)
        ents.extend([(ent.text, ent.label_) for ent in doc.ents])
    return ents