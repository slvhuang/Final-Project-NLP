import spacy
import nltk
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


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


def process_text_sen_sentiment(text):
    sents = nltk.sent_tokenize(text)
    ents = []
    for sent in sents:

        # add sentiment
        blob = TextBlob(sent)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # ner
        doc = nlp(sent)
        ents.extend([(ent.text, ent.label_, polarity, subjectivity) for ent in doc.ents])

    return ents
