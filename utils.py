import spacy
import nltk
import numpy as np

def tokenize_phrase(phrase, backend='spacy'):
    if backend=='nltk':
        return(nltk.word_tokenize(phrase))
    elif backend=='spacy':
        tokens_list = []
        nlp = spacy.load('en')
        tokens = nlp(str(phrase))
        for token in tokens:
            # ignore pescy unicode character that appears after certain emoji
            if token.text!='\uFE0F': 
                tokens_list.append(token.text)
        return tokens_list

# def map_to_ids(sentence, target):
    