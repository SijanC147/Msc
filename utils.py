import spacy
import nltk

def tokenize_phrase(phrase, backend='spacy'):
    if backend=='nltk':
        return(nltk.word_tokenize(phrase))
    elif backend=='spacy':
        tokens_list = []
        nlp = spacy.load('en')
        tokens = nlp(phrase)
        for token in tokens:
            # ignore pescy unicode character that appears after certain emoji
            if token.text!='\uFE0F': 
                tokens_list.append(token.text)
        return tokens_list