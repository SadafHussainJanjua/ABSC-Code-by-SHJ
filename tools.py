#!/usr/bin/env python3
"""Tools for Aspect-Based Hybrid Tweet Classifier.
"""
import re
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



def clean_text(text: str, stem = True) -> str:
    """Clear text."""

    text = re.sub(r"\W", " ", str(text))

    # Remove all single characters.
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)

    # Remove single characters from the start.
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text) 

    # Substituting multiple spaces with single space.
    text = re.sub(r"\s+", " ", text, flags = re.I)

    # Removing prefixed "b"
    text = re.sub(r"^b\s+", "", text)

    # Converting to lowercase.
    text = text.lower()

    # Remove punctuation marks.
    for ch in string.punctuation:
        text = text.replace(ch, "")

    # Remove stopwords.
    text = [word for word in text.split() 
            if word not in stopwords.words("english")]
    text = " ".join(text)
    
    # Lemmatization.
    if stem:
        stemmer = WordNetLemmatizer()
        text    = text.split()
        text    = [stemmer.lemmatize(word) for word in text]
        text    = " ".join(text)

    return text


def tokenizer(text, stem = True):
    """Tokenize text."""

    text   = clean_text(text, stem)
    tokens = word_tokenize(text)

    return tokens


if __name__ == '__main__':
    # Testing ...
    pass