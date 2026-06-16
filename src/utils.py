import os
import sys
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from src.exception import CustomException

ps = PorterStemmer()


def save_object(file_path, obj):
    """
    Save any python object as pickle file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load pickle object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def transform_text(text):
    """
    Text preprocessing for spam classification
    """
    try:
        text = str(text).lower()

        text = nltk.word_tokenize(text)

        # remove special characters
        y = [i for i in text if i.isalnum()]

        # remove stopwords
        stop_words = set(stopwords.words("english"))
        y = [
            i
            for i in y
            if i not in stop_words
            and i not in string.punctuation
        ]

        # stemming
        y = [ps.stem(i) for i in y]

        return " ".join(y)

    except Exception as e:
        raise CustomException(e, sys)