import os
import sys
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

@dataclass
class DataTransformationConfig:
    # Transformed data aur models ko save karne ke paths
    transformed_train_path=os.path.join("artifacts","transformed_train.csv")
    transformed_test_path=os.path.join("artifacts","transformed_test.csv")
    feature_encoder_path=os.path.join("artifacts","label_encoder.pkl")
    vectorizer_path=os.path.join("artifacts","vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        self.ps=PorterStemmer()

    def transform_text(self,text):
        """
        Main notebook text processing waala logic hi
        """        
        try:
            text=str(text).lower()
            text=nltk.word_tokenize(text)

            #remove special characters
            y=[i for i in text if i.isalnum()]

            #remove stopwords and punctuation
            stop_words=set(stopwords.words('english'))
            y=[i for i in y if i not in stop_words and i not in string.punctuation]

            #stemming
            y=[self.ps.stem(i) for i in y]

            return " ".join(y)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("data transformation has started")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Train and Test data loaded successfully inside Transformation")

            # drop unwanted columns as in notebook
            cols_to_drop=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
            for col in cols_to_drop:
                if col in train_df.columns:
                    train_df.drop(columns=[col], inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col], inplace=True)  


            # renaming columns as in notebook
            if 'v1' in train_df.columns and 'v2' in train_df.columns:

                train_df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

                test_df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

            # handle duplicates and missing values
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)

            train_df.dropna(subset=['target', 'text'], inplace=True)
            test_df.dropna(subset=['target', 'text'], inplace=True)

            # text transformation
            logging.info("applying text transformation(stemming & stopwords) removal")
            train_df['transformed_text']=train_df['text'].apply(self.transform_text)
            test_df['transformed_text']=test_df['text'].apply(self.transform_text)

            #3 label encoding on target
            logging.info("applying label encoding on target column")
            encoder=LabelEncoder()

            # Train par fit aur transform dono karenge
            train_df['target'] = encoder.fit_transform(train_df['target'])
            # Test par sirf transform karenge (Data Leakage se bachne ke liye)
            test_df['target'] = encoder.transform(test_df['target'])


            # save the label encoder object
            os.makedirs(os.path.dirname(self.transformation_config.feature_encoder_path), exist_ok=True)

            with open(self.transformation_config.feature_encoder_path,"wb") as f:
                pickle.dump(encoder,f)
            logging.info("Label Encoder object saved as pickle file")   

            train_df.to_csv(self.transformation_config.transformed_train_path, index=False, header=True)
            test_df.to_csv(self.transformation_config.transformed_test_path, index=False, header=True)
            logging.info("Transformed train and test datasets saved")

            # VECTORIZER ADD KAR RHE HAIN ------
            logging.info("Applying CountVectorizer/TF-IDF")
            from sklearn.feature_extraction.text import CountVectorizer # ya TfidfVectorizer
            
            # Tum Bag of Words use kar rhe ho toh CountVectorizer le lo
            cv = CountVectorizer(max_features=5000) 
            cv.fit(train_df['transformed_text'])

            # Vectorizer object ko save karna
            os.makedirs(os.path.dirname(self.transformation_config.vectorizer_path), exist_ok=True)
            with open(self.transformation_config.vectorizer_path, "wb") as f:
                pickle.dump(cv, f)
            logging.info("Vectorizer object saved as pickle file")

            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.feature_encoder_path,
                self.transformation_config.vectorizer_path
            )
        except Exception as e:
            raise CustomException(e, sys)


