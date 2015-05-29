import nltk
import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from text_processing import text_normalization_and_tokenization
from text_processing import replace_special_char

train = pd.read_csv("../data/train.csv").fillna("")
test  = pd.read_csv("../data/test.csv").fillna("")

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryBagOfWords',          'query',                       CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
                          ('TitleBagOfWords',          'product_title',               CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
                          ('DescriptionBagOfWords',    'product_description',         CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokensInDescription', 'query_tokens_in_description', SimpleTransform())])


trans_table = ''.join( [chr(i) for i in range(128)] + [' '] * 128 )

def remove_specials_chars(input_str):
    return input_str.translate( trans_table )

def preprocessing(string):
    import re
    html_balise = re.compile(r'</?\w+>')
    return html_balise.sub("",string)

def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    for i, row in data.iterrows():
        query = set(remove_specials_chars(x.lower()) for x in token_pattern.findall(row["query"]))
        title = set(remove_specials_chars(x.lower()) for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", len(query.intersection(title))*1./len(title))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", len(query.intersection(description))*1./len(description))

extract_features(train)
extract_features(test)

pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=200,
                                                         n_jobs=4,
                                                         min_samples_split=2,
                                                         random_state=1))])

train.product_description = train.product_description.apply(preprocessing)
train.product_description = train.product_description.apply(remove_specials_chars)

pipeline.fit(train, train["median_relevance"])

test.product_description = test.product_description.apply(preprocessing)
test.product_description = test.product_description.apply(remove_specials_chars)



predictions = pipeline.predict(test)

submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("python_benchmark.csv", index=False)
