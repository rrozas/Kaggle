import nltk
import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from nltk.corpus import stopwords
from text_processing import text_normalization_and_tokenization
from text_processing import replace_special_char
from quadratic_weighted_kappa import quadratic_weighted_kappa
from sklearn.linear_model import SGDClassifier


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
                          ('QueryTokens', 'query_tokens', SimpleTransform()),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('Token_number',       'query_lens',       SimpleTransform()),
                          ('Token_number',       'title_lens',       SimpleTransform()),
                          ('Token_number',       'description_lens',       SimpleTransform()),

                          ])


trans_table = ''.join( [chr(i) for i in range(128)] + [' '] * 128 )

def remove_specials_chars(input_str):
    return input_str.translate( trans_table )

def preprocessing(string):
    import re
    out = string
    garbage = [r'</?\w+>', r"<.*?>", "http", "www", "img", "border", "style", "px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    for i in garbage:
        regexp = re.compile(i)
        out = regexp.sub("",out)
    return out

def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    data["query_tokens"] = 0.0
    data["query_lens"] = 0
    data["title_lens"] = 0
    data["description_lens"] = 0

    for i, row in data.iterrows():
        query = set(remove_specials_chars(x.lower()) for x in token_pattern.findall(row["query"]))
        title = set(remove_specials_chars(x.lower()) for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        data.set_value(i, "query_lens", len(query))
        data.set_value(i, "title_lens", len(title))        
        data.set_value(i, "description_lens", len(description))
        if len(query) > 0:
            data.set_value(i, "query_tokens", len(query.intersection(title))*1./len(query))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", len(query.intersection(title))*1./len(title))
        if len(title) == 0 & len(description) > 0:
            data.set_value(i, "query_tokens_in_description", len(query.intersection(description))*1./len(description))

train.product_description = train.product_description.apply(preprocessing)
train.product_description = train.product_description.apply(remove_specials_chars)

test.product_description = test.product_description.apply(preprocessing)
test.product_description = test.product_description.apply(remove_specials_chars)

extract_features(train)
extract_features(test)

#clf = SGDClassifier(loss = 'log',alpha=.0001, n_iter=50, penalty='elasticnet' )
clf = RandomForestClassifier(n_estimators=200,
                                                       n_jobs=4,
                                                        min_samples_split=2,
                                                        random_state=1)

pipeline = Pipeline([("extract_features", features),
                     ("classify", clf)])



def scorer(estimator, X, y):
   return quadratic_weighted_kappa( y, estimator.predict(X) )

from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

skf = StratifiedKFold(train["median_relevance"], 5)
i = 0
y = train["median_relevance"]
y2 = train["relevance_variance"]
X = train.loc
for train_index, test_index in skf:
    print 'fold' , i
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print metrics.classification_report( y_test , y_pred )
    print quadratic_weighted_kappa( y_test , y_pred )
    print metrics.confusion_matrix( y_test , y_pred )
    print
    i += 1

#scores = cross_val_score( pipeline , train , train['median_relevance'] , cv=5 , scoring = scorer )
#print scores , scores.mean() , scores.std()

pipeline.fit(train, train["median_relevance"])

predictions = pipeline.predict(test)

submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("python_benchmark.csv", index=False)
