
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, metrics, grid_search
from sklearn.pipeline import Pipeline
import re

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

class FeatureMapper(BaseEstimator):
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


trans_table = ''.join( [chr(i) for i in range(128)] + [' '] * 128 )

def remove_specials_chars(input_str):
    return input_str.translate( trans_table )

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




def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('../data/train.csv').fillna("")
    test = pd.read_csv('../data/test.csv').fillna("")

    def preprocessing(string):
        import re
        out = string
        garbage = [r'</?\w+>']
         #r"<.*?>", "http", "www", "img", "border", "style", "px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
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
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    # do some lambda magic on text columns
    train['query_title'] = (train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    test['query_title'] = (test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    

    #                          Feature Set Name            Data Frame Column              Transformer
    features = FeatureMapper([
        #('QueryBagOfWords',          'query',                       CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
         #                 ('TitleBagOfWords',          'product_title',               CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
         #                 ('DescriptionBagOfWords',    'product_description',         CountVectorizer(max_features=200, strip_accents = 'ascii', stop_words = stopwords.words('english'))),
                          ('query_title',    'query_title',         tfv),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokens', 'query_tokens', SimpleTransform()),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('Token_number',       'query_lens',       SimpleTransform()),
                          ('Token_number',       'title_lens',       SimpleTransform()),
                          ('Token_number',       'description_lens',       SimpleTransform()),

                          ])

    # Fit TFIDF
    #tfv.fit(traindata)
    #X =  tfv.transform(traindata) 
    #X_test = tfv.transform(testdata)
    
    # Initialize SVD
    svd = TruncatedSVD()
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC()
    
    # Create the pipeline 
    clf = Pipeline([("extractFeatures", features),
                             ('svd', svd),
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [200, 400],
                  'svm__C': [10, 12]}
    
    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=1, iid=True, refit=True, cv=2)
                                     
    # Fit Grid Search Model
    model.fit(train, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(train,y)
    preds = best_model.predict(test)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("beating_the_benchmark_yet_again.csv", index=False)
