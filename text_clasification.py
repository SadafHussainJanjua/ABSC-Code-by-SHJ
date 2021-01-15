#!/usr/bin/env python3
"""* MULTI-LEVEL ASPECT BASED TEXT CLASSIFICATION USING HYBRID APPROACH"""
import os
import pandas as pd
import random
import statistics
import timeit
import itertools
from apyori import apriori
import pickle
from tools import tokenizer, clean_text
from textblob import TextBlob
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from config import DATABASE_FOLDER, DATABASE_FILE, FILE_JAR, FILE_MODELS_JAR, \
                   TEST_SIZE, RANDOM_STATE
import warnings
warnings.filterwarnings("ignore")
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
coreNLP_path = "C:/Users/Sadaf/Desktop/New folder2stanford-corenlp-3.9.2.jar"
os.environ["CORENLP_HOME"] = coreNLP_path
#TARGET           = ["Sentiment_4"]
TARGET           = ["Sentiment_positive"]
#TARGET           = ["Sentiment_positive"]
FEATURES         = list()
EXPLICIT_ASPECTS = set()
IMPLICIT_ASPECTS = set()

DATASET = DATABASE_FOLDER + DATABASE_FILE
data = pd.read_csv(DATASET)
print(data.head(10))
data    = data[data["Sentiment"].isin(["positive", "negative"])]
data = data.drop(["TweetId", "TweetDate"], axis=1)
data = data.reset_index(drop=True)



def tweets_text(data):
    """Object generator from dataset tweets.
    """
    rows = data.shape[0]
    for row in range(rows):
        yield data.loc[row, "TweetText"]


def update_aspect(data, aspect, row, sentiment):
    """Update an aspect on data to sentiment.
    """
    if aspect not in data.columns:
        data[aspect] = 0.0
    data.loc[row, aspect] = sentiment


def get_sentiment(tweet_text: str) -> float:
    """Returns the sentiment of an aspect.
    """
    obj       = TextBlob(tweet_text)
    sentiment = obj.sentiment.polarity
    return sentiment


def impute_explicit_aspects(data):
    explicit_aspects = set()
    transactions = list()

    for tweet in tweets_text(data):
        transactions.append(tokenizer(tweet))
        '''
        f = open('tokens-1', 'wb')
        pickle.dump(transactions, f)
        f.close()
        '''

    # print("Printing dataset.............!!!!!!")
    # print(transactions[0:20])

    association_rules = apriori(transactions, min_support=0.01)
    unique_topics = data['Topic'].unique()

    print("Printing association rules......!!!!")
    print("Looping over!")
    for item in association_rules:
        pair = list(item[0])
        for aspect in pair:
            if aspect in unique_topics:
                explicit_aspects.add(aspect)

    print(explicit_aspects)
    # explicit_aspects -= set(["co", "http", "rt"])
    # print(explicit_aspects)

    rows = data.shape[0]
    for row in range(rows):
        tweet_tokens = tokenizer(data.loc[row, "TweetText"])
        for aspect in explicit_aspects:
            if aspect in tweet_tokens:
                tweet_text = " ".join(tweet_tokens)
                sentiment = get_sentiment(tweet_text)
                update_aspect(data, aspect, row, sentiment)

    return explicit_aspects


def impute_implicit_aspects(data):
    """Imputes implicit aspects of the dataset.
    """
    implicit_aspects = set()

    parser = CoreNLPDependencyParser(url='http://localhost:9000')

    rows = data.shape[0]
    for row in range(rows):
        tweet   = data.loc[row, "TweetText"]
        results = parser.raw_parse(tweet)
        words   = next(results).triples()
        aspects = set()
        for word_1, _, word_2 in words:
            aspects.add(word_1[0])
            aspects.add(word_2[0])
        aspect     = " ".join(aspects)
        aspect     = clean_text(aspect)
        print("  - {}".format(aspect))
        implicit_aspects.add(aspect)
        sentiment  = get_sentiment(aspect)
        update_aspect(data, aspect, row, sentiment)

    return implicit_aspects


def data_preprocessing(data):
    """Data pre-processing: nominal variables to numbers (Topic and Sentiment)
    and elimination of string variables (TweetText)
    """

    topics     = pd.get_dummies(data["Topic"],  prefix = "Topic",drop_first = True)
    sentiments = pd.get_dummies(data["Sentiment"],  prefix = "Sentiment",drop_first = True)
    data       = data.drop(["Topic", "Sentiment", "TweetText"], axis = 1)

    data       = pd.concat([topics, sentiments, data], axis = 1)

    return data


def impute_overall_sentiment(data):
    """Calculates and imputes the overall sentiment index of the text
    according to aspects.
    """
  
    data["overall_sentiment"] = pd.np.nan
    #print(list(data.columns))
    aspects                   = EXPLICIT_ASPECTS | IMPLICIT_ASPECTS
    rows                      = data.shape[0]
    for row in range(rows):
        sentiments = list()
        sentiments.append(random.choice([float(data.loc[row, TARGET]), 1]))
        for aspect in aspects:
            sentiments.append(data.loc[row, aspect])

        data.loc[row, "overall_sentiment"] = statistics.mean(sentiments)

    return data


def feature_selection(data):
    """Selects relevant aspects using Principal Component Analysis (PCA).
    """
    #print("Printing data before IG and PCA>>>>>>")
    #pd.set_option('display.expand_frame_repr', False)
    #print(list(data.columns))
    FEATURES             = list(set(data.columns) - set(TARGET))
    pca                  = PCA(n_components = 2)
    X                    = StandardScaler().fit_transform(data[FEATURES])
    principal_components = pca.fit_transform(X)
    pca_data             = pd.DataFrame(data = principal_components,columns = ["principal_component_1","principal_component_2"])
    pca_data = pd.concat([pca_data, data[TARGET]], axis = 1)

    return pca_data



def training_testing_models(pca_data):
    """Train and test models for classification according to aspect-based
    sentiments.
    """

    FEATURES = list(set(pca_data.columns) - set(TARGET))

    X = pca_data[FEATURES]
    y = pca_data[TARGET]

    #print("Printing data after IG and PCA>>>>>>")
    #print(X[:10])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size = TEST_SIZE, random_state = RANDOM_STATE)

    ### Classifiers.
    CLASSIFIERS                               = dict()
    CLASSIFIERS["KNClassifier"]       = KNeighborsClassifier()
    CLASSIFIERS["LR"]         = LogisticRegression()
    CLASSIFIERS["SVC"]                        = SVC()
    CLASSIFIERS["DTClassifier"]     = DecisionTreeClassifier()
    CLASSIFIERS["GaussianNB"]                 = GaussianNB()
    CLASSIFIERS["RFClassifier"]     = RandomForestClassifier()
    CLASSIFIERS["MLPClassifier"]              = MLPClassifier(
                                                        activation = "tanh")
    CLASSIFIERS["ABClassifier"]         = AdaBoostClassifier()
    CLASSIFIERS["GBClassifier"] = GradientBoostingClassifier()
    CLASSIFIERS["ETClassifier"]       = ExtraTreesClassifier()

    print("\n * Testing {} Classifiers ...".format(len(CLASSIFIERS)))
    i = 0
    for name, model in CLASSIFIERS.items():
        i += 1
        start_time = timeit.default_timer()
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        print("\n  {} > {}.".format(i, name))
        print("      Accuracy : {}".format(accuracy_score(y_test, y_pred)))
        print("      Precision: {}".format(precision_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      Recall   : {}".format(recall_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      F1       : {}".format(f1_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      Execution time: {:.2f} secs.".format(timeit.default_timer(
                                                            ) - start_time))

def tunning_parameters_ann(pca_data):
    """.
    """
    FEATURES = list(set(pca_data.columns) - set(TARGET))

    X = pca_data[FEATURES]
    y = pca_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size = TEST_SIZE, random_state = RANDOM_STATE)

    layer_sizes = (50,), (50, 50), (50, 50, 50), (100,), (100, 100), (100, 100, 100)
    activation  = "identity", "logistic", "tanh", "relu"

    i = 0
    for layers, function in itertools.product(layer_sizes, activation):
        i += 1
        ANN = MLPClassifier(hidden_layer_sizes = layers, activation = function)
        start_time = timeit.default_timer()
        ANN.fit(X_train, y_train.values.ravel())
        y_pred = ANN.predict(X_test)
        print("\n  Iteration {}:".format(i))
        print("      Layers Sizes : {}".format(layers))
        print("      Activation   : {}".format(function))
        print("      Accuracy : {}".format(accuracy_score(y_test, y_pred)))
        print("      Precision: {}".format(precision_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      Recall   : {}".format(recall_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      F1       : {}".format(f1_score(y_test, y_pred,
                                                        average = "weighted")))
        print("      Execution time: {:.2f} secs.".format(timeit.default_timer(
                                                            ) - start_time))
    print("")


if __name__ == '__main__':
    print("")
    print(__doc__, end = "\n")
    print("* Calculating the explicit aspects of tweets using the Apriori Algorithm ...")
    EXPLICIT_ASPECTS = impute_explicit_aspects(data)
    print("* The explicit aspects obtained from this dataset are:", EXPLICIT_ASPECTS)

    print("* Calculating the Implicit Aspects of Tweets Using the SDP Algorithm ...")
    IMPLICIT_ASPECTS = impute_implicit_aspects(data)
    # print("* The implicit aspects obtained are:", IMPLICIT_ASPECTS)
    print("* Pre-processing the data before feature selection ...")
    data             = data_preprocessing(data)
    print("*Imputing overall sentiment metrics to tweets ...")
    data             = impute_overall_sentiment(data)
    # data.to_csv("test.csv", index = False)
    print("*Applying PCA for feature selection ...")
    pca_data         = feature_selection(data)
    res = input("> Would you like to train and test all 10 Machine Learning models? Y/N: ")
    if res.upper() in ("Y", "YES"):
        training_testing_models(pca_data)
    print("\n* Testing different parameters for ANN models.")
    input("* Press the ENTER key to get results of 24 iterations ...")
    tunning_parameters_ann(pca_data)
