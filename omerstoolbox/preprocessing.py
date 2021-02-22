import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.inspection import permutation_importance


def feature_permutation(model, X, y, n_repeats):
    '''returns feature importance dataframe for an instantiated model'''

    base_model = model.fit(X, y)

    # Perform Permutation
    permutation_score = permutation_importance(base_model, X, y, n_repeats=n_repeats)
    importance_df = pd.DataFrame(np.vstack((X.columns,
                                            permutation_score.importances_mean)).T) # Unstack results
    importance_df.columns = ['feature', 'score decrease']

    # Order by importance
    return importance_df.sort_values(by="score decrease", ascending = False)


def corr_table(df):
    '''returns correlation table for a DataFrame'''

    corr = df.corr()
    corr_df = corr.unstack().reset_index() # Unstack correlation matrix
    corr_df.columns = ['feature_1', 'feature_2', 'correlation'] # rename columns
    corr_df.sort_values(by="correlation", ascending=False, inplace=True) # sort by correlation
    corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']] # Remove self correlation
    return corr_df


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # remove punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')

    # remove numbers
    text = ''.join(word for word in text if not word.isdigit())

    # Remove stopwords
    word_tokens = word_tokenize(text)
    rem_stop = [w for w in word_tokens if w not in stop_words]

    # lemmatize
    text = ' '.join(lemmatizer.lemmatize(word) for word in rem_stop)

    return text.lower()
