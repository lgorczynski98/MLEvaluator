import numpy as np
from sklearn.datasets import make_moons, make_blobs
import pandas as pd
import os
import re
import re
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import HashingVectorizer


def generate_batch_xor(n, mu=0.5, sigma=0.5):
    """ Four gaussian clouds in a Xor fashion """
    X = np.random.normal(mu, sigma, (n, 2))
    yB0 = np.random.uniform(0, 1, n) > 0.5
    yB1 = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y0 = 2. * yB0 - 1
    y1 = 2. * yB1 - 1
    X[:,0] *= y0
    X[:,1] *= y1
    X -= X.mean(axis=0)
    return X, y0*y1

def generate_moons(n_samples, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y

def generate_blobs(n_samples, centers=2, n_features=2, random_state=0, cluster_std=0.5):
    X, y = make_blobs(n_samples=2000, centers=2, n_features=2, random_state=0, cluster_std=0.5)
    return X, y

def tokenizer(text):
    stop = stopwords.words('english')
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def generate_imdb_dataset():
    nltk.download('stopwords')
    vect = HashingVectorizer(decode_error='ignore',
                        n_features=2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)
    df = pd.read_csv(os.path.join('datasets', 'imdb', 'imdb_movie_data.csv'))
    X = df['Review'].to_numpy()
    X = vect.transform(X)
    y = df['Sentiment'].to_numpy().flatten()
    return X, y