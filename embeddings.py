import numpy as np
from gensim.models import KeyedVectors

def load_w2v_model(path):
    return KeyedVectors.load_word2vec_format(
        path,
        binary=True
    )

def avg_word2vec(words, model, num_features=300):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0

    for word in words:
        if word in model.key_to_index:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])

    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec