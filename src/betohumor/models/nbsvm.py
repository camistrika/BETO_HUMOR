from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_nbsvm_vectorizer(nbsvm_config):
    """Construye el TfidfVectorizer a partir de NbsvmConfig."""
    return TfidfVectorizer(
        ngram_range=nbsvm_config.ngram_range,
        min_df=nbsvm_config.min_df,
        max_df=nbsvm_config.max_df,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )


def build_nbsvm_classifier(nbsvm_config, seed):
    """Construye la Regresión Logística a partir de NbsvmConfig."""
    return LogisticRegression(C=nbsvm_config.C, max_iter=1000, random_state=seed)
