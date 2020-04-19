"""Text preprocessing."""
import heapq
import logging
import unicodedata

import nltk
import gensim
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

_logger = logging.getLogger(__name__)


def generate_support(sample, articles, word2vec_model, keep_n=8, min_threshold=0.40, min_examples=4):
    """Generate nlp metadata for a claim.

    This function finds sentences from the list of related articles that best
    match the contents of the claim.

    To achieve this, it first computes the tfidf and word2vec embeddings for
    both the claim, and all the sentences in the related articles. It then takes
    cosine similarity between each of these claim-sentence pairs to quantify
    their similarity (one tfidf score and one word2vec score for each pair). It
    takes the average of the tfidf and word2vec similarities and adds the
    top `keep_n` matching sentences for each claim along with their scores and
    source article identifier.

    Parameters
    ----------
    sample : dict
        Single entry from phase1 metadata.json containing claim,
        related_articles, and other metadata.
    articles_data : dict
        Dictionary of the related articles where the keys are the related
        article id and the values are a list of sentences (as strs).
    word2vec_model
        Loaded gensim word2vec model.
    keep_n : int
        Max number of related sentences to keep.
    min_threshold : float
        Minimum score (between 0 and 1) the claim-sentence pair must be achieved
        to be included in the output (after `min_examples` examples have been
        chosen).
    min_examples : int
        Minimum number of examples to include in the output.

    Returns
    -------
    sample : dict
        Returns the generated nlp metadata.
    """
    # get sentences from related_articles
    corpus = [(ref,sentence) for ref in sample["related_articles"] for sentence in articles[str(ref)]]
    references, sentences = map(list, zip(*corpus))

    # append and pad claim sentence
    sentences.append(sample["claim"])
    references.append(None)

    # get tf_idf vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)

    # get sentence word2vec vectors
    def word2vec_sentence(sentence):
        words = nltk.tokenize.word_tokenize(sentence)
        vectors = []
        for word in words:
            try:
                vectors.append(word2vec_model[word])
            except:
                continue
        return np.nan if not vectors else np.mean(vectors, axis=0)
    word2vec_vectors = [word2vec_sentence(sentence) for sentence in sentences]

    # calculate and sort cosine similarities for embeddings
    support = []
    for ref, sentence, tfidf_vector, word2vec_vector in zip(references, sentences, tfidf_vectors, word2vec_vectors):
        support.append({
            "source_article": ref,
            "text": sentence,
            "scores": {
                "tfidf": float(cosine_similarity(tfidf_vectors[-1], tfidf_vector)),
                "word2vec": 0.0 if np.isnan(np.min(word2vec_vector)) else float(1 - spatial.distance.cosine(word2vec_vectors[-1], word2vec_vector))
            }
        })
    # calculate overall similarity (mean of cosine similarities)
    for s in support:
        s["score"] = float(np.mean(list(s["scores"].values()), axis=0))
    support.pop() # remove the claim sentence itself from support

    # sort or get nlargest
    if keep_n:
        support = heapq.nlargest(keep_n, support, key=lambda x: x["score"])
        support = sorted(support, key=lambda x: x["score"], reverse=True)
        support = [s for i,s in enumerate(support) if s["score"] >= min_threshold or i < min_examples]
    else:
        support = sorted(support, key=lambda x: x["score"], reverse=True)

    return support


def load_word2vec(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def split_sentences(text):
    """Returns the input text split into a list of sentences."""
    return [sentence.strip() for sentence in nltk.tokenize.sent_tokenize(text)]


def clean_text(text):
    """Cleans the text of whitespace and control chars."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip()


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False
