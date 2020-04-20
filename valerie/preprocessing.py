"""Preprocessors."""
import heapq
import logging

import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from . import utils

_logger = logging.getLogger(__name__)


class ArticlePreprocessor:
    """Preprocessor for Articles."""

    def __init__(self, articles, word2vec_path):
        """Constructor for preprocessor.

        Parameters
        ----------
        articles : dict of (id, Article)
            Dict of id to Article.
        word2vec_path : str
            Path to saved word2vec model file.
        """
        self.articles = articles
        self.word2vec_model = utils.load_word2vec(word2vec_path)

    def generate_support(self, claim, related_articles, keep_n=8, min_threshold=0.40, min_examples=4):
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
        claim : str
            The claim.
        related_articles : list
            List of ids/references for related articles to the claim.
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
        corpus = [(ref,sentence) for ref in related_articles for sentence in self.articles[str(ref)].body]
        references, sentences = map(list, zip(*corpus))

        # append and pad claim sentence
        sentences.append(claim)
        references.append(None)

        # get tf_idf vectors
        _logger.info("... generating tdidf vectors ...")
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)

        # get sentence word2vec vectors
        _logger.info("... fetching word2vec embeddings ...")
        def word2vec_sentence(sentence):
            words = nltk.tokenize.word_tokenize(sentence)
            vectors = []
            for word in words:
                try:
                    vectors.append(self.word2vec_model[word])
                except:
                    continue
            return np.nan if not vectors else np.mean(vectors, axis=0)
        word2vec_vectors = [word2vec_sentence(sentence) for sentence in sentences]

        # calculate and sort cosine similarities for embeddings
        _logger.info("... computing cosine similarity scores ...")
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

        # sort and get nlargest
        if keep_n:
            support = heapq.nlargest(keep_n, support, key=lambda x: x["score"])
            support = sorted(support, key=lambda x: x["score"], reverse=True)
            support = [s for i,s in enumerate(support) if s["score"] >= min_threshold or i < min_examples]
        else:
            support = sorted(support, key=lambda x: x["score"], reverse=True)

        return support
