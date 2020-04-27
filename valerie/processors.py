"""Preprocessors."""
import heapq
import logging
import threading
import multiprocessing

import nltk
import numpy as np
from tqdm import tqdm
from scipy import spatial
from transformers import InputExample
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from . import utils

_logger = logging.getLogger(__name__)

_process_this = None
_process_this_lock = threading.Lock()

class MultiClaimSupportProcessor:
    """Processor for multiple claim support pair examples."""

    def __init__(self, articles, word2vec_file, keep_n=8, min_threshold=0.40, min_examples=4, nproc=1):
        """Constructor for ClaimPreprocessor.

        Parameters
        ----------
        train_claims : list of Claim
            Training claims.
        dev_claims : list of Claim
            Dev claims.
        articles : dict of (id, Article)
            Dict of id to Article.
        word2vec_file : str
            Path to saved word2vec model file.
        keep_n : int
            Max number of related sentences to keep.
        min_threshold : float
            Minimum score (between 0 and 1) the claim-sentence pair must be
            achieved to be included in the output (after `min_examples` examples
            have been chosen).
        min_examples : int
            Minimum number of examples to include in the output.
        nproc : int
            Number of processors to use.
        """
        self.articles = articles
        self.word2vec_model = utils.load_word2vec(word2vec_file)
        self.keep_n = keep_n
        self.min_threshold = 0.40
        self.min_examples = min_examples
        self.nproc = nproc

    def get_labels(self):
        """See base class."""
        return [0,1,2]

    def generate_examples(self, claims):
        """Gets`InputExample`s for a set of claims."""
        examples = []
        for claim in claims:
            support = self._generate_support(claim)
            for s in support:
                examples.append(InputExample(
                    guid=claim.id,
                    text_a=claim.claim,
                    text_b=s["text"],
                    label=claim.label
                ))
        return examples

    # def generate_examples(self, claims):
    #     """Gets`InputExample`s for a set of claims."""
    #     with _process_this_lock:
    #         global _process_this
    #         _process_this = (self, claims)
    #         with multiprocessing.Pool(self.nproc) as pool:
    #             examples = list(tqdm(pool.imap(
    #                 self._generate_examples, range(len(claims)), chunksize=100),
    #             total=len(claims)))
    #     # flatten examples list
    #     examples = [e for example in examples for e in example]
    #     return examples

    # @staticmethod
    # def _generate_examples(claim_index):
    #     processor, claims = _process_this
    #     claim = claims[claim_index]

    #     support = processor._generate_support(claim)
    #     examples = [InputExample(
    #         guid=claim.id,
    #         text_a=claim.claim,
    #         text_b=s["text"],
    #         label=claim.label
    #     ) for s in support]

    #     return examples

    def _generate_support(self, claim):
        """Finds sentences related to the claim using related articles.

        Finds related sentences to the claim from it's list of related articles
        using a score computed with the following steps:

        1.  Compute the tfidf and word2vec embeddings for the claim and all
            sentences in the related articles.
        2.  Take the cosine similarity between each of claim-sentence pair
            (one tfidf score and one word2vec score for each pair).
        3.  Take the average of the tfidf and word2vec similarity scores

        Parameters
        ----------
        claim : str
            A Claim.

        Returns
        -------
        examples : list
            Returns a list of dict of each related sentences' metadata
            (score, text, articles id).
        """
        # get sentences from related_articles
        corpus = [
            (ref,sentence)
            for ref in claim.related_articles
            for sentence in utils.split_sentences(self.articles[ref].content)
        ]
        references, sentences = map(list, zip(*corpus))

        # append and pad claim sentence
        sentences.append(claim.claim)
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
                    vectors.append(self.word2vec_model[word])
                except:
                    continue
            return np.nan if not vectors else np.mean(vectors, axis=0)
        word2vec_vectors = [word2vec_sentence(sentence) for sentence in sentences]

        # calculate and sort cosine similarities for embeddings
        support = []
        for ref, sentence, tfidf_vector, word2vec_vector in zip(references, sentences, tfidf_vectors, word2vec_vectors):
            tfidf_score = float(cosine_similarity(tfidf_vectors[-1], tfidf_vector))
            word2vec_score = 0.0 if np.isnan(np.min(word2vec_vector)) else float(1 - spatial.distance.cosine(word2vec_vectors[-1], word2vec_vector))
            support.append({
                "article_id": ref,
                "text": sentence,
                "scores": {
                    "tfidf": tfidf_score,
                    "word2vec": word2vec_score
                },
                "score": float(sum([tfidf_score, word2vec_score])) / 2
            })
        support.pop() # remove the claim sentence itself from support

        # sort and get nlargest
        if self.keep_n:
            support = heapq.nlargest(self.keep_n, support, key=lambda x: x["score"])
            support = sorted(support, key=lambda x: x["score"], reverse=True)
            support = [s for i,s in enumerate(support) if s["score"] >= self.min_threshold or i < self.min_examples]
        else:
            support = sorted(support, key=lambda x: x["score"], reverse=True)

        return support
