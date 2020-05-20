"""Preprocessors."""
import heapq
import logging
import threading

import nltk
import numpy as np
from tqdm import tqdm
from scipy import spatial
from transformers import InputExample
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from . import utils
from .preprocessing import split_sentences

_logger = logging.getLogger(__name__)


class MultiClaimSupportProcessor:
    """Processor for multiple claim support pair examples."""

    def __init__(
        self,
        articles,
        word2vec_file,
        min_examples=4,
        max_examples=8,
        min_threshold=0.40,
    ):
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
        min_examples : int
            Minimum number of examples to include in the output.
        max_examples : int
            Max number of related sentences to include in the output.
        min_threshold : float
            Minimum score (between 0 and 1) the claim-sentence pair must be
            achieved to be included in the output (after `min_examples` examples
            have been chosen).
        """
        self.articles = articles
        self.word2vec_model = utils.load_word2vec(word2vec_file)
        self.min_examples = min_examples
        self.max_examples = max_examples
        self.min_threshold = 0.40

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def generate_examples(self, claims):
        """Gets`InputExample`s for a set of claims."""
        examples = []
        for claim in claims:
            support = self._generate_support(claim)
            for s in support:
                examples.append(
                    InputExample(
                        guid=claim.id,
                        text_a=claim.claim,
                        text_b=s["text"],
                        label=claim.label,
                    )
                )
        return examples

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
        document = [
            (ref, sentence)
            for ref in claim.related_articles
            for sentence in split_sentences(self.articles[ref].content)
        ]
        references, sentences = map(list, zip(*document))

        # restrict the number of sentences allowed to be processed to 256
        # long documents will bog down computation times
        # (255 since we append the claim itself, making it 256)
        # references = references[:255]
        # sentences = sentences[:255]

        # append and pad claim sentence
        # sentences.append(claim.claim)
        # references.append(None)

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
        for ref, sentence, tfidf_vector, word2vec_vector in zip(
            references, sentences, tfidf_vectors, word2vec_vectors
        ):
            # we use two different cosine functions since the tf_idf vectors are
            # sparse (so scipy.distance won't work on them, we are forced to use
            # sklearn), and the word2vec vectors are of shape (embedding_size)
            # so we can use scipy without having to reshape the vectors (and scipy
            # is faster at computing the similarity score)
            tfidf_score = float(cosine_similarity(tfidf_vectors[-1], tfidf_vector))
            word2vec_score = (
                0.0
                if np.isnan(np.min(word2vec_vector))
                else float(
                    1 - spatial.distance.cosine(word2vec_vectors[-1], word2vec_vector)
                )
            )
            support.append(
                {
                    "article_id": ref,
                    "text": sentence,
                    "scores": {"tfidf": tfidf_score, "word2vec": word2vec_score},
                    "score": float(sum([tfidf_score, word2vec_score])) / 2,
                }
            )
        support.pop()  # remove the claim sentence itself from support

        if self.max_examples:
            support = heapq.nlargest(
                self.max_examples, support, key=lambda x: x["score"]
            )
            support = [
                s
                for i, s in enumerate(support)
                if s["score"] >= self.min_threshold or i < self.min_examples
            ]
        else:
            support = sorted(support, key=lambda x: x["score"], reverse=True)

        return support


class SingleClaimSupportProcessor(MultiClaimSupportProcessor):
    """Processor for single claim support pair examples."""

    def __init__(
        self,
        articles,
        word2vec_file,
        min_examples=3,
        max_examples=5,
        min_threshold=0.50,
    ):
        """See SingleClaimSupportProcessor."""
        super.__init__(
            articles, word2vec_file, max_examples, min_threshold, min_examples
        )

    def generate_example(self, claim):
        support = self._generate_support(claim)

        support_text = ""
        for s in support:
            support_text += s["text"] + " "
        support_text.strip()

        example = InputExample(
            guid=claim.id, text_a=claim.claim, text_b=support_text, label=claim.label
        )
        return example

    def generate_examples(self, claims):
        examples = [self.generate_example(claim) for claim in claims]
        return examples


def tfidf_cosine_similarity(v1, v2):
    # tf_idf vectors are sparse (so scipy.distance won't work on them, we are
    # forced to use sklearn)
    return float(cosine_similarity(v1, v2))


def word2vec_embed_sentence(tokens, word2vec_model):
    vectors = []
    for token in tokens:
        try:
            vectors.append(word2vec_model[token])
        except:
            continue
    return np.nan if not vectors else np.mean(vectors, axis=0)


def word2vec_cosine_similarity(v1, v2):
    # word2vec vectors are of shape (embedding_size) so we can use scipy
    # without having to reshape the vectors vs using sklearn (plus scipy
    # is faster at computing the similarity score)
    return (
        0.0
        if np.isnan(np.min(v1)) or np.isnan(np.min(v2))
        else float(1 - spatial.distance.cosine(v1, v2))
    )


def generate_sentence_similarity_scores(subject, sentences, word2vec_model, avg=True):
    """Generates similarity scores for the subject sentence against other sentences.

    Scores are computed in the following steps:

    1.  Compute the tfidf and word2vec embeddings for the subject and sentences.
    2.  Take the cosine similarity between each of subject-sentence pairs
        (one tfidf score and one word2vec score for each pair).
    3.  Take the average of the tfidf and word2vec similarity scores
    """
    sentences += subject

    # tfidf sentence vectors
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # try the stopwords option?
    tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)
    assert len(sentences) == len(tfidf_vectors)

    # word2vec sentence vectors
    word2vec_vectors = [
        word2vec_embed_sentence(sentence, word2vec_model) for sentence in sentences
    ]
    assert len(sentences) == len(word2vec_vectors)

    # calculate and sort cosine similarity scores for each vector
    scores = []
    for tfidf_vector, word2vec_vector in zip(tfidf_vectors, word2vec_vectors):
        scores.append(
            {
                "tfidf": tfidf_cosine_similarity(tfidf_vectors[-1], tfidf_vector),
                "word2vec": word2vec_cosine_similarity(
                    word2vec_vectors[-1], word2vec_vector
                ),
            }
        )
    scores.pop()  # remove the subject sentence itself from the scores

    if avg:
        return [sum(score.values()) / len(score.values()) for score in scores]
    else:
        return scores
