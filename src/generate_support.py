#### Setup ####
# Python Standard Library
import heapq
import logging
import multiprocessing

# Third Party Packaged
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]
logger.info('*** Logging Configurated ***')


#### Create Support for Single Claim ####
def generate_support_from_claim(sample, keep_n=5):
    # Get sentences from related_articles
    corpus = [(ref,sentence) for ref in sample["related_articles"] for sentence in articles_data[str(ref)]]
    references, sentences = map(list, zip(*corpus))
    
    # Append and pad claim sentence
    sentences.append(sample["claim"])
    references.append(None)

    # Get tf_idf vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    
    # Calculate and sort cosine similarities
    support = []
    for ref, sentence, vector in zip(references, sentences, vectors):
        support.append({
            "source_article": ref,
            "text": sentence,
            "cosine_similarity": float(cosine_similarity(vectors[-1], vector))
        })
    support.pop() # remove the claim sentence itself from support

    # Sort or get nlargest
    if keep_n:
        support = heapq.nlargest(keep_n, support, key=lambda x: x["cosine_similarity"])
    else:
        support = sorted(keep_n, key=lambda x: x["cosine_similarity"], reverse=True)
    
    # Update and return sample
    sample["support"] = support
    return sample


#### Create Support for All Data ####
def generate_support(data, _articles_data, keep_n=5, nproc=1): 
    # Make articles_data global
    global articles_data
    articles_data = _articles_data

    # Multiprocessing
    pool = multiprocessing.Pool(nproc)

    # Generate support for each claim
    logger.info("... generating support ...")
    samples = []
    for sample in tqdm(pool.imap_unordered(generate_support_from_claim, data), total=len(data)):
        samples.append(sample)

    return samples
