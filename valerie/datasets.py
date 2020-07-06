import os
import glob
import logging
import datetime
import multiprocessing

import tldextract
import pandas as pd
from tqdm.auto import tqdm

from valerie.data import Claim, Article

_logger = logging.getLogger(__name__)


####################
####### Base #######
####################


class ValerieDataset:
    name = "_valerie_dataset"

    def __init__(self, claims, articles=None):
        self.claims = list(set(claims))
        _logger.info(
            "%s claims set change %d --> %d", self.name, len(claims), len(self.claims),
        )

        if articles:
            self.articles = list(set(articles))
            _logger.info(
                "%s articles set change %d --> %d",
                self.name,
                len(articles),
                len(self.articles),
            )

    @classmethod
    def df_to_claims(cls, df, row_to_claim):
        claims = []
        misses = 0
        for i, row in tqdm(
            df.iterrows(), total=len(df), desc="{} to claims".format(cls.name)
        ):
            # if phase1/phase2, do not try/except an error when parsing df
            if cls.name == "phase1" or cls.name == "phase2":
                claims.append(row_to_claim(i, row))
            else:
                try:
                    claims.append(row_to_claim(i, row))
                except:
                    misses += 1
                    continue
        _logger.info("missed row to claim conversions: %d", misses)
        return claims

    @classmethod
    def from_raw(cls):
        raise NotImplementedError

    @classmethod
    def from_all_filter(cls, claims, articles=None):
        _claims = [c for c in claims if c.dataset_name == cls.name]
        if articles:
            _articles = [a for a in articles if a.dataset_name == cls.name]
        return cls(_claims, _articles)


####################
##### Internal #####
####################


class Phase1Dataset(ValerieDataset):
    name = "phase1"

    @classmethod
    def from_raw(
        cls, metadata_file="data/phase1/raw/metadata.json", articles_dir=None, nproc=1
    ):
        # because pandas logging sucks
        if not os.path.isfile(metadata_file):
            raise ValueError(
                "metadata file {} was not found or is not file".format(metadata_file)
            )
        if articles_dir and not os.path.isdir(articles_dir):
            raise ValueError(
                "articles dir {} was not found or is not dir".format(articles_dir)
            )

        df = pd.read_json(metadata_file)
        claims = cls.df_to_claims(df, cls.row_to_claim)

        articles = None
        if articles_dir:
            articles = cls.articles_from_phase1(articles_dir, nproc)

        return cls(claims, articles)

    @classmethod
    def row_to_claim(cls, i, row):
        row = dict(row)
        _id = row.pop("id")

        # only parse related articles if it exists
        # (we do this check since related_articles is a removed field for the eval)
        related_articles = {}
        if "related_articles" in row:
            for rel_art in row.pop("related_articles"):
                rel_art = cls.name + "/" + str(rel_art) + ".txt"
                related_articles[rel_art] = rel_art

        return Claim(
            _id, related_articles=related_articles, dataset_name=cls.name, **row
        )

    @staticmethod
    def articles_from_phase1(articles_dir, nproc=1):
        fpaths = glob.glob(os.path.join(articles_dir, "*.txt"))

        pool = multiprocessing.Pool(nproc)
        articles = []
        for article in tqdm(
            pool.imap_unordered(_articles_from_phase1_visit, fpaths),
            total=len(fpaths),
            desc="loading articles from phase1",
        ):
            articles.append(article)

        return articles


def _articles_from_phase1_visit(fpath):
    with open(fpath, encoding="utf8") as fi:
        art_id = os.path.basename(fpath)
        article = Article.from_txt(art_id, fi.read(), dataset_name=Phase1Dataset.name)
    return article


class Phase2Dataset(ValerieDataset):
    name = "phase2"

    @classmethod
    def from_raw(
        cls, metadata_file="data/phase2-1/raw/metadata.json", articles_dir=None, nproc=1
    ):
        # because pandas logging sucks
        if not os.path.isfile(metadata_file):
            raise ValueError(
                "metadata file {} was not found or is not file".format(metadata_file)
            )
        if articles_dir and not os.path.isdir(articles_dir):
            raise ValueError(
                "articles dir {} was not found or is not dir".format(articles_dir)
            )

        df = pd.read_json(metadata_file)
        claims = cls.df_to_claims(df, cls.row_to_claim)

        articles = None
        if articles_dir:
            articles = cls.articles_from_phase2(articles_dir, claims, nproc=nproc)

        return cls(claims, articles)

    @classmethod
    def row_to_claim(cls, i, row):
        row = dict(row)
        _id = row.pop("id")

        # only parse related articles if it exists
        # (we do this check since related_articles is a removed field for the eval)
        related_articles = {}
        if "related_articles" in row:
            for k, v in row.pop("related_articles").items():
                rel_art = cls.name + "/" + os.path.basename(k)
                related_articles[rel_art] = v

        return Claim(
            _id, related_articles=related_articles, dataset_name=cls.name, **row
        )

    @staticmethod
    def articles_from_phase2(articles_dir, claims, nproc=1):
        fpaths = glob.glob(os.path.join(articles_dir, "*.html"))

        pool = multiprocessing.Pool(nproc)
        articles = []
        for article in tqdm(
            pool.imap_unordered(_articles_from_phase2_visit, fpaths),
            total=len(fpaths),
            desc="loading article from phase2",
        ):
            articles.append(article)

        # fetch the urls for each article from the claims and perform tldextract
        misses = 0
        art_index_to_url = {
            k: v for claim in claims for k, v in claim.related_articles.items()
        }
        for article in articles:
            try:
                article.url = art_index_to_url[article.index]
            except:
                misses += 1
                continue

            article.source = tldextract.extract(article.url).domain
        _logger.info("missed art index to url conversions: %d", misses)

        return articles


def _articles_from_phase2_visit(fpath):
    with open(fpath, encoding="utf8") as fi:
        art_id = os.path.basename(fpath)
        article = Article.from_html(art_id, fi.read(), dataset_name=Phase2Dataset.name)
    return article


####################
##### External #####
####################


class FakeNewsTop50Dataset(ValerieDataset):
    """https://github.com/BuzzFeedNews/2018-12-fake-news-top-50.git"""

    name = "fake_news_top50"

    @classmethod
    def from_raw(
        cls,
        top_csv="data/external/2018-12-fake-news-top-50/data/top_2018.csv",
        sites_csvs=[
            "data/external/2018-12-fake-news-top-50/data/sites_2016.csv",
            "data/external/2018-12-fake-news-top-50/data/sites_2017.csv",
            "data/external/2018-12-fake-news-top-50/data/sites_2018.csv",
        ],
    ):
        df = pd.read_csv(top_csv)

        sites = []
        for sites_csv in sites_csvs:
            with open(sites_csv) as fi:
                sites += fi.read().splitlines()
        sites = list(set(sites))

        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        dataset.sites = sites
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        # TODO: consider lowercasing the input claim (all words
        # start with capital currently)
        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            date=row["published_date"],
            claimant="Facebook user",
            label=0,
            dataset_name=cls.name,
        )


class FakeNewsKaggleDataset(ValerieDataset):
    """https://www.kaggle.com/c/fake-news/"""

    name = "fake_news_kaggle"

    @classmethod
    def from_raw(cls, train_csv="data/external/fake-news/train.csv"):
        df = pd.read_csv(train_csv)
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        # label 0 for reliable
        # label 1 for unreliable
        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            claimant=row["author"],
            label=0 if row["label"] else 2,
            dataset_name=cls.name,
        )


class FakeNewsNetDataset(ValerieDataset):
    """https://github.com/KaiDMML/FakeNewsNet.git"""

    name = "fake_news_net"

    @classmethod
    def from_raw(
        cls,
        politifact_fake_csv="data/external/FakeNewsNet/dataset/politifact_fake.cs",
        politifact_real_csv="data/external/FakeNewsNet/dataset/politifact_real.cs",
        gossipcop_fake_csv="data/external/FakeNewsNet/dataset/gossipcop_fake.cs",
        gossipcop_real_csv="data/external/FakeNewsNet/dataset/gossipcop_real.cs",
        name="fake_news_net",
    ):
        df = pd.concat(
            [
                pd.read_csv(politifact_fake_csv).assign(label=0),
                pd.read_csv(politifact_real_csv).assign(label=2),
                pd.read_csv(gossipcop_fake_csv).assign(label=0),
                pd.read_csv(gossipcop_real_csv).assign(label=2),
            ],
            ignore_index=True,
        )

        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            claimant=tldextract.extract(row["news_url"]).domain,
            label=row["label"],
            dataset_name=cls.name,
        )


class GeorgeMcIntireDataset(ValerieDataset):
    """https://github.com/GeorgeMcIntire"""

    name = "george_mcintire"

    @classmethod
    def from_raw(cls, data_csv="data/external/george-mcintire/fake_or_real_news.csv"):
        df = pd.read_csv(data_csv, skiprows=1, names=["id", "title", "text", "label"])
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            label=0 if row["label"] == "FAKE" else 1,
            dataset_name=cls.name,
        )


class ISOTDataset(ValerieDataset):
    """https://www.uvic.ca/engineering/ece/isot/datasets/"""

    name = "isot"

    @classmethod
    def from_raw(
        cls,
        fake_csv="data/external/ISOT/Fake.csv",
        true_csv="data/external/ISOT/True.csv",
    ):
        df = pd.concat(
            [
                pd.read_csv(fake_csv).assign(label=0),
                pd.read_csv(true_csv).assign(label=2),
            ],
            ignore_index=True,
        )
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        try:  # December 31, 2017
            _date = datetime.datetime.strptime(row["date"], "%B %d, %Y")
        except:  # 19-Feb-18
            try:
                _date = datetime.datetime.strptime(row["date"], "%d-%b-%y")
            except:  # Dec 31, 2017
                try:
                    _date = datetime.datetime.strptime(row["date"], "%b %d, %Y")
                except:
                    _date = None

        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            date=_date.strftime("%Y-%m-%d") if _date else None,
            label=row["label"],
            dataset_name=cls.name,
        )


class LiarDataset(ValerieDataset):
    """https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"""

    name = "liar"

    @classmethod
    def from_raw(cls, data_tsv="data/external/liar/train.tsv"):
        df = pd.read_csv(
            data_tsv,
            sep="\t",
            names=[
                "id",
                "label",
                "statement",
                "subject(s)",
                "speaker",
                "speaker's job title",
                "state info",
                "party affiliation",
                "total credit history count",
                "barely true counts",
                "false counts",
                "half true counts",
                "mostly true counts",
                "context (venue/location of speech or statement)",
                "pants on fire counts",
            ],
        )
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        if row["label"] == "false":
            _lab = 0
        elif row["label"] == "true":
            _lab = 2
        else:
            _lab = 1

        return Claim(
            cls.name + "/" + str(i),
            claim=row["statement"],
            claimant=row["speaker"] if isinstance(row["speaker"], str) else None,
            label=_lab,
            dataset_name=cls.name,
        )


class MrisdalDataset(ValerieDataset):
    """https://www.kaggle.com/mrisdal/fake-news"""

    name = "mrisdal"

    @classmethod
    def from_raw(cls, data_csv="data/external/mrisdal/fake.csv"):
        df = pd.read_csv(data_csv)
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        if row["ord_in_thread"] != 0:
            raise ValueError("must be main post")
        return Claim(
            cls.name + "/" + str(i),
            claim=row["title"],
            claimant=row["site_url"],
            date=datetime.datetime.strptime(
                row["published"].split("T")[0], "%Y-%m-%d"
            ).strftime("%Y-%m-%d"),
            label=0,
            dataset_name=cls.name,
        )


####################
##### Combined #####
####################


class CombinedDataset(ValerieDataset):
    name = "combined"

    @classmethod
    def from_raw(cls):
        datasets = [
            Phase2Dataset.from_raw(),
            Phase1Dataset.from_raw(),
            FakeNewsTop50Dataset.from_raw(),
            FakeNewsKaggleDataset.from_raw(),
            FakeNewsNetDataset.from_raw(),
            GeorgeMcIntireDataset.from_raw(),
            ISOTDataset.from_raw(),
            LiarDataset.from_raw(),
            MrisdalDataset.from_raw(),
        ]

        # IMPORTANT that phase2 is first dataset in datasets and combined_claims_set
        # is on the left side of the union. The elements of the left set will be used
        # for the new set in a union, and we want phase2 claims to override
        # other claims since it has more useful and complete data. This also means,
        # the order of the datasets should follow the priority of their information
        assert isinstance(datasets[0], Phase2Dataset)

        _logger.info("... constructing combined dataset ...")
        combined_claims_set = set()
        for dataset in datasets:
            prev_len = len(combined_claims_set)
            combined_claims_set = combined_claims_set | set(dataset.claims)
            _logger.info(
                "%s: %d --> %d (+ %d = %d - %d)",
                dataset.name,
                prev_len,
                len(combined_claims_set),
                len(combined_claims_set) - prev_len,
                len(dataset.claims),
                prev_len + len(dataset.claims) - len(combined_claims_set),
            )
        return cls(list(combined_claims_set))
