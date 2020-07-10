import os
import glob
import json
import logging
import datetime
import multiprocessing

import tldextract
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from valerie.data import Claim, Article, combine_claims

_logger = logging.getLogger(__name__)


####################
####### Base #######
####################


class ValerieDataset:
    def __init__(self, claims, articles=None):
        self.claims = list(set(claims))
        _logger.info(
            "%s claims set change %d --> %d",
            self.__class__.__name__,
            len(claims),
            len(self.claims),
        )

        if articles:
            self.articles = list(set(articles))
            _logger.info(
                "%s articles set change %d --> %d",
                self.__class__.__name__,
                len(articles),
                len(self.articles),
            )

    def train_test_split(self, **kwargs):
        _logger.info("... performing train_test_split ...",)
        _labels = [claim.label for claim in self.claims]
        train_claims, test_claims, _, _ = train_test_split(
            self.claims, _labels, stratify=_labels, **kwargs
        )
        self.train_claims = train_claims
        self.test_claims = test_claims

        _logger.info("len of all claims:    %d", len(self.claims))
        _logger.info("len of train claims:  %d", len(self.train_claims))
        _logger.info("len of test claims:   %d", len(self.test_claims))

    def train_test_split_subdataset(self, subdataset_name, **kwargs):
        """Train test split for a subdataset of a combined dataset.

        Performs a train test split on the specified subdataset that's within the
        current combined dataset. This is useful if you want all your test data
        to only come from a single dataset, rather than all the combined ones.
        """
        _logger.info(
            "... performing train_test_split_subdataset on subdataset %s ...",
            subdataset_name,
        )
        sub_claims = [
            claim for claim in self.claims if claim.dataset_name == subdataset_name
        ]
        not_sub_claims = [
            claim for claim in self.claims if claim.dataset_name != subdataset_name
        ]

        sub_labels = [claim.label for claim in sub_claims]
        train_claims, test_claims, _, _ = train_test_split(
            sub_claims, sub_labels, stratify=sub_labels, **kwargs
        )
        self.train_claims = train_claims + not_sub_claims
        self.test_claims = test_claims

        _logger.info("len of all claims:    %d", len(self.claims))
        _logger.info("len of train claims:  %d", len(self.train_claims))
        _logger.info("len of test claims:   %d", len(self.test_claims))

    @classmethod
    def df_to_claims(cls, df, row_to_claim):
        claims = []
        misses = 0
        for i, row in tqdm(
            df.iterrows(), total=len(df), desc="{} to claims".format(cls.__name__)
        ):
            # if phase1/phase2, do not try/except an error when parsing df
            if cls.__name__ in [
                Phase1Dataset.__name__,
                Phase1Dataset.__name__,
                Phase2TrialDataset.__name__,
            ]:
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
        _claims = [c for c in claims if c.dataset_name == cls.__name__]
        if articles:
            _articles = [a for a in articles if a.dataset_name == cls.__name__]
        return cls(_claims, _articles)


####################
##### Internal #####
####################


class Phase1Dataset(ValerieDataset):
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
                rel_art = cls.__name__ + "/" + str(rel_art) + ".txt"
                related_articles[rel_art] = rel_art

        return Claim(
            _id, related_articles=related_articles, dataset_name=cls.__name__, **row
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
        article = Article.from_txt(
            art_id, fi.read(), dataset_name=Phase1Dataset.__name__
        )
    return article


class Phase2Dataset(ValerieDataset):
    @classmethod
    def from_raw(
        cls, metadata_file="data/phase2-3/raw/metadata.json", articles_dir=None, nproc=1
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
                rel_art = cls.__name__ + "/" + os.path.basename(k)
                related_articles[rel_art] = v

        return Claim(
            _id, related_articles=related_articles, dataset_name=cls.__name__, **row
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
        article = Article.from_html(
            art_id, fi.read(), dataset_name=Phase2Dataset.__name__
        )
    return article


class Phase2DisjointDataset(Phase2Dataset):
    @classmethod
    def from_raw(cls, unlabelled_metadata_file, labelled_metadata_file):
        with open(unlabelled_metadata_file) as fi:
            trial_metadata_unlabelled = json.load(fi)
        with open(labelled_metadata_file) as fi:
            trial_labels = json.load(fi)

        trial_metadata = [
            {
                **claim,
                "label": trial_labels[str(claim["id"])]["label"],
                "related_articles": trial_labels[str(claim["id"])]["related_articles"],
            }
            for claim in trial_metadata_unlabelled
        ]

        df = pd.DataFrame(trial_metadata)
        claims = cls.df_to_claims(df, cls.row_to_claim)

        return cls(claims)


class Phase2TrialDataset(Phase2DisjointDataset):
    @classmethod
    def from_raw(
        cls,
        unlabelled_metadata_file="data/phase2-trial/raw/2_trial_metadata.json",
        labelled_metadata_file="data/phase2-trial/raw/2_trial_labels.json",
    ):
        return super().from_raw(
            unlabelled_metadata_file=unlabelled_metadata_file,
            labelled_metadata_file=labelled_metadata_file,
        )


class Phase2ValidationDataset(Phase2DisjointDataset):
    @classmethod
    def from_raw(
        cls,
        unlabelled_metadata_file="data/phase2-validation/raw/val_metadata_p2.json",
        labelled_metadata_file="data/phase2-validation/raw/2_labels.json",
    ):
        return super().from_raw(
            unlabelled_metadata_file=unlabelled_metadata_file,
            labelled_metadata_file=labelled_metadata_file,
        )


####################
##### External #####
####################


class FakeNewsTop50Dataset(ValerieDataset):
    """https://github.com/BuzzFeedNews/2018-12-fake-news-top-50.git"""

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
            str(i),
            claim=row["title"],
            date=row["published_date"],
            claimant="Facebook user",
            label=0,
            dataset_name=cls.__name__,
        )


class FakeNewsKaggleDataset(ValerieDataset):
    """https://www.kaggle.com/c/fake-news/"""

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
            str(i),
            claim=row["title"],
            claimant=row["author"],
            label=0 if row["label"] else 2,
            dataset_name=cls.__name__,
        )


class FakeNewsNetDataset(ValerieDataset):
    """https://github.com/KaiDMML/FakeNewsNet.git"""

    @classmethod
    def from_raw(
        cls,
        politifact_fake_csv="data/external/FakeNewsNet/dataset/politifact_fake.csv",
        politifact_real_csv="data/external/FakeNewsNet/dataset/politifact_real.csv",
        gossipcop_fake_csv="data/external/FakeNewsNet/dataset/gossipcop_fake.csv",
        gossipcop_real_csv="data/external/FakeNewsNet/dataset/gossipcop_real.csv",
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
            str(i),
            claim=row["title"],
            claimant=tldextract.extract(row["news_url"]).domain,
            label=row["label"],
            dataset_name=cls.__name__,
        )


class GeorgeMcIntireDataset(ValerieDataset):
    """https://github.com/GeorgeMcIntire"""

    @classmethod
    def from_raw(cls, data_csv="data/external/george-mcintire/fake_or_real_news.csv"):
        df = pd.read_csv(data_csv, skiprows=1, names=["id", "title", "text", "label"])
        dataset = cls(cls.df_to_claims(df, cls.row_to_claim))
        dataset.df = df
        return dataset

    @classmethod
    def row_to_claim(cls, i, row):
        return Claim(
            str(i),
            claim=row["title"],
            label=0 if row["label"] == "FAKE" else 1,
            dataset_name=cls.__name__,
        )


class ISOTDataset(ValerieDataset):
    """https://www.uvic.ca/engineering/ece/isot/datasets/"""

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
            str(i),
            claim=row["title"],
            date=_date.strftime("%Y-%m-%d") if _date else None,
            label=row["label"],
            dataset_name=cls.__name__,
        )


class LiarDataset(ValerieDataset):
    """https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"""

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
            str(i),
            claim=row["statement"],
            claimant=row["speaker"] if isinstance(row["speaker"], str) else None,
            label=_lab,
            dataset_name=cls.__name__,
        )


class MrisdalDataset(ValerieDataset):
    """https://www.kaggle.com/mrisdal/fake-news"""

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
            str(i),
            claim=row["title"],
            claimant=row["site_url"],
            date=datetime.datetime.strptime(
                row["published"].split("T")[0], "%Y-%m-%d"
            ).strftime("%Y-%m-%d"),
            label=0,
            dataset_name=cls.__name__,
        )


####################
##### Combined #####
####################


class LeadersDataset(ValerieDataset):
    @classmethod
    def from_raw(cls):
        datasets = [
            Phase2Dataset.from_raw(),
            Phase1Dataset.from_raw(),
        ]

        assert isinstance(datasets[0], Phase2Dataset)
        return cls(combine_datasets_claims(datasets))


class Phase2CombinedDataset(ValerieDataset):
    @classmethod
    def from_raw(
        cls, datasets=[],
    ):
        datasets = [Phase2Dataset.from_raw()] + [
            dataset.from_raw() for dataset in datasets
        ]
        assert isinstance(datasets[0], Phase2Dataset)
        return cls(combine_datasets_claims(datasets))


class CombinedDataset(ValerieDataset):
    @classmethod
    def from_raw(
        cls,
        datasets=[
            Phase2Dataset,
            Phase1Dataset,
            FakeNewsTop50Dataset,
            FakeNewsKaggleDataset,
            FakeNewsNetDataset,
            GeorgeMcIntireDataset,
            ISOTDataset,
            LiarDataset,
            MrisdalDataset,
        ],
    ):
        datasets = [dataset.from_raw() for dataset in datasets]
        return cls(combine_datasets_claims(datasets))


def combine_datasets_claims(datasets):
    claims_lists = [dataset.claims for dataset in datasets]
    logging_names = [dataset.__class__.__name__ for dataset in datasets]
    return combine_claims(claims_lists, logging_names=logging_names)


name_to_dataset = {
    Phase1Dataset.__name__: Phase1Dataset,
    Phase2Dataset.__name__: Phase2Dataset,
    Phase2DisjointDataset.__name__: Phase2DisjointDataset,
    Phase2TrialDataset.__name__: Phase2TrialDataset,
    Phase2ValidationDataset.__name__: Phase2ValidationDataset,
    FakeNewsTop50Dataset.__name__: FakeNewsTop50Dataset,
    FakeNewsKaggleDataset.__name__: FakeNewsKaggleDataset,
    FakeNewsNetDataset.__name__: FakeNewsNetDataset,
    GeorgeMcIntireDataset.__name__: GeorgeMcIntireDataset,
    ISOTDataset.__name__: ISOTDataset,
    LiarDataset.__name__: LiarDataset,
    MrisdalDataset.__name__: MrisdalDataset,
    LeadersDataset.__name__: LeadersDataset,
    Phase2CombinedDataset.__name__: Phase2CombinedDataset,
    CombinedDataset.__name__: CombinedDataset,
}
