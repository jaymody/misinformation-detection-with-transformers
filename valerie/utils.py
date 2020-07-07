"""Utility functions."""
import logging

from tqdm.auto import tqdm

_logger = logging.getLogger(__name__)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(logfile=None):
    """Gets a nicely formatted logger."""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    sh = TqdmHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def stats(values, plot=False):
    import statistics

    # workaround for multiple modes
    try:
        mode = statistics.mode(values)
    except:
        mode = None

    d = {
        "len": len(values),
        "max": max(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "mode": mode,
        "stdev": statistics.stdev(values),
    }

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.distplot(values)
        plt.show()

    return d
