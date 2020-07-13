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


def get_logger(logfile=None, use_tqdm_handler=True):
    """Gets a nicely formatted logger."""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    if use_tqdm_handler:
        sh = TqdmHandler()
    else:
        sh = logging.StreamHandler()
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


def log_title(lg, text, length=80, character="."):
    text = " " + text + " "
    lg.info("")
    lg.info(character * length)
    lg.info(text.center(length, character))
    lg.info(character * length)
    lg.info("")


def titleize(text, length=80, character="-"):
    text = " " + text + " "
    return "\n\n\n{}\n{}\n{}\n\n\n".format(
        character * length, text.center(length, character), character * length
    )
