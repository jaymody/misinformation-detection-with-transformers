"""Utility functions."""
import logging

import tqdm

_logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(logfile=None):
    """Gets a nicely formatted logger."""
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [sh, TqdmLoggingHandler()]

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.handlers.append(fh)

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
