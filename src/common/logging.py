import logging, sys

def get_logger(name: str = "olist") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s | %(name)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger