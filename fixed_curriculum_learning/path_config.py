import os


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PACKAGE_ROOT, "data")


def data_subdir(*parts):
    return os.path.join(DATA_ROOT, *parts)
