"""The common module contains common functions and classes used by the other modules.
"""
import yaml
from pathlib import Path


def load_yml(_input: str):
    with open(_input, "r") as f:
        args = yaml.safe_load(f)

    # #tests for later maybe
    # assert a1 == a2, "PAth and str are not same"
    return args

# test = load_yml("C:\\fao-models\\runc-resnet-epochs20-batch64-lr001-seed5-lrdecay5-tfrecords-all.yml")
# print(test)

