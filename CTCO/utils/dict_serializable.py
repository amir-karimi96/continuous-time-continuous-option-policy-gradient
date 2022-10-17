import torch
import numpy as np


def _get_numpy_save(filename, **kwargs):
    return np.savez(filename, **kwargs)


def _get_numpy_load(filename):
    return np.load(filename, allow_pickle=True)


class DictSerializable:

    load_fn = None

    def __init__(self, save_fn):
        """
        Create a serializable object.
        :param save_fn: it is a function that takes a filename as a first argument and then **kwargs are values to save
        """
        self.save_fn = save_fn

    @staticmethod
    def get_numpy_save():
        return _get_numpy_save

    @staticmethod
    def get_numpy_load():
        return _get_numpy_load

    @staticmethod
    def get_torch_load():
        Exception("Not Implemented Yet")

    @staticmethod
    def get_torch_save():
        Exception("Not Implemented Yet")

    def _get_dict(self):
        raise Exception("Not Implemented")

    def save(self, file_name):
        self.save_fn(file_name, **self._get_dict())

    @staticmethod
    def load_from_dict(**kwargs):
        raise Exception("Not Implemented")