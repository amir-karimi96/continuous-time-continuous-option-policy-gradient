import numpy as np
from typing import Dict

# from herl.config import np_type
from dict_serializable import DictSerializable
np_type = np.float64

class Variable:
    """
    This represent a variable in a specific domain (e.g., a state in a RL problem,
    the angle in the pendulum environment ect.)
    We assume the variable to be float, but we might generalise this in next development.
    """

    def __init__(self, name, length, description="", latex_symbol=""):
        """
        :param name: The name of the variable. This name will be used in the code to refer to the variable.
        :type name: str
        :param length: The dimensionality of the variable. E.g., lenght=3 if the variable represent a 3d-position.
        :type length: int
        :param description: The usere might want to associate a description of the variable.
        :type description: str
        :param latex_symbol: For plot purposes, we can associate to the variable some symbol in latex code,
        such as $\theta$, $x$...
        :type latex_symbol: str
        """
        self.name = name
        self.length = length
        self.description = description
        self.latex_symbol = latex_symbol
        self.location = None
        self.displacement = None

    def assign(self, location):
        """
        Assign the variable a position in the dataset.
        :param location: The position of the variable in the dataset.
        :type location: int
        :return: None.
        """
        self.location = location
        self.displacement = self.location + self.length


class Domain:

    """
    This class describes a domain, such as a reinforcement learning proble, a state description, ...
    The domain is defined by a ordered set of variables.
    """

    def __init__(self, *variables):
        """
        Create a domain with some variable. We can also add add variable later to the domain with the method
        `self.add_variable`.
        :param variables: variable to add to the domain
        :type variables: Variable
        """
        self.size = 0
        self.variables = []
        self.variable_dict = {}
        for var in variables:
            self.add_variable(var)

    def copy(self):
        return Domain(*self.variables)

    def add_variable(self, variable):
        """
        Add a variable to the domain.
        :param variable: The variable to add.
        :type variable: Variable
        :return: None
        """
        for v in self.variables:
            if v.name == variable.name:
                raise Exception("Variable %s is already contained in the domanin." % variable.name)
        variable.assign(self.size)
        self.variables.append(variable)
        self.variable_dict[variable.name] = variable
        self.size += variable.length

    def get_variable(self, name):
        """
        Get a variable given its name.
        :param name: Name of the variable.
        :return: The variable (if exists a variable with the specified name).
        :rtype: Variable
        """
        return self.variable_dict[name]


class Dataset(DictSerializable):

    load_fn = DictSerializable.get_numpy_load()
    """
    This class defines a generic dataset.
    """
    def __init__(self, domain: Domain, n_max_row: int=int(10E6)):
        """
        A dataset is specified on a specific Domain. For example, for the Iris dataset, the domain would be something
        like Domain(Variable("sepal_l", 1), Variable("sepal_w", 1), Variable("petal_l", 1), Variabble("petal_w", 1),
        Variable("class", 1)).
        The dataset has a fixed maximum size to allow efficient allocation and data retrival.
        :param domain: The domain of the data.
        :param n_max_row: Number maximum of rows. The data's structure is a circular buffer, therefore if the user add
        more data than the n_max row, the first data will be overwritten.
        """
        self.domain = domain
        self.memory = np.zeros((n_max_row, domain.size), dtype=np_type)
        self.max_size = n_max_row
        self.real_size = 0
        self.pointer = 0
        self.indexes = np.arange(0, self.max_size)
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())

    @staticmethod
    def load_from_dict(**kwargs):
        dataset = Dataset(kwargs["domain"], kwargs["memory"].shape[0])
        dataset.memory = kwargs["memory"]
        dataset.pointer = kwargs["pointer"]
        dataset.real_size = kwargs["real_size"]
        return dataset

    @staticmethod
    def load(file_name: str, domain: Domain):
        """
        :param file_name:
        :param domain:
        :return:
        """
        file = Dataset.load_fn(file_name)
        return Dataset.load_from_dict(domain=domain, **file)

    def _get_dict(self):
        return dict(memory=self.memory, real_size=self.real_size, pointer=self.pointer)

    def notify(self, **kargs: np.ndarray) -> bool:
        """
        Insert a new row in the dataset. Each kwarg represent a variable associated to the Variable.
        E.g., for the iris dataset one should run
        `self.notify(sepal_l=1., sepal_w=1., petal_l=0.2, petal_w=0.3)`
        :param kargs: The value associated to the variable
        :return: Return True if there was a "overflow" and some data has been rewritten (like a circular buffer)
        """
        old_pointer = self.pointer
        for k, v in kargs.items():
            variable = self.domain.variable_dict[k]
            self.memory[self.pointer, variable.location:variable.location+variable.length] = v
        self.real_size = min(self.real_size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        return self.pointer < old_pointer

    def notify_batch(self, **kargs: np.ndarray) -> bool:
        """
        Insert a new row in the dataset. Each kwarg represent a variable associated to the Variable.
        E.g., for the iris dataset one should run
        `self.notify(sepal_l=1., sepal_w=1., petal_l=0.2, petal_w=0.3)`
        :param kargs: The value associated to the variable
        :return: Return True if there was a "overflow" and some data has been rewritten (like a circular buffer)
        """
        old_pointer = self.pointer
        batch_size = None
        end_size = None
        init_size = None
        for k, v in kargs.items():
            if batch_size is None:
                batch_size = v.shape[0]
                if batch_size > self.max_size:
                    raise Exception("The batch must have smaller dimension of the dataset.")
                end_size = min(batch_size, self.max_size - self.real_size)
                init_size = batch_size - end_size
            if batch_size != v.shape[0]:
                raise Exception("All the variable must have batch of same size!")
            variable = self.domain.variable_dict[k]
            self.memory[self.pointer:self.pointer+end_size,
                variable.location:variable.location+variable.length] = v[:end_size, :]
            if init_size != 0:
                self.memory[0:init_size,
                    variable.location:variable.location+variable.length] = v[-init_size:, :]
        self.real_size = min(self.real_size + batch_size, self.max_size)
        self.pointer = (self.pointer + batch_size) % self.max_size
        return self.pointer < old_pointer or batch_size > self.max_size

    def get_minibatch(self, size: int=128):
        """
        Retrive a random (mini)batch of data.
        :param size: The size of the batch (which should be less than the data already inserted in the database)
        :type size: int
        :return: Dataset batch
        :rtype: dict
        """
        if size > self.real_size:
            raise("You are requesting more samples than the samples stored")

        indx = np.random.choice(self.indexes[:self.real_size], size=size, replace=False)
        result = self.memory[indx, :]
        return {k: result[:, v.location:v.location+v.length] for k, v in self.domain.variable_dict.items()}

    def get_minibatch_sampling_strategy(self, sampling_variable:str, size: int=128):
        """
        Retrive a random (mini)batch of data.
        :param size: The size of the batch (which should be less than the data already inserted in the database)
        :type size: int
        :return: Dataset batch
        :rtype: dict
        """
        if size > self.real_size:
            raise("You are requesting more samples than the samples stored")
        variable = self.domain.variable_dict[sampling_variable]
        weights = self.memory[self.indexes[:self.real_size], variable.location:variable.location + variable.length]
        weights = weights.ravel()
        indx = np.random.choice(self.indexes[:self.real_size], size=size, replace=True, p=weights/np.sum(weights))
        result = self.memory[indx, :]
        return {k: result[:, v.location:v.location+v.length] for k, v in self.domain.variable_dict.items()}

    def set(self, memory: np.ndarray):
        if memory.shape[1] != self.domain.size:
            raise("Wrong format!")
        self.memory = memory
        self.pointer = 0
        self.max_size = self.memory.shape[0]
        self.real_size = self.max_size

    def flush(self):
        """
        Empty the database
        :return: None.
        """
        self.memory = np.zeros((self.max_size, self.domain.size), dtype=np_type)
        self.real_size = 0
        self.pointer = 0

    def is_full(self):
        """
        Is the database full?
        :return:
        """
        return self.real_size == self.max_size

    def is_empty(self):
        """
        Is the database empty?
        :return:
        """
        return self.real_size == 0

    def get_sequential(self, indx_in: int, indx_end: int):
        if indx_in < indx_end:
            return self.memory[indx_in:indx_end]
        else:
            return np.concatenate([self.memory[indx_in:], self.memory[:indx_end]], axis=0)

    def get_full(self) -> Dict[str, np.ndarray]:
        return self._get_full(self.memory)

    def _get_full(self, memory: np.ndarray) -> Dict[str, np.ndarray]:
        result = memory[:self.real_size, :]
        return {k: result[:, v.location:v.location + v.length] for k, v in self.domain.variable_dict.items()}

    def where(self, **kwargs):
        """
        We obtain all the data satisfying a certain condition (all the conditions are in AND).
        :param kwargs: Lambda function expressing the condition. the name of the parameter corresponds to the name
        of the dataset.
        :type kwargs: lambda
        :return:
        """
        memory = self.memory[:self.real_size]

        for name, cond in kwargs.items():
            var = self.domain.get_variable(name)
            indx = np.argwhere(cond(memory[:,var.location:var.displacement]))[:, 0]
            memory = memory[indx]

        return self._get_full(memory)


class RLDataset(Dataset):

    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, domain: Domain, n_max_row=int(10E6)):
        Dataset.__init__(self, domain, n_max_row)

        self._trajectory_based = False
        self._trajectory_open = False
        self._trajectory_index = []

    def notify_new_trajectory(self):
        """
        Before starting a new trajectory, notify that you are starting a new trajectory.
        Each trajectory
        :return:
        """
        self._trajectory_open = True
        self._trajectory_based = True
        self._trajectory_index.append(self.pointer)

    def notify_end_trajectory_collection(self):
        """
        Close the current trajectory, without opening a new trajectory.
        :return:
        """
        self._trajectory_index.append(self.pointer)
        self._trajectory_open = False

    def get_trajectory_list(self):
        if self._trajectory_based:
            if len(self._trajectory_index) > 1:
                return [self._get_full(self.get_sequential(indx_in, indx_end))
                 for indx_in, indx_end in zip(self._trajectory_index[:-1], self._trajectory_index[1:])], \
                       self._trajectory_open
            raise Exception("There must be at least one trajectory closed in order to return trajectories")
        raise Exception("This is not a trajectory-based dataset")

    def _get_dict(self):
        return dict(memory=self.memory, real_size=self.real_size, pointer=self.pointer,
                    trajectory_based=self._trajectory_based, trajectory_open=self._trajectory_open,
                    trajectory_index=self._trajectory_index)

    @staticmethod
    def load_from_dict(**kwargs):
        dataset = RLDataset(kwargs["domainr"], kwargs["memory"].shape[0])
        dataset.memory = kwargs["memory"]
        dataset.pointer = kwargs["pointer"]
        dataset.real_size = kwargs["real_size"]
        dataset._trajectory_based = kwargs["trajectory_based"]
        dataset._trajectory_open = kwargs["trajectory_open"]
        dataset._trajectory_index = kwargs["trajectory_index"]
        return dataset

    @staticmethod
    def load(file_name: str, domain: Domain):
        """
        Load a dataset.
        :param file_name: the name of the file we want to load.
        :param environment_descriptor: The environment on which the dataset is defined.
        :return:
        """
        file = Dataset.load_fn(file_name)
        return Dataset.load_from_dict(domain=domain, **file)

    @property
    def train_ds(self):
        print("train_ds is deprecated. Please remove it from your code. You don't need it anymore.")
        return self


class MLDataset(DictSerializable):

    load_fn = DictSerializable.get_numpy_load()

    """
    This class represent a typical database for machine learning. It allows the possibility to separate data onto two
    sets, one for training and one for validating (or test).
    """
    def __init__(self, domain, n_max_row=int(10E6), validation=0.1):
        """
        Likewise the classic dataset, we are constructing a database on a given domain.
        :param domain: Domain of the dataset
        :type domain: Domain
        :param n_max_row: Size of the database
        :type n_max_row: int
        :param validation: Portion of the data should be in the validation set (between 0 and 1).
        :type validation: float
        """
        self.domain = domain
        self.train_ds = Dataset(domain, n_max_row - int(n_max_row*validation))
        self.validation_ds = Dataset(domain, int(n_max_row*validation))
        self.validation = validation
        self.n_max_row = n_max_row
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())

    def _get_dict(self):
        return dict(validation=self.validation,
                    train_ds=self.train_ds._get_dict(),
                    validation_ds=self.validation_ds._get_dict(),
                    n_max_row=self.n_max_row)

    @staticmethod
    def load_from_dict(**kwargs):
        domain = kwargs["domain"]
        dataset = MLDataset(domain, kwargs["n_max_row"], validation=kwargs["validation"])
        dataset.train_ds = Dataset.load_from_dict(domain=domain, **kwargs["train_ds"].item())
        dataset.validation_ds = Dataset.load_from_dict(domain=domain, **kwargs["validation_ds"].item())
        return dataset

    @staticmethod
    def load(file_name, domain):
        """
        :param file_name:
        :return:
        """
        file = MLDataset.load_fn(file_name)
        return MLDataset.load_from_dict(domain=domain, **file)

    def notify(self, **kwargs):
        """
        Insert a new row in the dataset. Each kwarg represent a variable associated to the Variable.
        E.g., for the iris dataset one should run
        `self.notify(sepal_l=1., sepal_w=1., petal_l=0.2, petal_w=0.3)`.
        The data will be randomly allocated to the training set or to the validation set, depending the
        validation probability specified in the constructor.
        :param kargs: The value associated to the variable
        :type kargs: np.ndarray
        :return: None.
        """
        if self.validation_ds.is_full():
            self.train_ds.notify(**kwargs)
        else:
            if np.random.uniform() < self.validation:
                self.validation_ds.notify(**kwargs)
            else:
                self.train_ds.notify(**kwargs)

    def get_minibatch(self, size, train=True):
        """
        Retrive a random (mini)batch of data. If train=True, the data will be retreived from the training set,
        or from the validation set otherwise.
        :param size: The size of the batch (which should be less than the data already inserted in the database)
        :type size: int
        :return: Dataset batch
        :rtype: dict
        """
        if train:
            return self.train_ds.get_minibatch(size)
        else:
            return self.validation_ds.get_minibatch(size)

    def flush_validation(self):
        """
        Flush only the validation set. (it might be useful to use the validation data only once)
        :return: None.
        """
        self.validation_ds.flush()

    def flush(self):
        """
        Empty both validation and training sets.
        :return: None.
        """
        self.validation_ds.flush()
        self.train_ds.flush()

    def is_full(self):
        """
        Check if both the sets are full.
        :return: None.
        """
        return self.validation_ds.is_full() and self.train_ds.is_full()


def create_ml_dataset(dataset, validation=0.1):
    """
    Create a proper ML dataset with train set and validation set given a normal dataset
    :param dataset: A classic database
    :type dataset: Dataset
    :return:  a MLDataset
    :rtype: MLDataset
    """
    train_size = int(dataset.real_size*(1-validation))
    validation_size = dataset.real_size - train_size

    indxs = np.arange(dataset.real_size)
    np.random.shuffle(indxs)
    train_indxs = indxs[:train_size]
    validation_indxs = indxs[train_size:train_size+validation_size]

    ret = MLDataset(dataset.domain, dataset.real_size)

    ret.train_ds.set(dataset.memory[train_indxs, :])
    ret.validation_ds.set(dataset.memory[validation_indxs, :])

    return ret