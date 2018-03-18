"""
Data management class.


"""

import numpy as np


class DataManager:

    _label_names = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.features = None
        self.labels = None
        self.onehot_labels = None
        self.initiate()

    @staticmethod
    def unpack_data(dataset):
        """
        :param dataset: dataset filename.
        :return: labels and features of the data. Labels as strings.
        """
        with np.load(dataset) as data:
            dataset_features = data["features"]
            dataset_labels = data["labels"]

        return dataset_features, dataset_labels

    @classmethod
    def label_count(cls, labels):
        """
        Labels are stored in the order that they are appended to the label_names.
        label_names is stored betweenn
        :param labels: labels of the dataset

        """

        label_names = []
        for label in labels:
            if label not in label_names:
                label_names.append(label)

        if cls._label_names is None:
            cls._label_names = label_names
        else:
            for label in label_names:
                if label not in cls._label_names:
                    print("Label ", label, " is not a registered label."
                          , " Please ensure that only one dataset is opened using this class.")

    @classmethod
    def one_hot(cls, labels):
        """
        Converts the said labels to one_hot encoded labels
        :param labels: Labels of the dataset
        :return: one_hot encoded labels, encoded using the order in _label_names
        """
        if cls._label_names is not None:
            one_hot_label_list = []
            for lb in range(len(labels)):
                num_labels = len(cls._label_names)
                one_hot_label = np.zeros(num_labels)

                for i in range(num_labels):
                    if cls._label_names[i] == labels[lb]:
                        one_hot_label[i] = 1
                        one_hot_label_list.append(one_hot_label)

            return np.asarray(one_hot_label_list)

        else:
            return None

    def initiate(self):
        """
        Call method to initiate all variables.
        """
        self.features, self.labels = self.unpack_data(self.dataset)
        self.label_count(self.labels)
        self.onehot_labels = self.one_hot(self.labels)

    # Getters
    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_onehot_labels(self):
        return self.onehot_labels

    def get_label_names(self):
        return self._label_names

    def get_features_shape(self):
        return self.get_features().shape

    def get_features_dtype(self):
        return self.get_features().dtype

    def get_labels_dtype(self, onehot=False):
        if onehot:
            return self.get_onehot_labels().dtype
        else:
            return self.get_labels().dtype

    def get_labels_shape(self, onehot=False):
        if onehot:
            return self.get_onehot_labels().shape
        else:
            return self.get_labels().shape

