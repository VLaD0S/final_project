
"""
Object to help organise the data in data_loader.py
"""


class Data:
    """
    Class which holds two arrays (which are treated like stacks) which are used to hold data and labels
    """
    def __init__(self, ):

        self.data_shape = ()
        self.data = []
        self.labels = []
        self.initiate = False

        # optional
        self.paths = []
        self.label_stats = None

    # Mandatory
    def initiate_shape(self, shape):
        """
        Initialises data shape, i.e (160, 160 , 3) for an RGB image of resolution 160x160
        """
        self.data_shape = shape
        self.initiate = True

    # optional
    def set_labelstats(self, label_stats):
        self.label_stats = label_stats

    def append(self, data, label, path="|nopath|"):
        """
        Use to append the data and its corresponding label to the arrays.
        :param data: data as multidimensional tuple
        :param label: label as string(or anything else)
        :param path: path of the image/data component
        """
        if self.validate(data):
            self.data.append(data)
            self.labels.append(label)
            self.paths.append(path)

    # Checker methods
    def validate(self, data):
        """
        Checks if data is suitable to be appended.
        :param data: data to be inputted
        :return: True if data is of the right shape, and the data arrays are of the same length. False otherwise.
        """
        if not self.initiate:
            print("You must use Data.initiate_shape(data_shape) to specify the input data shape")
            return False

        if data.shape == self.data_shape:
            if len(self.data) == len(self.labels):
                return True
            else:
                print("Data and label datasets are not of the same size.")
                return False
        else:
            print("Required data shape:" + str(self.data_shape) + ", Received: " + data.shape)
            return False

    def double_check(self):
        for i in range(len(self.data)):

            if not self.data[i].shape == self.data_shape:
                print("Error : image at " + self.paths[i] + " not of the size " + str(self.data_shape))
                return False
        return True

    # Getters
    def check_initiate(self):
        return self.initiate

    def get_size(self):
        """
        :return: Dataset sizes
        """
        ds = len(self.data)
        return ds

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def get_shape(self):
        return self.data_shape

    def get_paths(self):
        return self.paths

    def get_labelstats(self):
        return self.label_stats
