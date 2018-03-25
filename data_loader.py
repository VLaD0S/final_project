"""
ToDo: Tests, maybe tweak arguments so they look nicer.
ToDo: Add a function which specifies the image shape?
ToDo: Add save directory
-- "it ain't broke...but lets fix it anyway " --
"""

import os
import argparse
import random
from PIL import Image
import numpy as np
from data_management.data_holder import Data


def shuffler(data_list):
    # creating a random list of numbers
    size = list(range(data_list.get_size()))
    random.shuffle(size)

    # copying original data into three corresponding lists
    raw_data = data_list.get_data()
    raw_labels = data_list.get_labels()
    raw_paths = data_list.get_paths()

    # creating new Data object
    shuffled_data = Data()
    shuffled_data.initiate_shape(data_list.get_shape())

    # copying the original data in the random order into the new Data object
    for no in size:
        shuffled_data.append(raw_data[no], raw_labels[no], raw_paths[no])

    if shuffled_data.double_check():
        return shuffled_data
    else:
        print("Failed to shuffle the data...")
        return False


def pack_and_save(data_root, data_list, directory):
    filename = data_root.lower() + ".npz"
    if data_list.double_check():

        data = shuffler(data_list)

        features = data.get_data()
        labels = data.get_labels()

        if directory not in os.listdir("."):
            os.makedirs(directory)

        filename = os.path.join(directory, filename)
        np.savez(filename, features=features, labels=labels)

        print("Saved data as " + filename + ", of size " + str(data.get_size()) + ".")

    else:
        print("Failed to save the " + filename + ". Aborting...")
        return False


def split_save(data_root, data_list, test_size, validation_size, directory):
    """
    Splits the original dataset into Training data, Test data and Validation data.
    Saves the data at the end.
    :param, data_root: data root name, to be passed on.
    :param data_list: original dataset
    :param test_size: proportion of the data for the Test dataset
    :param validation_size: proportion of the data for the Validation dataset
    :param directory: directory where data will be saved
    :return:
    """

    # validation method

    # getting the labels and the end pointers of the data from each label
    labels, pointers = data_list.get_labelstats()
    shape = data_list.get_shape()

    # copying the data from the original data_list
    all_features = data_list.get_data()
    all_labels = data_list.get_labels()
    all_paths = data_list.get_paths()

    # creating the three datasets
    training_data = Data()
    test_data = Data()
    validation_data = Data()

    # initialising all the datasets with the original data
    training_data.initiate_shape(shape)
    test_data.initiate_shape(shape)
    validation_data.initiate_shape(shape)

    # ensure that the test data and the validation data are of viable proportions
    validate_sets(test_size, validation_size)

    # used to calculate where the label changes in the data.
    data_pointer = 0

    for no in list(range(len(pointers))):

        # calculates the amount of data in a label
        label_size = pointers[no] - data_pointer

        # determines a list of all the indexes of the items of a given label, and shuffles them
        label_keys = list(range((pointers[no] - label_size), pointers[no]))
        random.shuffle(label_keys)

        # calculate the number of test and validation elements
        label_test_size = int(label_size * test_size)
        label_validation_size = int(label_size * validation_size)

        # For each label, a proportion of the items are selected to be part of the test and validation sets.
        for item in list(range(len(label_keys))):

            # gets an "id" of an element within the label
            elem_id = label_keys[item]

            if label_test_size >= 0:
                test_data.append(all_features[elem_id], all_labels[elem_id], all_paths[elem_id])
                label_test_size = label_test_size - 1

            elif label_validation_size >= 0:
                validation_data.append(all_features[elem_id], all_labels[elem_id], all_paths[elem_id])
                label_validation_size = label_validation_size - 1

            else:
                training_data.append(all_features[elem_id], all_labels[elem_id], all_paths[elem_id])

        # resets the data pointer
        data_pointer = pointers[no]

        # - logging
        print("Done label "+labels[no])

    pack_and_save(str(data_root + "_test"), test_data, directory)
    pack_and_save(str(data_root + "_validation"), validation_data, directory)
    pack_and_save(str(data_root + "_training"), training_data, directory)


def validate_sets(test_size, validation_size):
    """
    Checks if the test and validation sets are of viable proportions.
    :param test_size:
    :param validation_size:
    :return: True if the test and validation sizes are viable, false otherwise
    """
    validation = True
    error = "Ensure that the selected sizes are between 0 and 1, and add up to less than 1."

    if test_size <= 0 or validation <= 0:
        validation = False

    if (test_size + validation_size) >= 1:
        validation = False

    if not validation:
        print(error)

    return validation


def main(data_root, test_size, validation_size, directory="data"):
    """
    > Goes through the data folder, loading a Data() object with the image data and label.
    > Exports a file called <data_root>.npz containing all the data.
    :param data_root: path to main data folder
    :param test_size: proportion of data allocated to the test set
    :param validation_size: proportion of data allocated to the validation set
    :param directory: directory where data will be saved
    """

    data_list = Data()

    # lists containing the labels and the position at which they change in the data.
    label_list = []
    label_point = []
    label_stats = (label_list, label_point)

    # counters
    label_total = 0
    image_total = 0

    # iterate through root folder
    for label in os.listdir(data_root):
        label_path = os.path.join(data_root, label)
        image_counter = 0

        # iterate through each sub folder (label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            image_opener = Image.open(image_path)

            # ToDo: parameter to resize data/ move data resize to model?
            image_opener = image_opener.resize((160, 160), Image.ANTIALIAS)
            image_data = np.asarray(image_opener, dtype="float32")

            # checks if data is initiated with a particular data size. Uses the first image to define
            if not data_list.check_initiate():
                data_list.initiate_shape(image_data.shape)

            # save image inside the data_list
            data_list.append(image_data, label, image_path)

            # - counter
            image_counter = image_counter + 1

        # - counter
        label_total = label_total + 1
        image_total = image_total + image_counter

        label_list.append(label)
        label_point.append(image_total)

    # passes a list of labels and their corresponding end index in the list.
    data_list.set_labelstats(label_stats)

    split_save(data_root, data_list, test_size, validation_size, directory)


if __name__ == '__main__':

    # argument parser

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Main data folder. The sub-folders of this folder are the labels")
    parser.add_argument("--test_size", type=float, help="Proportion of the data to be allocated for testing,"
                        "as float between 0 and 1.")
    parser.add_argument("--valid_size", type=float,
                        help="Proportion of the data to be allocated for validation,"
                             "as float between 0 and 1.")
#    parser.add_argument("--dest_folder", help="Directory where models will be saved.")

    args = parser.parse_args()

    if args.data:

        if args.test_size and args.valid_size:

            if (args.test_size >= float(0)) and (args.valid_size >= float(0)):

                if validate_sets(args.test_size, args.valid_size):

                    main(args.data, args.test_size, args.valid_size, )

                else:
                    print("Failed to validate the sets")
            else:
                print("Please specify a valid testsize and validsize combination.")
        else:
            print("Please specify a test and validation proportion.")
    else:

        print("Please specify a data source.")
