"""
Image Dataset Classes

XXX: doc
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: Proprietary

# -- Imports
import os
from os import path
import csv
import numpy as np
import logging

#from pythor3 import io # XXX: get io back from pythor2


class ImageDatasetCSV(object):

    def __init__(self, dir, csv_fname):

        csvr = csv.reader(open(csv_fname))
        rows = [row for row in csvr]
        ncols = len(rows[0])

        assert ncols == 3

        # train images
        train_fnames = [row[0]
                        for row in rows
                        if row[2] == "train"]
        train_fnames = [path.join(dir, fname)
                        for fname in train_fnames]
        train_labels = np.array([row[1]
                                 for row in rows
                                 if row[2] == "train"])
        assert np.unique(train_labels).size > 1
        ntrain = len(train_fnames)

        # test images
        test_fnames = [row[0]
                       for row in rows
                       if row[2] == "test"]
        test_fnames = [path.join(dir, fname)
                       for fname in test_fnames]
        test_labels = np.array([row[1]
                                for row in rows
                                if row[2] == "test"])
        assert np.unique(train_labels).size > 1
        ntest = len(test_fnames)

        self.train_fnames = train_fnames
        self.train_labels = train_labels
        self.ntrain = ntrain

        self.test_fnames = test_fnames
        self.test_labels = test_labels
        self.ntest = ntest

    def load_train_data(self):

        logging.info(">>> Load train data...")
        return [io.imread(fn)
                for fn in self.train_fnames], self.train_labels

    def load_test_data(self):

        logging.info(">>> Load test data...")
        return [io.imread(fn)
                for fn in self.test_fnames], self.test_labels


class ImagePairDatasetCSV(object):

    def __init__(self, dir, csv_fname):

        csvr = csv.reader(open(csv_fname))
        rows = [row for row in csvr]
        ncols = len(rows[0])

        assert ncols == 4

        # train images
        train_fnames = [row[:2]
                        for row in rows
                        if row[3] == "train"]
        train_fnames = [(path.join(dir, fname1),
                         path.join(dir, fname2))
                        for fname1, fname2 in train_fnames]
        train_labels = np.array([str(row[2])
                                 for row in rows
                                 if row[3] == "train"])
        assert np.unique(train_labels).size > 1
        ntrain = len(train_fnames)

        # test images
        test_fnames = [row[:2]
                       for row in rows
                       if row[3] == "test"]
        test_fnames = [(path.join(dir, fname1),
                        path.join(dir, fname2))
                       for fname1, fname2 in test_fnames]
        test_labels = np.array([str(row[2])
                                for row in rows
                                if row[3] == "test"])
        assert np.unique(train_labels).size > 1
        ntest = len(test_fnames)

        self.train_fnames = train_fnames
        self.train_labels = train_labels
        self.ntrain = ntrain

        self.test_fnames = test_fnames
        self.test_labels = test_labels
        self.ntest = ntest

    def load_train_data(self):

        logging.info(">>> Load train data...")
        return [map(io.imread, fns)
                for fns in self.train_fnames], self.train_labels

    def load_test_data(self):

        logging.info(">>> Load test data...")
        return [map(io.imread, fns)
                for fns in self.test_fnames], self.test_labels


DEFAULT_EXTENSIONS = ['.png', '.jpg']

class ImageDatasetDir(object):

    def __init__(self, thedir, extensions=DEFAULT_EXTENSIONS):

        # -- get image filenames
        img_path = path.abspath(thedir)
        print "Image source:", img_path

        # navigate tree structure and collect a list of files to process
        if not path.isdir(img_path):
            raise ValueError, "%s is not a directory" % (img_path)
        tree = os.walk(img_path)
        fnames = []
        categories = tree.next()[1]
        for root, dirs, files in tree:
            if dirs != []:
                msgs = ["invalid image tree structure:"]
                for d in dirs:
                    msgs += ["  "+"/".join([root, d])]
                msg = "\n".join(msgs)
                raise Exception, msg
            fnames += [ root+'/'+f for f in files
                       if os.path.splitext(f)[-1] in extensions ]
        fnames.sort()
        print len(categories), "categories found:"
        print categories
        labels = [path.basename(path.split(fname)[-2]) for fname in fnames]

        self.fnames = fnames
        self.labels = labels

    def load_full_data(self):

        logging.info(">>> Load full data...")
        return [io.imread(fn)
                for fn in self.fnames], self.labels


