import unittest
import numpy as np
from matplotlib import pyplot as plt

import main
from classifiers import NMC
from loaders import DataLoaderMNIST
from main import plot_ten_digits



class TestNMC(unittest.TestCase):

    @staticmethod
    def _createDigitCopies(x, y, numCopies, digit):
        """
        This function searches the "digit" in the y array.
        Once found, it saves its index.
        The index is then used to select the relative figure from the x array (grey scale data).
        The selected figure is then copied numCopies times and returned.

        """
        n_samples = x.shape[0]
        idx = list(range(0, n_samples))  # [0 1 ... 999]  np.linspace

        i = 0
        while i < n_samples:
            #search the "digit"
            if digit == y[i]:
                break
            i = i + 1

        #make some copies!
        xtr = np.zeros((numCopies, x.shape[1]))
        ytr = np.zeros((numCopies,), dtype=np.int32)
        for copyIdx in range(0, numCopies):
            xtr[copyIdx, :] = np.array(x[i, :]).copy()
            ytr[copyIdx] = y[i]

        xts = xtr.copy()
        yts = ytr.copy()

        #print(xtr)
        #print(ytr)
        #print(xts)
        #print(yts)

        return xtr, ytr, xts, yts

    def setUp(self):
        self.x0 = np.zeros(shape=(100, 10))
        self.y0 = np.zeros(shape=(50,))
        self.clf = NMC()


        filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"
        data_loader = DataLoaderMNIST(filename=filename, n_samples=10000)

        x, y = data_loader.load_data()
        #plot_ten_digits(x, y)
        #plt.show()

        # test only work with figure "0", see how "main.fit" create the centroid array (E.g. you can't create an
        # array full of just 2's as a training set and then pass it to the fit function; it will break the
        # centroid array, because "k" in main.fit will not index the class "k" anymore! The for will just go from
        # range(0, (num_classes) 1), so k will just take the value 0, corresponding to class/digit "0",
        # even with xtr full of 2's!!! Of course, the function will try to look for ytr == k, so ytr == 0, but it will
        # not find a single "0" and will not even bother to go look after the class "1", let alone "2"!!!
        # CENTROIDS WILL BE EMPTY!!
        #
        # FIT ONLY WORKS IF YOU PASS A TRAINING SET FULL OF DIGITS (0 through 9!)
        self.xtr, self.ytr, self.xts, self.yts = self._createDigitCopies(x, y, 1000, 0)
        self.centroids = np.zeros(shape=(1, self.xtr.shape[1]))

    def test_fit(self):
        self.assertRaises(TypeError, self.clf.fit, xtr=None, ytr=None)
        self.assertRaises(ValueError, self.clf.fit,
                          xtr=np.zeros(shape=(100, 10)),
                          ytr=np.zeros(shape=(50,)))


        #plot_ten_digits(self.xtr, self.ytr)
        #plt.show()

        self.centroids = main.fit(self.xtr, self.ytr)

        # show the centroid for the figure
        plt.imshow(self.centroids[0, :].reshape(28, 28), cmap='gray')
        plt.show()

        # if we provide the same figure over and over again for training, the mean should be equal to the normal figure!
        self.assertTrue(np.array_equal(self.centroids[0, :], self.xtr[0, :]))

    def test_predict(self):
        prediction = main.predict(self.xts, self.centroids)
        self.assertTrue(np.array_equal(prediction, self.yts))



