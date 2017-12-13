import unittest
import os
from classifier import Classifier

class TestClassifier(unittest.TestCase):

    def test_train(self):
        classifier = Classifier(unit_test=True)
        self.assertGreaterEqual(classifier.accuracy , 0.97)
        self.assertTrue(os.path.isfile(classifier._trained_model_filename))
        acc = classifier.accuracy
        print(acc)

    def test_model(self):
        classifier = Classifier(unit_test=True)

if __name__ == '__main__':
    unittest.main()
