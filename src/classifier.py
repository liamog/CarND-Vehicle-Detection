'''Vehicle classifier module'''
import os
import tempfile

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import cv2
import preprocess as pp

#pylint ignore-too-few-public-methods
class Classifier():
    """
    Classifier class.
    """
    def __init__(self, unit_test=False):
        self._unit_test = unit_test

        if self._unit_test:
            with tempfile.NamedTemporaryFile() as temp:
                self._trained_model_filename = temp.name
        else:
            self._trained_model_filename = "../cached/trained_model.p"

        self.clf = None

        if os.path.isfile(self._trained_model_filename):
            self.clf = joblib.load(self._trained_model_filename)
        if self.clf is None:
            self._train()

    def _train(self):
        self.clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.001)
        # The arg argument for walk, and subsequently ext for step
        exten = '.png'
        vehicle_files = []
        nonvehicle_files = []
        if self._unit_test:
            vehicle_folder = 'training_data/unit_test_data/vehicles'
            non_vehicle_folder = 'training_data/unit_test_data/non-vehicles'
        else:
            vehicle_folder = 'training_data/vehicles'
            non_vehicle_folder = 'training_data/non-vehicles'

        #pylint: disable=unused-variable
        for dirpath, dirnames, files in os.walk(vehicle_folder):
            for name in files:
                if name.lower().endswith(exten):
                    vehicle_files.append(os.path.join(dirpath, name))
        #pylint: disable=unused-variable
        for dirpath, dirnames, files in os.walk(non_vehicle_folder):
            for name in files:
                if name.lower().endswith(exten):
                    nonvehicle_files.append(os.path.join(dirpath, name))

        x_vehicles = []
        y_vehicles = []
        x_nonvehicles = []
        y_nonvehicles = []

        for filename in vehicle_files:
            img_rgb = cv2.imread(filename)

            channel = pp.select_channel(img_rgb, config.HOG_COLOR_CHANNEL)
            features = pp.extract_hog_features(channel)
            x_vehicles.append(features)
            y_vehicles.append(1)

        for filename in nonvehicle_files:
            img_rgb = cv2.imread(filename)

            channel = pp.select_channel(img_rgb, config.HOG_COLOR_CHANNEL)
            features = pp.extract_hog_features(channel)
            x_nonvehicles.append(features)
            y_nonvehicles.append(0)

        assert x_vehicles
        assert x_nonvehicles

        x_unscaled = np.concatenate((x_vehicles, x_nonvehicles))
        y_full = np.concatenate((y_vehicles, y_nonvehicles))

        x_full = np.vstack(x_unscaled).astype(np.float64)

        x_scaler = StandardScaler().fit(x_full)
        scaled_x = x_scaler.transform(x_full)

        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y_full, test_size=0.33, random_state=42)

        clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.001, verbose=True)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)

        self.accuracy = accuracy_score(y_test, y_predict)

        joblib.dump(clf, self._trained_model_filename)

    def predict(self, features):
        """
        Predict if a vehicle is present in the set of features
            :param self:
            :param features:
        """
        return self.clf.predict(features)
