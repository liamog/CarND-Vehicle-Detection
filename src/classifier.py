'''Vehicle classifier module'''
import os
import pathlib
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
            with tempfile.NamedTemporaryFile() as temp:
                self._trained_model_scaler_filename = temp.name
        else:
            self._trained_model_filename = "cached/trained_model.p"
            self._trained_model_scaler_filename = "cached/trained_model_scaler.p"
            pathlib.Path("cached").mkdir(parents=True, exist_ok=True)

        self.clf = None
        self.x_scaler = None

        if os.path.isfile(self._trained_model_filename):
            self.clf = joblib.load(self._trained_model_filename)

        if os.path.isfile(self._trained_model_scaler_filename):
            self.x_scaler = joblib.load(self._trained_model_scaler_filename)

        if self.clf is None or self.x_scaler is None:
            self._train()

    def _train(self):
        self.clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.001)

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
            x_vehicles.append(self._extract_img_features(img_rgb))
            y_vehicles.append(1)

        for filename in nonvehicle_files:
            img_rgb = cv2.imread(filename)
            x_nonvehicles.append(self._extract_img_features(img_rgb))
            y_nonvehicles.append(0)

        assert x_vehicles
        assert x_nonvehicles

        x_unscaled = np.concatenate((x_vehicles, x_nonvehicles))
        y_full = np.concatenate((y_vehicles, y_nonvehicles))

        x_full = np.vstack(x_unscaled).astype(np.float64)

        self.x_scaler = StandardScaler().fit(x_full)
        scaled_x = self.x_scaler.transform(x_full)

        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y_full, test_size=0.33, random_state=42)

        self.clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.001, verbose=True)
        self.clf.fit(x_train, y_train)

        y_predict = self.clf.predict(x_test)

        self.accuracy = accuracy_score(y_test, y_predict)

        joblib.dump(self.clf, self._trained_model_filename)
        joblib.dump(self.x_scaler, self._trained_model_scaler_filename)

    def predict(self, features):
        """
        Predict if a vehicle is present in the set of features
            :param self:
            :param features:
        """
        scaled_features = self.x_scaler.transform(features.reshape(1, -1))
        return self.clf.predict(scaled_features)

    def _extract_img_features(self, img):
        img_for_hog = pp.convert_img_for_hog(img)
        return pp.extract_hog_features(img_for_hog)
