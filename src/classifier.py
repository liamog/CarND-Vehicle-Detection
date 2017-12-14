'''Vehicle classifier module'''
import os
import pathlib
import tempfile

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
            with tempfile.NamedTemporaryFile() as temp:
                self._parameter_search_results = temp.name
        else:
            self._trained_model_filename = config.RESULTS_FOLDER + "/model/trained_model.p"
            self._parameter_search_results = config.RESULTS_FOLDER + \
                "/model/parameter_search_results.p"
            self._trained_model_scaler_filename = config.RESULTS_FOLDER + \
                "/model/trained_model_scaler.p"
            pathlib.Path(config.RESULTS_FOLDER + "/model").mkdir(parents=True, exist_ok=True)

        self.clf = None
        self.x_scaler = None

        if os.path.isfile(self._trained_model_filename):
            self.clf = joblib.load(self._trained_model_filename)
            print(self.clf)

        if os.path.isfile(self._trained_model_scaler_filename):
            self.x_scaler = joblib.load(self._trained_model_scaler_filename)

        if self.clf is None or self.x_scaler is None:
            self._train()

    def _train(self):
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
            img = cv2.imread(filename)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_vehicles.append(self._extract_img_features(img_rgb))
            y_vehicles.append(1)

        for filename in nonvehicle_files:
            img = cv2.imread(filename)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_nonvehicles.append(self._extract_img_features(img_rgb))
            y_nonvehicles.append(0)

        assert x_vehicles
        assert x_nonvehicles

        x_unscaled = np.concatenate((x_vehicles, x_nonvehicles))
        y_full = np.concatenate((y_vehicles, y_nonvehicles))

        x_full = np.vstack(x_unscaled).astype(np.float64)

        self.x_scaler = StandardScaler().fit(x_full)
        scaled_x = self.x_scaler.transform(x_full)

        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y_full, test_size=0.2, random_state=rand_state)

        svr = svm.SVC()
        gs = GridSearchCV(svr, config.PARAM_GRID, n_jobs=-1)
        gs.fit(x_train, y_train)
        print(gs.cv_results_)
        print(gs.best_estimator_)
        joblib.dump(gs.cv_results_, self._parameter_search_results)

        self.clf = gs.best_estimator_

        y_predict = self.clf.predict(x_test)

        self.accuracy = accuracy_score(y_test, y_predict)
        print("accuracy = {}".format(self.accuracy))

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
        hog = pp.extract_hog_features(img_for_hog)

        if config.USE_SPATIAL or config.USE_COLOR_HIST:
            other = pp.extract_other_features(img)
            return np.concatenate((hog, other))
        return hog
