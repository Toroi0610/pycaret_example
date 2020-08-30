import os
from shutil import copy2
import pandas as pd


setting = pd.read_csv("setting_automl.csv", index_col=0)
mode = setting.loc["mode", "property0"]
# Importing pycaret
if mode == "classification":
    from pycaret.classification import *
elif mode  == "regression":
    from pycaret.regression import *


class Pycaret_CLI:

    def __init__(self, path_setting_file):

        setting = pd.read_csv(path_setting_file, index_col=0)
        self.mode = setting.loc["mode", "property0"]
        self.path_train_file = setting.loc["path_training_file", "property0"]
        self.path_test_file = setting.loc["path_test_file", "property0"]
        # self.preprocessing = dict(setting.loc["preprocessing", "property0"])
        self.target = setting.loc["target", "property0"]
        self.module = setting.loc["module", "property0"]
        self.model_list = setting.loc["models"].values.tolist()
        self.metric = setting.loc["metric", "property0"]
        self.exp_name = setting.loc["exp_name", "property0"]

        os.makedirs(self.exp_name, exist_ok=True)
        os.makedirs(self.exp_name+"/model", exist_ok=True)
        os.makedirs(self.exp_name+"/data", exist_ok=True)
        os.makedirs(self.exp_name+"/predict", exist_ok=True)
        os.makedirs(self.exp_name+"/setting", exist_ok=True)
        copy2(path_setting_file, self.exp_name+"/setting")
        copy2(self.path_train_file, self.exp_name+"/data")
        copy2(self.path_test_file, self.exp_name+"/data")




    def load_data(self):
        """[summary]
        Load training, test data. Their path is written in setting file.

        Returns:
            [type]: [description]
        """
        train = pd.read_csv(self.path_train_file)
        test = pd.read_csv(self.path_test_file)

        return train, test


    def setup_automl_env(self, train):

        print("SETUP EXPERIMENTS")
        exp_0 = setup(data=train, target=self.target,
                      html=False, silent=True)
        print("Finished !!")

        return exp_0


    def training_model(self):
        if self.module == "compare":
            if self.model_list == ["all"]:
                print("Compare models and get a best model")
                best_model = compare_models(sort=self.metric)
            else:
                best_model = compare_models(include=self.model_list, sort=self.metric)
        elif self.module == "tune":
            if self.model_list == ["all"]:
                best_model = compare_models(sort=self.metric)
            else:
                best_model = tune_model(best_model, sort=self.metric)

        save_model(best_model, self.exp_name+"/model/best_model")

        return best_model


    def prediction(self, model, test):
        result = predict_model(model, test)
        result.to_csv(self.exp_name+"/predict/result.csv", index=False)
        return result