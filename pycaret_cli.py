import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *
from pycaret.datasets import get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston_data = get_data('boston')
train, test = train_test_split(boston_data)

print("SETUP EXPERIMENTS")
exp_0 = setup(data=train, target="medv", html=False, silent=True)
print("Finished !!")

# compare all baseline models and select top 5
print("Compare models")
top5 = compare_models(n_select = 5)
# tune top 5 base models
print("Tune models")
tuned_top5 = [tune_model(i) for i in top5]
# ensemble top 5 tuned models
print("Bagging models")
bagged_top5 = [ensemble_model(i) for i in tuned_top5]
# blend top 5 base models
print("Blend models")
blender = blend_models(estimator_list = top5)

print("Save blend models")
save_model(blender, "blender")

print("Predict test")
result = predict_model(blender, data=test.iloc[:, :-1])

print("Mean Square Error => ", mean_squared_error(test["medv"], result["Label"]))
print("Save Predict")

result.to_csv("submit_by_pycaret_top5_blender.csv", header=False, index=False)