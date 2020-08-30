import streamlit as st
import os
import pandas as pd
import base64
import zipfile
from shutil import make_archive, rmtree


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download result {file_label}</a>'
    return href


setting = pd.read_csv("setting_automl.csv", index_col=0)

st.sidebar.image("logo.png", width=40)
st.sidebar.markdown(
    "# Auto-ML using pycaret by Toroi"
)

experiment_name = st.sidebar.text_input("Experiments name => ", "experiment")
setting.loc["exp_name", "property0"] = experiment_name

mode = st.sidebar.selectbox(
    "Pycaret", ["classification", "regression"]
)

if st.sidebar.button("Start!!"):
    setting.loc["mode", "property0"] = mode
    setting.to_csv("setting_automl.csv")

models = "all"

def main():

    target = False
    ignore_features = False

    uploaded_training_file = None
    uploaded_test_file = None

    uploaded_training_file = st.file_uploader("Training data [.csv]",
                                              type=["csv"],
                                              encoding='auto',
                                              key=None)

    uploaded_test_file = st.file_uploader("Test data [.csv]",
                                          type=["csv"],
                                          encoding='auto',
                                          key=None)

    if uploaded_training_file is not None and uploaded_test_file is not None:
        train = pd.read_csv(uploaded_training_file)
        test = pd.read_csv(uploaded_test_file)
        target = st.selectbox(
            "target", list(train.columns)
        )
        setting.loc["target", "property0"] = target
        setting.to_csv("setting_automl.csv")

        if st.button('Run'):
            from app import Pycaret_CLI
            pcl = Pycaret_CLI("setting_automl.csv")
            exp = pcl.setup_automl_env(train)
            best_model = pcl.training_model()
            pred_result = pcl.prediction(best_model, test)
            os.chdir(pcl.exp_dir)
            st.success("Done!")
            make_archive(pcl.exp_name, 'zip', root_dir=pcl.exp_dir)
            st.markdown(get_binary_file_downloader_html(f'{pcl.exp_name}.zip', 'ZIP'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()