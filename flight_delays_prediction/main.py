import os
from typing import Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import wandb

def prepare_dataset(filename: str):
    """
    Perform basic data cleaning

    Args:
        filename (str): The name of the file
    """
    parent_directory = os.path.dirname(os.getcwd())
    dataset_folder_path = os.path.join(parent_directory, 'dataset')

    df = pd.read_csv(os.path.join(dataset_folder_path, filename))

    calendar_type_cols = ["Month", "DayofMonth", "DayOfWeek"]
    for col in df.columns:
        if col in calendar_type_cols:
            df[col] = df[col].str.replace('c-', '').astype(int)

    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]

    encoder_list: Dict[str, LabelEncoder] = []

    for col in df.columns:
        if col in categorical_columns:
            le = LabelEncoder()
            le.fit(df[col].tolist())
            df[col] = le.transform(df[col].tolist())
            encoder_list.append({"encoder_name": col, "encoder": le})
            
    return df

cleaned_trained_df = prepare_dataset("flight_delays_train.csv")
cleaned_test_df = prepare_dataset("flight_delays_test.csv")

with wandb.init(project="flight-delay-prediction", job_type="load-data") as run:
    flight_delays_data = wandb.Artifact(name="flight_delays_data", type="dataset")
    train_df_table = wandb.Table(columns=cleaned_trained_df.columns, dataframe=cleaned_trained_df)
    test_df_table = wandb.Table(columns=cleaned_test_df.columns, dataframe=cleaned_test_df)
    flight_delays_data.add(train_df_table, "training_table")
    flight_delays_data.add(test_df_table, "testing_table")
    run.log_artifact(flight_delays_data)