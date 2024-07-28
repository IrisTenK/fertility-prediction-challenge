"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from scipy import stats
import joblib
import os

def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example
    # Create new variable with age
    df["age"] = 2024 - df["birthyear_bg"]

    # Imputing missing values in age with the mean
    df["age"] = df["age"].fillna(df["age"].mean())

    df["age_sq"]= df["age_bg"]**2

    # Years with partner
    df["years_partner"]= 2020- df["cf20m029"]

    # expand variable cf20m128 by adding another variable variability in thinking that the person will have more children in the future?,
    df['variability_moreChildren'] = df[["cf11d128", "cf12e128", "cf13f128", "cf14g128", "cf15h128", "cf16i128", "cf17j128", "cf18k128", "cf19l128", "cf20m128"]].std(axis=1)

    # Assuming df is your DataFrame
    columns = ["cf20m129", "cf19l129", "cf18k129", "cf17j129", "cf16i129", "cf15h129", "cf14g129", "cf13f129", "cf12e129", "cf11d129"]

    # Calculate the z-scores across the specified columns
    df['variability_NumberChildren'] = df[columns].apply(stats.zscore, axis=1).std(axis=1)

    # Selecting variables for modelling
    keepcols = ["nomem_encr", "woonvorm_2020", 'cf20m024', 'cf20m029', "cf20m128", "cf20m129","years_partner",
                "cf20m130", "birthyear_bg","nettohh_f_2020", "ci20m379", "cf20m013","cf20m020", "cf20m022",
                "cf20m025", 'ch20m219', "burgstat_2020","gender_bg", "migration_background_bg",
                "oplmet_2020","ci20m006","ci20m007",'cr20m093',"cv20l041","cv20l043","cv20l044","age_bg","age_sq",
                "variability_moreChildren", 'variability_NumberChildren'] 

    # Keeping data with variables selected
    df = df[keepcols]

    return df


def predict_outcomes(df, background_df=None, model_path="model_HG.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
