#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import joblib
import os




print(np.__version__)


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model."""
    
       # Create age squared
    df["age_sq"]= df["age_bg"]**2

     # Years with partner
    df["years_partner"]= 2020- df["cf20m029"]

    # (2) expand variable cf20m128 by adding another variable variability in thinking that the person will have more children in the future?,
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
    cleaned_df = df[keepcols]

    return cleaned_df


# In[76]:


def train_save_model(cleaned_df, outcomes_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcomes_df, on="nomem_encr")

     # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  

    # Upsampling
    children= model_df[model_df['new_child']==1]
    nochildren= model_df[model_df['new_child']==0]

    from sklearn.utils import resample
    children_upsample=resample(children, replace=True, n_samples=int(0.60*len(nochildren)), random_state=42)
    print(children_upsample['new_child'].sum())

    data_upsampled= pd.concat([nochildren, children_upsample])
    print(data_upsampled["new_child"].value_counts())
    
    

    ## Categorize variables
    numerical_columns = ["age_bg", "age_sq", "nettohh_f_2020","birthyear_bg", "years_partner", "variability_moreChildren", 'variability_NumberChildren']
    categorical_columns = ["cf20m128", "cf20m013", "cf20m024","cf20m025",
                           "burgstat_2020", "oplmet_2020", 'ch20m219', 'cr20m093',
                          "gender_bg", "migration_background_bg","woonvorm_2020" ]
    categorical_columns_ordinal = ["cf20m020", "cf20m129", "cf20m130", "cf20m022","ci20m006","ci20m007",
                                   "cv20l041","cv20l043","cv20l044","ci20m379"]
    
    # HG Boost model
    from sklearn.ensemble import HistGradientBoostingClassifier
    categorical_columns_plus_ordinal = ["cf20m128", "cf20m013", "cf20m024","cf20m025",
                           "burgstat_2020", "oplmet_2020", 'ch20m219', 'cr20m093',
                          "gender_bg", "migration_background_bg","woonvorm_2020",
                                    "cf20m020", "cf20m129", "cf20m130", "cf20m022","ci20m006","ci20m007",
                                   "cv20l041","cv20l043","cv20l044","ci20m379"]

    from sklearn.preprocessing import OrdinalEncoder

    ordinal_preprocessor2 = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value=-1)

    from sklearn.compose import ColumnTransformer

    preprocessor2= ColumnTransformer([("ordinal-encoder", ordinal_preprocessor2, categorical_columns_plus_ordinal)],
    remainder="passthrough")

    model_HG = make_pipeline(preprocessor2,HistGradientBoostingClassifier())
    model_HG.fit(data_upsampled[["nomem_encr", "woonvorm_2020", 'cf20m024', 'cf20m029', "cf20m128", "cf20m129","years_partner",
                "cf20m130", "birthyear_bg","nettohh_f_2020", "ci20m379", "cf20m013","cf20m020", "cf20m022",
                "cf20m025", 'ch20m219', "burgstat_2020","gender_bg", "migration_background_bg",
                "oplmet_2020","ci20m006","ci20m007",'cr20m093',"cv20l041","cv20l043","cv20l044","age_bg","age_sq",
                "variability_moreChildren", 'variability_NumberChildren']], data_upsampled['new_child'])


    # Save the model
    joblib.dump(model_HG, "model.joblib")


# In[77]:


def predict_outcomes(df, background_df=None, model_path="model_XG.joblib"):
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


# In[78]:


#Load the background data

import pandas as pd


### Run clean df function
cleaned_df = clean_df(df) 

### Train and save model
train_save_model(cleaned_df ,outcomes_df)


# In[ ]:




