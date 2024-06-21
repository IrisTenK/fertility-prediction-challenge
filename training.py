"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    random.seed(1) # not useful here because logistic regression deterministic
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    ## Create imputer to impute missing values in the pipeline
    imputer = KNNImputer(n_neighbors=2, weights="uniform").set_output(transform = "pandas")

    ## Normalize variables
    numerical_columns = ["age"]
    categorical_columns = ["woonvorm_2020", "cf20m003"]
        
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
    
    # Logistic regression model
    #model = LogisticRegression()
    model = make_pipeline(imputer, preprocessor, LogisticRegression(max_iter=500))


    # Fit the model
    model.fit(model_df[['age', 'woonvorm_2020', "cf20m003"]], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")
