# Imports
import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def read_dataframe(filename: str) -> pd.DataFrame:
    # Reading parquet file, followed by EDA
    df = pd.read_parquet(filename)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/pd.Timedelta(minutes=1)
    df = df[(df['duration']<=60) & (df['duration']>=1)]
    return df

# We will work with only these two cat features, the pickup and dropoff zone in NYC to predict the trip duration
# We filter these features and convert them to string to allow DictVectorizer to vectorize categories
def feature_engineer(df: pd.DataFrame, fit_dv: bool = False) -> tuple:
    cat_features = ['PULocationID','DOLocationID']
    df[cat_features] = df[cat_features].astype('str')
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    # Converting input feature to dicts for the vectorizers (Dict vectorizer works well for sparse matrices and handles missing values through imputation)
    dicts = df[cat_features + numerical].to_dict(orient='records') 
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def preprocess(filename: str) -> pd.DataFrame:
    df = read_dataframe(filename)
    df = clean(df)
    return df

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = 'green'):
    # Load
    df_train = preprocess(os.path.join(raw_data_path, f'{dataset}_tripdata_2023-01.parquet'))
    df_train = preprocess(os.path.join(raw_data_path, f'{dataset}_tripdata_2023-01.parquet'))
    df_train = preprocess(os.path.join(raw_data_path, f'{dataset}_tripdata_2023-01.parquet'))

     # Extract the target
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

