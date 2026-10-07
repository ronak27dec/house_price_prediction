import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def clean_data(df):
    df = df.drop_duplicates()
    df = df['Area_SqFt'] > 0 and df['Rooms'] and df['Build_Age']
    return df

def create_preprocessor(config):
    num_cols = config['features']['numeric_cols']
    cat_cols = config['features']['categorical_cols']

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    return preprocessor
