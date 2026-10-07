import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config():
    with open('../config/config.yaml', mode='r') as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    path = config['data']['raw_path']
    return pd.read_csv(path)

def split_data(df, config):
    target = config['features']['target']
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']

    X = df.drop(target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    return (X_train, X_test, y_train, y_test)

