import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

raw_data = pd.read_csv('../data/diabetic_data.csv')

def encode_object_features(df):
    encoded_df = df.copy()
    for column in df.columns:
        if df[column].dtype == 'object':
            encoded_df[column], _ = pd.factorize(df[column])
    return encoded_df

encoded_data = encode_object_features(raw_data)
print(encoded_data.info())

X, y = encoded_data.drop(columns=['readmitted']), encoded_data['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)