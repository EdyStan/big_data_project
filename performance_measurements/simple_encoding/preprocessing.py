# import the necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# define the constants of the file.
accuracies_path = '../../conclusions/accuracies.csv'  # the file in which we draw our results
raw_data = pd.read_csv('../../data/diabetic_data.csv')  # the file from which we extract the data

def write_conclusions(model, accuracy):
    """
    Draws the results into the files in which we store the conclusions of this project.
    The results are displayed at:
    ROW = `the preprocessing method`
    COLUMN = `the supervised model used for predictions`
    """

    # we store the name of the current folder. It symbolizes the preprocessing method we're using
    folder_name = os.getcwd().split('/')[-1]  
    model_name = type(model).__name__  # the name of the model
    accuracies_df = pd.read_csv(accuracies_path)  # load the accuracies data into a pd.DataFrame

    # extract the ROW index for the accuracy metric, according to the preprocessing method we're using
    accuracies_row_index = accuracies_df.loc[accuracies_df['PreparationMethod'] == folder_name].index[0]
    # we draw the results at coordinates ROW and COLUMN
    accuracies_df.loc[accuracies_row_index, model_name] = accuracy
    # dump the updated data into the corresponding csv file
    accuracies_df.to_csv(accuracies_path, index=False) 


def encode_object_features(df):
    """
    Encode all the object-typed columns from the dataset.
    Return the updated pd.DataFrame.
    """
    encoded_df = df.copy()
    for column in df.columns:
        if df[column].dtype == 'object':
            encoded_df[column], _ = pd.factorize(df[column])
    return encoded_df

# encode the data set
encoded_data = encode_object_features(raw_data)

# Extract features and labels and then split them into train and test data
X, y = encoded_data.drop(columns=['readmitted']), encoded_data['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)