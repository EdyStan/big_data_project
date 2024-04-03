# import the necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV


# define the constants of the file.
f1_scores_path = '../../conclusions/f1_scores_macro.csv'  # the file in which we draw the F1 scores
accuracies_path = '../../conclusions/accuracies.csv'  # the file in which we draw the accuracies
raw_data = pd.read_csv('../../data/diabetic_data.csv')  # the file from which we extract the data

def write_conclusions(model, f1_score, accuracy):
    """
    Draws the results into the files in which we store the conclusions of this project.
    The results are displayed at:
    ROW = `the preprocessing method`
    COLUMN = `the supervised model used for predictions`
    """

    # we store the name of the current folder. It symbolizes the preprocessing method we're using
    folder_name = os.getcwd().split('/')[-1]  
    model_name = type(model).__name__  # the name of the model
    f1_df = pd.read_csv(f1_scores_path)  # load the macro F1 scores data into a pd.DataFrame
    acc_df = pd.read_csv(accuracies_path)  # load the accuracies data into a pd.DataFrame

    # extract the ROW index for the macro F1 score and accuracy metrics, according to the preprocessing method we're using
    f1_row_index = f1_df.loc[f1_df['PreparationMethod'] == folder_name].index[0]
    acc_row_index = acc_df.loc[acc_df['PreparationMethod'] == folder_name].index[0]

    # we draw the results at coordinates ROW and COLUMN
    f1_df.loc[f1_row_index, model_name] = f1_score
    acc_df.loc[acc_row_index, model_name] = accuracy
    
    # dump the updated data into the corresponding csv file
    f1_df.to_csv(f1_scores_path, index=False) 
    acc_df.to_csv(accuracies_path, index=False)


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


def applyPCA(encoded_df):
    """
    Principal Component Analysis (PCA) 
    orders dimensions by the amount of variance they explain
    Black box explained:
    1. Centers the data by extracting the mean of each feature
    2. Calculates the covariance matrix (the relationship between each feature)
    3. Computes eigenvalues and eigenvectors of the covariance matrix
    (eigenvalues can be interpreted as the magnitude of variance, and the eigenvectors
    as the directions of the maximum variance)
    4. Principal components are chosen based on the amount of variance we want to retain in the dataset
    5. The original data is projected onto this new space 
    (the centered data matrix is multiplied by the projection matrix)
    """

    pca_dimensions = PCA()
    pca_dimensions.fit(encoded_df)
    # we look for instances with the smallest explained variance (eigenvalue / total variance)
    cumulative_variance = np.cumsum(pca_dimensions.explained_variance_ratio_)
    # we choose the percentage of variance to be left
    variance_left = 0.9999
    num_features = np.argmax(cumulative_variance >= variance_left) + 1
    print(f"Number of features for {variance_left} explained variance: {num_features}")

    # As a result of the previous calculations, we compute PCA with the chosen number of principal components
    # Moreover, we check if the inverse transformation takes us back to a distinguishable image
    pca = PCA(n_components=num_features)
    reduced_df = pca.fit_transform(encoded_df)
    return pd.DataFrame(reduced_df)

# encode the data set
encoded_data = encode_object_features(raw_data)

# Extract features and labels and then split them into train and test data
raw_X, y = encoded_data.drop(columns=['readmitted']), encoded_data['readmitted']

X = applyPCA(raw_X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the scoring metric as being the macro F1 score
scoring_metric = make_scorer(f1_score, average='macro')
