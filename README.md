Used the following [dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Regarding the absurd number of notebooks, here is an easy way to identify their content just by looking at the file name:

The name follows a clear structure: prep\_[**n**]\_alg\_[**m**].ipynb

Where:

- **n** means the index of the data preparation method.
    1. Simple encoding
    2. [...]
    3. [...]
    4. [...]

- **m** means the index of the algorithm used for predictions.
    1. Random Forest Classifier
    2. Support Vector Classifier (Linear Kernel)
    3. K Nearest Neighbors
    4. Naive Bayes