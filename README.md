This project aims to predict [...] from the following [dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

Over this data set, we used 4 preprocessing methods, each of them being preceded by 4 supervised models. The performance was measured using accuracy as the main metric of evaluation.

The folder `performance_measurements` contains the actual code of this project. This folder, in turn, is composed of 1 file (in which we take a first glimpse of the data) and 4 subfolders representing the preprocessing methods used:

- Simple encoding - just encoding the object-typed features.

- [...]

- [...]

- [...]

Each of these subfolders contain 5 files. The first file contains the data preprocessing and is run at the beginning of each other four files. The other 4 files contain one supervised model each, and evaluate it according to the true labels. The supervised models that we used are the following:

- Random Forest Classifier.

- Support Vector Classifier (linear kernel).

- K Nearest Neighbors.

- Naive Bayes.


Finally, the results and the project report can be found in the folder `conclusions`.