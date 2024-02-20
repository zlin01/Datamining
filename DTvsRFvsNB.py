# Please install 
# pip install scikit-learn
# pip install pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import time
import sys

# Task_2
# Function to read data and split into training and test sets, using sklearn.model_selection
def task_2_split_data(input_file, train_out, test_out):
    # Read the dataset
    data = pd.read_csv(input_file, header=None)
    # Split the dataset into training and test sets, set up train is 80% and test is 20%
    # Set up random state to keep the split data same, using random_state to fix the value
    train_set, test_set = train_test_split(data, train_size=0.8, test_size=0.2, random_state=3)
    # Save the training and test sets to text files
    train_set.to_csv(train_out, index=False, header=False)
    test_set.to_csv(test_out, index=False, header=False)
    return train_set, test_set
# Comment out since we are not using it
# task_2_split_data('crx.data', 'credit_trainset.txt', 'credit_testset.txt')
# task_2_split_data('adult.data', 'census_trainset.txt', 'census_testset.txt')

# Convert the data to the data that can be use in the ml
def encode_features(data):
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == type(object):
            data[column] = label_encoder.fit_transform(data[column])
    return data

def load_data(training_file, test_file):
    training_data = pd.read_csv(training_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    training_data = encode_features(training_data)
    test_data = encode_features(test_data)
    return training_data, test_data

def split_data(dataset):
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return x, y

def task5_result(input_algoritm, X_train, y_train, X_test, y_test):
    start = time.time()
    input_algoritm.fit(X_train, y_train)
    predictions = input_algoritm.predict(X_test)
    runtime = time.time() - start
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    print(f"Classification Report:\n{classification_report(y_test, predictions)}")
    print(f"Runtime: {runtime}s")
    return accuracy, precision, recall, predictions

# Using scikit-learn library to training the decision tree
# Credit for DT: https://scikit-learn.org/stable/modules/tree.html#tree
def decision_tree(training_file, test_file):
    training_data, test_data = load_data(training_file, test_file)
    # Return to two arrays as (x=n_samples, y=n_feature)
    x_train, y_train = split_data(training_data)
    x_test, y_test = split_data(test_data)
    dt = DecisionTreeClassifier()
    accuracy, precision, recall, predictions = task5_result(dt, x_train, y_train, x_test, y_test)
    task_4(predictions, y_test)
    return accuracy, precision, recall

def random_forest(training_file, test_file):
    training_data, test_data = load_data(training_file, test_file)
    x_train, y_train = split_data(training_data)
    x_test, y_test = split_data(test_data)
    # Set up maximum of two splits easier for life
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    accuracy, precision, recall, predictions = task5_result(rf, x_train, y_train, x_test, y_test)
    task_4(predictions, y_test)
    return accuracy, precision, recall

def naive_bayes(training_file, test_file):
    training_data, test_data = load_data(training_file, test_file)
    x_train, y_train = split_data(training_data)
    x_test, y_test = split_data(test_data)
    nb = GaussianNB()
    accuracy, precision, recall, predictions = task5_result(nb, x_train, y_train, x_test, y_test)
    task_4(predictions, y_test)
    return accuracy, precision, recall

# Task 4 to print out the line
def task_4(predictions, y_test):
    for i, (predicted, true) in enumerate(zip(predictions, y_test), start=1):
        accuracy = int(predicted == true)
        print('ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}'.format(i, predicted, true, accuracy))

# Task 6
def ignore_tuple(input_file, output_train, output_test):
    # Read the dataset
    data = pd.read_csv(input_file, header=None, na_values='?')
    # check number of missing values
    # miss_v = data.isnull().sum()
    # print(f"Missing values per column before dropping:\n{miss_v}")
    # delete the missing data
    data_cleaned = data.dropna()
    # Count how many rows were deleted
    print(f"Deleted {len(data) - len(data_cleaned)} rows with missing values.")
    # Split the dataset into training and test sets, set up train is 80% and test is 20%
    train_set, test_set = train_test_split(data_cleaned, train_size=0.8, test_size=0.2, random_state=3)
    # Save the cleaned and split datasets to text files
    train_set.to_csv(output_train, index=False, header=False)
    test_set.to_csv(output_test, index=False, header=False)
    return train_set, test_set

def main(training_file, test_file):
    # copy for the task_2_split_data
    tt, t1 = train_test_split(pd.read_csv('crx.data', header=None), test_size=0.2, random_state=3)
    # print(tt)
    t2, t3 = train_test_split(pd.read_csv('adult.data', header=None), test_size=0.2, random_state=3)
    print(f"Number of instances in the Credit training dataset: {len(tt)}")
    print(f"Number of instances in the Credit test dataset: {len(t1)}")
    print(f"Number of instances in the Credit training dataset: {len(t2)}")
    print(f"Number of instances in the Credit test dataset: {len(t3)}")

    print("********** Task6 ***********")
    task6_train, task6_test = ignore_tuple('crx.data', 'Task6_credit_trainset.txt', 'Task6_credit_testset.txt')
    print(f"Number of instances in the new Credit training dataset after handling missing data: {len(task6_train)}")
    print(f"Number of instances in the new Credit test dataset after handling missing data: {len(task6_test)}")

    print("\nDecision Tree Results on cleaned data:")
    decision_tree('Task6_credit_trainset.txt', 'Task6_credit_testset.txt')
    print("\nRandom Forest Results on cleaned data:")
    random_forest('Task6_credit_trainset.txt', 'Task6_credit_testset.txt')
    print("\nNaive Bayes Results on cleaned data:")
    naive_bayes('Task6_credit_trainset.txt', 'Task6_credit_testset.txt')
    print("********** Task6 ***********")

    print("Decision Tree Results:")
    decision_tree(training_file, test_file)

    print("\nRandom Forest Results:")
    random_forest(training_file, test_file)

    print("\nNaive Bayes Results:")
    naive_bayes(training_file, test_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DTvsRFvsNB.py <training_file> <test_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
