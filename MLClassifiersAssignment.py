# import libs
# Importing our toy dataset
from sklearn.datasets import load_iris
# Import our DT classifier
from sklearn import tree
# This is data numpy
import numpy as np

# function to limit the amount of times I need to type "print()" within a program
# especially for intros and big blocks of text
def text(text_to_print):
    print(text_to_print)

def intro():
    text("\n\nThis application takes in a dataset complete with information on several peoples iris specifications.\n"
         "Using this information, the target names, or row names will be displayed, same with\n"
         "the feature names, or column names. All pulled from the original dataset without downloading or opening\n"
         "the document. From there, the application will be able to classify the information based on the\n"
         "dataset the application was given.")

    # User input for interaction with user
    repeat = True
    while repeat:
        ans = input("Ready to begin?  Type Y/N\n")

        if ans == "Y" or ans == "y":
            repeat = False
        elif ans == "N" or ans == "n":
            exit()
        else:
            text("Please type \"Y\" or \"N\" and try again...")

def getTargetNames(raw_data):
    text("\n\t*** Target Names ***")
    for tn in raw_data.target_names:
        text(tn)

def getFeatureNames(raw_data):
    text("\n\t*** Feature Names ***")
    for fn in raw_data.target_names:
        text(fn)

# inlcuded to limit the code required in the user_given_data code block
def getInt():
    # included to handle non-expected input from user without breaking/halting the applications execution
    while True:
        try:
            x = int(input("Please enter a number: "))
            if x > 150:
                print("Oops! That number is greater than 150. Try again...")
                break
            elif x < 0:
                print("Oops! That number is less than 0. Try again...")
                break
            else:
                return x
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

def user_given_data(raw_data):
    repeat = True
    while repeat:
        # 150 max for dataset
        text("\nPlease enter three numerical integers under the number 150. Example: 1 , 17, 64  ")
        num1 = getInt()
        num2 = getInt()
        num3 = getInt()

        # Three samples of test data given by user
        test_data_by_index = [num1, num2, num3]
        training_data(raw_data, test_data_by_index)

        ans = input("Would you like to test another set of data? Type Y/N\n")
        if ans == "Y" or ans == "y":
            pass
        elif ans == "N" or ans == "n":
            exit()
        else:
            text("Please type \"Y\" or \"N\" and try again...")

def training_data(raw_data, test_data_by_index):
    # Training data
    train_target = np.delete(raw_data.target, test_data_by_index)
    train_data = np.delete(raw_data.data, test_data_by_index, axis=0)

    # Testing data
    test_target = raw_data.target[test_data_by_index]
    test_data = raw_data.data[test_data_by_index]
    makePrediction(train_target, train_data, test_target, test_data)

def makePrediction(train_target, train_data ,test_target, test_data):
    # Create the Decision Tree
    dt_classifier = tree.DecisionTreeClassifier()
    # Train
    dt_classifier.fit(train_data, train_target)

    # displaying results
    print("\n\t*** Test Results ***")
    print("\nTest target labels are ", test_target)
    print("\nPredictions by our decicion tree labels are ", dt_classifier.predict(test_data))

def main():
    # load in scikit dataset
    iris_raw_data = load_iris()

    intro()
    getTargetNames(iris_raw_data)
    getFeatureNames(iris_raw_data)
    user_given_data(iris_raw_data)

if __name__ == "__main__":
    main()






