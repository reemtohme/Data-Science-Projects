import random

def split_training_testing(data, p_test):
      
    """
    Splits the data into training and testing sets

    Assumes 
    data: a list of datapoints (datapoints can be of any type, e.g., int, float, strings, or objects).
    p_test: percentage of data to be used for testing. It is of int or float type
    
    Returns a tuple (train, test), where train is the training set and test is the testing set
    """
    
    # Raise ValueError if p_test is outside the boundaries 0 and 100
    if p_test < 0 or p_test > 100:
        raise ValueError("Test percentage must be a number between 0 and 100.")
    
    # Calculate the number of test samples based on p_test
    test_samples = int(len(data) * (p_test / 100))
    
    # Shuffle the data to ensure randomness
    shuffled_data = data[:]
    random.shuffle(shuffled_data)
    
    # Split the shuffled data into train and test sets
    test = shuffled_data[:test_samples]
    train = shuffled_data[test_samples:]
    
    return train, test

def confusion_matrix(predicted, actual, positive_class):
    
    """
    Calculates the confusion matrix metrics based on the predicted and actual class values for binary classification.

    Assumes:
    - predicted: a list of predicted binary class labels.
    - actual: a list of actual binary class labels.
    - positive_class: class label considered as the "positive" class
    - the class labels can be of any comparable type (e.g., strings, integers, floats), 
      but must be consistent within each list (i.e., all elements in the 'predicted' and 'actual' lists 
      must have the same type).
    - the positive class is one of the binary class labels, hence having same type

    Returns a tuple (TP, FP, TN, FN) representing:
    TP (True Positives), FP (False Positives), TN (True Negatives), FN (False Negatives).
    """
    
    # Check if predicted and actual lists are equal in length. If not, raise ValueError
    if len(predicted) != len(actual):
        raise ValueError("Predicted and actual lists must have the same length.")
    
    # Initialize counts
    TP = FP = TN = FN = 0

    # Loop through each pair of predicted and actual list elements
    for pred, act in zip(predicted, actual):
        # Count TP - predicted and actual both match the positive class
        if pred == positive_class and act == positive_class:
            TP += 1
        # Count FP - predicted matches the positive class but actual does not
        elif pred == positive_class and act != positive_class:
            FP += 1  
        # Count TN - neither predicted nor actual match the positive class
        elif pred != positive_class and act != positive_class:
            TN += 1 
        #Count FN - predicted does not match the positive class but actual does
        elif pred != positive_class and act == positive_class:
            FN += 1  

    return TP, FP, TN, FN

def accuracy(predicted, actual, positive_class):
    """
    Calculates the accuracy metric based on the confusion matrix for binary classification.
    
    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.

    Returns 
    - Accuracy: a float representing the proportion of correctly classified samples out of all samples.
    - In case of division by zero, returns NaN and prints an error message.
    """
    try:
        TP, FP, TN, FN = confusion_matrix(predicted, actual, positive_class)
        return (TP + TN) / (TP + TN + FP +FN)
    except ZeroDivisionError:
        print("Division by zero is undefined.")
        return float('nan')

def sensitivity(predicted, actual, positive_class):
    """
    Calculates the sensitivity metric based on the confusion matrix for binary classification.
    
    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.
    
    Returns
    - Sensitivity: a float representing the ratio of true positives to the sum of true positives and false negatives.
    - In case of division by zero, returns NaN and prints an error message.
    """
    try:
        TP, FP, TN, FN = confusion_matrix(predicted, actual, positive_class)
        return TP / (TP + FN)
    except ZeroDivisionError:
        print("Division by zero is undefined.")
        return float('nan')
    
def specificity(predicted, actual, positive_class):
    """
    Calculates the speifictity metric based on the confusion matrix for binary classification.
    
    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.
    
    Returns
    - Specificity: a float representing the ratio of true negatives to the sum of true negatives and false positives.
    - In case of division by zero, returns NaN and prints an error message.
    
    """
    try:
        TP, FP, TN, FN = confusion_matrix(predicted, actual, positive_class)
        return TN / (TN + FP)
    except ZeroDivisionError:
        print("Division by zero is undefined.")
        return float('nan')

def pos_pred_val(predicted, actual, positive_class):
    """
    Calculates the positive predictive value (precision).

    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.

    Returns:
    - Positive Predictive Value (TP / (TP + FP)).
    - NaN if division by zero occurs.
    """
    try:
        TP, FP, TN, FN = confusion_matrix(predicted, actual, positive_class)
        return TP / (TP + FP)
    except ZeroDivisionError:
        print("Division by zero is undefined.")
        return float('nan')

def neg_pred_val(predicted, actual, positive_class):
    """
    Calculates the negative predictive value.

    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.

    Returns:
    - Negative Predictive Value (TN / (TN + FN)).
    - NaN if division by zero occurs.
    """
    try:
        TP, FP, TN, FN = confusion_matrix(predicted, actual, positive_class)
        return  TN / (TN + FN)
    except ZeroDivisionError:
        print("Division by zero is undefined.")
        return float('nan')

def print_eval_metrics(predicted, actual, positive_class):
    """
    Prints evaluation metrics

    Assumes:
    - predicted: A list of predicted class labels (can be of any type).
    - actual: A list of actual class labels (same type as predicted).
    - positive_class: The class considered as the "positive" class for classification.
    
    """
    # Print the corresponding evaluation metrics using their defined functions
    print(f"Accuracy: {accuracy(predicted, actual, positive_class)}")
    print(f"Sensitivity: {sensitivity(predicted, actual, positive_class)}")
    print(f"Specificity: {specificity(predicted, actual, positive_class)}")
    print(f"Positive predictive value: {pos_pred_val(predicted, actual, positive_class)}")
    print(f"Negative predictive value: {neg_pred_val(predicted, actual, positive_class)}")
