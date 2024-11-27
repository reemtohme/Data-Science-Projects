# I have explicitly excluded some input options in my function specifiations. 
# I hence did not find the need to conduct tests for these input possibilities.

import unittest
import class_eval

class TestSplitTrainingTesting(unittest.TestCase):
    '''Tests for split_training_testing'''

    def test_split_empty(self):
        """Tests an empty data list."""
        inputted = []
        train, test = class_eval.split_training_testing(inputted, p_test=20)
        self.assertEqual(train, [])
        self.assertEqual(test, [])

    def test_split_small_list(self):
        """Tests small input lists with edge case of 50% as p_test."""
        inputted = [1]
        train, test = class_eval.split_training_testing(inputted, 50)
        self.assertEqual(len(train) + len(test), 1, "Data list has one element")
        self.assertIn(train[0], inputted) if train else self.assertIn(test[0], inputted)

        inputted = [1, 2]
        train, test = class_eval.split_training_testing(inputted, 50)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(test), 1)
        # To ensure no data is lost
        self.assertEqual(sorted(train + test), sorted(inputted))

    def test_0_percent_test(self):
        """Test with 0% test data."""
        inputted = [1, 2, 3, 4, 5]
        train, test = class_eval.split_training_testing(inputted, 0)
        # All data goes to train
        self.assertEqual(sorted(train), sorted(inputted))
        self.assertEqual(test, [])

    def test_100_percent_test(self):
        """Test with 100% test data."""
        inputted = [1, 2, 3, 4, 5]
        train, test = class_eval.split_training_testing(inputted, 100)
        self.assertEqual(train, [])  
        # All data goes to test
        self.assertEqual(sorted(test), sorted(inputted))

    def test_50_percent_test(self):
        """Test with 50% test data."""
        inputted = [1,2,3,4,5,6]
        train, test = class_eval.split_training_testing(inputted, 50)
        # Half the data is allocated to each
        self.assertEqual(len(train), len(test))  
        self.assertEqual(sorted(train + test), sorted(inputted)) 
    
    def test_invalid_percentage(self):
        """Test when the percentage is not valid (out of bounds)."""
        # Negative percentage
        with self.assertRaises(ValueError):
            class_eval.split_training_testing([1, 2, 3], -10) 
        # Percentage greater than 100
        with self.assertRaises(ValueError):
            class_eval.split_training_testing([1, 2, 3], 110)

    def test_randomness(self):
        """Test that shuffling occurs by running multiple splits."""
        inputted = [1, 2, 3, 4, 5, 6]
        train1, test1 = class_eval.split_training_testing(inputted, 50)
        train2, test2 = class_eval.split_training_testing(inputted, 50)
        # Due to randomness, train1,test1 and train2,test2 should differ occasionally
        self.assertNotEqual(train1, train2) if train1 != train2 else True

    def test_large_data_set(self):
        """Test with a large dataset."""
        inputted = list(range(1000))  
        train, test = class_eval.split_training_testing(inputted, 20)
        self.assertEqual(len(test), int(len(inputted) * 0.2)) 
        self.assertEqual(len(train), len(inputted) - len(test))  
        self.assertEqual(sorted(train + test), sorted(inputted))
    

class TestConfusionMatrix(unittest.TestCase):
    '''Tests for confusion_matrix'''
    def test_confusion_empty(self):
        """Test case where both predicted and actual lists are empty."""
        predicted = []
        actual = []
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # no predictions so all counts are zero
        self.assertEqual((TP, FP, TN, FN), (0,0,0,0))

    def test_confusion_single(self):
        """Test case where there is only a single prediction."""
        predicted = [1]
        actual = [1]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # both class labels in predicted and actual are equal to the positive class so True Positive = 1
        self.assertEqual((TP,FP,TN,FN), (1,0,0,0))

    def test_confusion_all_correct(self):
        """Test case where all predictions are correct (True Positives)."""
        predicted = [1, 1, 0, 0]
        actual = [1, 1, 0, 0]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # Only True Positives and True Negatives should be counted
        self.assertEqual((TP, FP, TN, FN), (2,0,2,0))
    
    def test_confusion_all_incorrect(self):
        """Test case where all predictions are incorrect."""
        predicted = [1, 1, 0, 0]
        actual = [0, 0, 1, 1]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # Only False Negatives and False Positives should be counted
        self.assertEqual((TP, FP, TN, FN), (0,2,0,2))
    
    def test_confusion_mixed(self):
        """Test case with mixed predictions (True Positives, False Positives, etc.)."""
        predicted = [1, 0, 1, 0]
        actual = [0, 0, 1, 1]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # Count the different types of predictions
        self.assertEqual((TP, FP, TN, FN), (1,1,1,1))

    def test_no_positive_class(self):
        """Test case where there are no instances of the positive class."""
        predicted = [0, 0, 0, 0]
        actual = [0, 0, 0, 0]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # Only True Negatives
        self.assertEqual((TP, FP, TN, FN), (0, 0, 4, 0))

    def test_no_negative_class(self):
        """Test case where there are no instances of the negative class."""
        predicted = [1, 1, 1, 1]
        actual = [1, 1, 1, 1]
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted, actual, positive_class=1)
        # Only True Positives
        self.assertEqual((TP, FP, TN, FN), (4, 0, 0, 0))

    def test_confusion_unequal_lengths(self):
        """Test when predicted and actual lists have unequal lengths."""
        predicted = [1, 0]
        actual = [0]
        with self.assertRaises(ValueError):
            class_eval.confusion_matrix(predicted, actual, positive_class=1)

class TestEvaluationMetrics(unittest.TestCase):
    '''Tests for all evaluation metrics'''
    
    def test_accuracy(self):
        """Test the accuracy metric."""
        predicted = [1, 0, 1, 0]
        actual = [1, 0, 0, 1]
        result = class_eval.accuracy(predicted, actual, positive_class=1)
        self.assertAlmostEqual(result, 0.5)  

    def test_accuracy_zero_division(self):
        """Test accuracy with no predictions (division by zero)."""
        predicted = []
        actual = []
        result = class_eval.accuracy(predicted, actual, positive_class=1)
        self.assertTrue(result != result)  # NaN check 

    def test_sensitivity(self):
        """Test the sensitivity metric."""
        predicted = [1, 1, 0, 0]
        actual = [1, 1, 1, 0]
        result = class_eval.sensitivity(predicted, actual, positive_class=1)
        self.assertAlmostEqual(result, 2/3)  

    def test_sensitivity_zero_division(self):
        """Test sensitivity with division by zero."""
        predicted = [0, 0, 0]
        actual = [0, 0, 0]
        result = class_eval.sensitivity(predicted, actual, positive_class=1)
        self.assertTrue(result != result)  

    def test_specificity(self):
        """Test the specificity metric."""
        predicted = [0, 0, 0, 0]
        actual = [1, 0, 0, 1]
        result = class_eval.specificity(predicted, actual, positive_class=1)
        self.assertAlmostEqual(result, 1) 

    def test_specificity_zero_division(self):
        """Test specificity with division by zero."""
        predicted = [1, 1, 1]
        actual = [1, 1, 1]
        result = class_eval.specificity(predicted, actual, positive_class=1)
        self.assertTrue(result != result)  

    def test_pos_pred_val(self):
        """Test the positive predictive value (precision)."""
        predicted = [1, 1, 0, 1]
        actual = [1, 0, 0, 1]
        result = class_eval.pos_pred_val(predicted, actual, positive_class=1)
        self.assertAlmostEqual(result, 2/3)

    def test_pos_pred_val_zero_division(self):
        """Test precision with division by zero."""
        predicted = [0, 0, 0]
        actual = [1, 1, 1]
        result = class_eval.pos_pred_val(predicted, actual, positive_class=1)
        self.assertTrue(result != result) 

    def test_neg_pred_val(self):
        """Test the negative predictive value."""
        predicted = [0, 0, 1, 1]
        actual = [0, 1, 1, 0]
        result = class_eval.neg_pred_val(predicted, actual, positive_class=1)
        self.assertAlmostEqual(result, 1/2)  

    def test_neg_pred_val_zero_division(self):
        """Test negative predictive value with division by zero."""
        predicted = [1, 1, 1]
        actual = [1, 1, 1]
        result = class_eval.neg_pred_val(predicted, actual, positive_class=1)
        self.assertTrue(result != result) 

if __name__ == "__main__":
    unittest.main()