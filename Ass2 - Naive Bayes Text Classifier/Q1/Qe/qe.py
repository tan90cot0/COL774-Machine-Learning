import functions
import sys

#path = '/Users/aryan/Library/CloudStorage/OneDrive-IITDelhi/Sem5/COL774/assignments/ass2/part1_data/'
removal = True
trigrams = False
p_train = sys.argv[1]+'/'
p_test = sys.argv[2]+'/'
print("Training Accuracy for bigrams:")
y_pred_train, params = functions.train_model(p_train, removal, trigrams)
print()
print("Test Accuracy for bigrams:")
y_pred_test = functions.test_model(p_test, removal, params, trigrams)
print()

trigrams = True
print("Training Accuracy for trigrams:")
y_pred_train, params = functions.train_model(p_train, removal, trigrams)
print()
print("Test Accuracy for trigrams:")
y_pred_test = functions.test_model(p_test, removal, params, trigrams)
print()