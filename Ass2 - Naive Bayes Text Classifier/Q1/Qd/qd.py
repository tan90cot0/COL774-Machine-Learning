import functions
import sys

#path = '/Users/aryan/Library/CloudStorage/OneDrive-IITDelhi/Sem5/COL774/assignments/ass2/part1_data/'
removal = True
y, words_matrix,py, pos, neg = functions.get_data(sys.argv[1]+'/', removal)
vocab= functions.get_vocab(words_matrix)
phi_1, phi_0 = functions.get_params(vocab, words_matrix, y)
print("Training Accuracy is:")
y_pred = functions.get_accuracy(words_matrix, py, phi_1, phi_0, vocab, y, True)
print()
y, words_matrix,py2, pos2, neg2 = functions.get_data(sys.argv[2]+'/', removal)
print("Test Accuracy is:")
y_pred = functions.get_accuracy(words_matrix, py, phi_1, phi_0, vocab, y, False)
print()
functions.get_wordcloud(pos, neg)