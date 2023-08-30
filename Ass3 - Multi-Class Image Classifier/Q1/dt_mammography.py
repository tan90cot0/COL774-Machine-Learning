import functions1
import sys

train_data_path = sys.argv[1]
validation_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_path = sys.argv[4]
question_number = sys.argv[5]
paths = (train_data_path, validation_data_path, test_data_path)

part_function = {'a':functions1.part_a, 'b':functions1.part_b, 'c': functions1.part_c, 'd': functions1.part_d}

if question_number in ['a', 'b', 'c', 'd']:
    data = functions1.get_data(paths, question_number, None)
    result,classifier = part_function[question_number](data, output_path)
    if question_number!='d':
        functions1.visualise(classifier, output_path + "/Tree_part_" + question_number)
elif question_number=='e':
    result = functions1.part_e(paths, output_path)
else:
    data = functions1.get_data(paths, question_number, None)
    result = functions1.part_f(data)
        
file = open(output_path + '/1_' + question_number + ".txt", "w")
file.writelines(result)
file.close()