import csv
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

NUMBER_OF_LINES = 22284  

file_dir = "C:/Users/HP/Downloads/Gene_Chip_Data/"
output_file = 'C:/Users/HP/Desktop/7th sem/Mtech/BioInfo/Data/processed.csv'


def read_only_lines(f, start, finish):
    for ii, line in enumerate(f):
        if ii >= start and ii < finish:
            yield line
        elif ii >= finish:
            return


def mean_around_zero(string_list):
    total = 0
    zero_mean_list = []

    for i in range(1, len(string_list)):
        current = float(string_list[i])
        total += current
        zero_mean_list.append(current)

    mean = total / (len(string_list) - 1)

    for x in range(0, len(zero_mean_list)):
        zero_mean_list[x] -= mean

    return zero_mean_list


def normalize(lst, max_val):
    for i in range(0, len(lst)):
        lst[i] /= max_val

    return lst


def publish(publish_list):
    publish_list = StandardScaler().fit_transform(publish_list)
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(publish_list)

line_start = 1
line_end = 3000
x = []
counter = 0
for line in read_only_lines(open(file_dir + 'microarray.original.txt'), line_start, line_end):

    word_list = line.split()
    float_list = []
    for i in range(1, len(word_list)):
        current = float(word_list[i])
        float_list.append(current)

    x.append(float_list)

x = np.array(x).transpose()
x = StandardScaler().fit_transform(x)

top_row = []
for i in range(0, line_end - line_start):
    top_row.append(i)
x = np.vstack([top_row, x])
np.savetxt("processed.csv", x, delimiter=",")
