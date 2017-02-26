import numpy as np
import math
import random
import os
import cPickle as pickle
import idx2numpy
import numpy as np
import argparse

max_allowable_stroke = 1023
max_allowable_ascii = 70

mnist_train_labels_file = '/develop/data/mnist/download/train-labels-idx1-ubyte.idx'
mnist_test_labels_file = '/develop/data/mnist/download/t10k-labels-idx1-ubyte.idx'

mnist_train_path_template = "/develop/data/mnist-digits-stroke-sequence-data/sequences/trainimg-{}-inputdata.txt"
mnist_test_path_template = "/develop/data/mnist-digits-stroke-sequence-data/sequences/testimg-{}-inputdata.txt"

    # def __init__(self, args, logger, limit = 500):
    #     self.data_dir = args.data_dir
    #     self.batch_size = args.batch_size
    #     # self.tsteps = args.tsteps
    #     self.data_scale = args.data_scale # scale data down by this factor
    #     # self.ascii_steps = args.tsteps/args.tsteps_per_ascii
    #     self.logger = logger
    #     self.limit = limit # removes large noisy gaps in the data

    #     data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
    #     stroke_dir = self.data_dir + "/lineStrokes"
    #     ascii_dir = self.data_dir + "/ascii"

    #     if not (os.path.exists(data_file)) :
    #         self.logger.write("\tcreating training data cpkl file from raw source")
    #         self.preprocess(stroke_dir, ascii_dir, data_file)

    #     self.load_preprocessed(data_file)
    #     self.reset_batch_pointer()

def preprocess(out_template):
    # create data file from raw xml files from iam handwriting source.
    print("\treading labels...")
    train_labels = idx2numpy.convert_from_file('/develop/data/mnist/download/train-labels-idx1-ubyte.idx')
    test_labels = idx2numpy.convert_from_file('/develop/data/mnist/download/t10k-labels-idx1-ubyte.idx')
    
    def getStrokes(is_test, idx):
        if is_test:
            pathfile = mnist_test_path_template.format(idx)
        else:
            pathfile = mnist_train_path_template.format(idx)

        stroke_list = []
        with open(pathfile) as f:
            for line_terminated in f:
                line = line_terminated.rstrip('\n')
                nums = map(int, line.split())
                stroke_list.append(nums)

        strokes = np.array(stroke_list, dtype=np.int16)[:,:3]
        return strokes

    # function to read each individual xml file
    def getAscii(is_test, idx):
        if is_test:
            return str(test_labels[idx])
        else:
            return str(train_labels[idx])

    # pad strokes to equal length
    def strokes_postprocess(strokes, longest_stroke):
        padlen = longest_stroke + 1
        numstrokes = len(strokes)
        stroke_data = np.zeros((numstrokes, padlen, 3), dtype=np.int16)
        for n in range(numstrokes):
            len_current_stroke = len(strokes[n])
            stroke_data[n][:len_current_stroke] = strokes[n]
        return np.array(stroke_data)

    # pad ascii strings to equal length
    def ascii_postprocess(ascii, longest_ascii):
        padlen = longest_ascii + 1
        ascii_data = []
        for s in ascii:
            ascii_data.append(s.ljust(padlen, '_'))
        return ascii_data

    print("\treading strokes...")
    strokes = []
    asciis = []
    alphabet = set()
    longest_stroke = 0
    longest_ascii = 0
    for i in range(70000):
        if i%5000 == 0:
            print("Stroke {}".format(i))
        is_test = (i >= 60000)
        if is_test:
            idx = i - 60000
        else:
            idx = i
        stroke = getStrokes(is_test, idx)
        ascii = getAscii(is_test, idx)
        alphabet = alphabet.union(set(ascii))
        if len(stroke) > max_allowable_stroke:
            print("stroke for {} is too long ({}), skipping".format(ascii, len(stroke)))
        elif len(ascii) > max_allowable_ascii:
            print("ascii for {} is too long, skipping".format(ascii))
        else:
            if len(stroke) > longest_stroke:
                longest_stroke = len(stroke)
            if len(ascii) > longest_ascii:
                longest_ascii = len(ascii)
            strokes.append(stroke)
            asciis.append(ascii)

    print("Longest stroke: {}".format(longest_stroke))
    print("Longest ascii: {}".format(longest_ascii))
    assert(len(strokes)==len(asciis)), "There should be a 1:1 correspondence between stroke data and ascii labels."
    strokes = strokes_postprocess(strokes, longest_stroke)
    # asciis = ascii_postprocess(asciis, longest_ascii)
    strokes_train = strokes[:60000]
    strokes_test = strokes[60000:]
    asciis_train = asciis[:60000]
    asciis_test = asciis[60000:]
    stroke_length = strokes[0].shape[0]
    ascii_length = len(asciis[0])
    alphabet_string = "".join(sorted(alphabet))
    metadata = {
        'stroke_length': stroke_length,
        'ascii_length': ascii_length,
        'alphabet': alphabet_string
        }
    print("Metadata: {}".format(metadata))
    data_file = out_template.format("train")
    f = open(data_file,"wb")
    pickle.dump([metadata,strokes_train,asciis_train], f, protocol=2)
    f.close()
    data_file = out_template.format("test")
    f = open(data_file,"wb")
    pickle.dump([metadata,strokes_test,asciis_test], f, protocol=2)
    f.close()
    print("\tfinished parsing dataset. saved {} lines".format(len(strokes)))

def main():
    parser = argparse.ArgumentParser()

    #general model params
    parser.add_argument('--out-template', type=str, default='data/mnist_strokes_{}.cpkl', help='save file with test/train blank')
    args = parser.parse_args()
    preprocess(args.out_template)

if __name__ == '__main__':
    main()
