import os
import collections
import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print "reading text file"
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print "loading preprocessed files"
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = list(zip(*count_pairs))
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'w') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(map(self.vocab.get, data))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

        newline = self.vocab['\n']
        nextnew = np.where(self.tensor == newline)
        n_tunes = nextnew[0].size

        print "preparing chunks..."
        pos = 0
        n_tune = 0
        x_chunks = np.array([], dtype=int)
        y_chunks = np.array([], dtype=int)
        for nl in nextnew[0]:
            while (pos + self.seq_length < nl):
                x_chunks = np.append(x_chunks, self.tensor[pos : pos + self.seq_length])
                y_chunks = np.append(y_chunks, self.tensor[pos + 1 : pos + self.seq_length + 1])
                pos += 1
            n_tune += 1
            print "done tune ", n_tune, "/", n_tunes
            pos += self.seq_length

        print "reshaping to seq_length..."
        x_chunks = np.copy(np.reshape(x_chunks, (-1, self.seq_length)))
        y_chunks = np.copy(np.reshape(y_chunks, (-1, self.seq_length)))

        print "truncating to match batch_size"
        x, y = x_chunks.shape
        self.num_batches = x / self.batch_size
        print "%i batches" % self.num_batches
        x_chunks.resize((self.num_batches * self.batch_size, self.seq_length))
        y_chunks.resize((self.num_batches * self.batch_size, self.seq_length))

        self.x_batches = np.vsplit(x_chunks, self.num_batches)
        self.y_batches = np.vsplit(y_chunks, self.num_batches)
        print "batches ready!"

        #### NB ####
        #
        # It might be an idea to randomize the X/Y pairs in the batches,
        # for that we need a tmp array containing them. Some hstack/vsplit involved.
        # Following is just an unfinished draft of that
        #
        ############
        #
        # print "stacked: "
        # xy_stacks = np.hstack((x_chunks, y_chunks))
        # x, y = xy_stacks.shape
        # print x
        # num_batches = x / self.batch_size
        # print num_batches
        # xy_stacks.resize(num_batches * self.batch_size, y)
        # print xy_stacks.shape
        # xy_stacks.resize((num_batches * self.batch_size * 2, self.seq_length))
        # print "state of the stuff "
        # print xy_stacks
        # # np.random.shuffle(chunks)
        # # print chunks

        #### ORIGINAL CODE ####
        # self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        # xdata = self.tensor
        # ydata = np.copy(self.tensor)
        # ydata[:-1] = xdata[1:]
        #### necessary, but quite weird! (y_n = x_0...)
        # ydata[-1] = xdata[0]
        # self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        # self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

