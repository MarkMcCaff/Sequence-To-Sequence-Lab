# Based on un-cleaned version of rnnclassifier-LM-template.py
# based on "shuffle-detection-RNN-answer.py", with necessary changes copied from working "rnnclassifier-LM-GRU.py"
import random
import aesara as theano
import collections
import numpy

#Total number of classes we need to map our sequences to.
n_words = 200 # Total number of words we can support: 200 sufficient for our task since our vocabulary is small
n_classes = n_words # Since we are predicting a word to generate based on already generated words

#We represent the input sequence as a vector of integers (word id-s):
input_indices = theano.tensor.ivector('input_indices')
#We want to predict:
target_class = theano.tensor.iscalar('target_class') #e.g. could be sentiment level
#All words in the language are represented as trainable vectors:
word_embedding_size = 10    #the size of those vectors

rng = numpy.random.RandomState(0)
def random_matrix(num_rows, num_columns):
    return numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(num_rows, num_columns)))

word_embeddings = theano.shared(random_matrix(n_words, word_embedding_size), 'word_embeddings')
#word_embeddings = theano.shared(numpy.ones((n_words, word_embedding_size)), 'word_embeddings')
#This represents the input sequence (e.g. a sentence):
input_vectors = word_embeddings[input_indices]

recurrent_size = 100
W_x = theano.shared(random_matrix(recurrent_size, word_embedding_size), 'W_x')
W_h = theano.shared(random_matrix(recurrent_size, recurrent_size), 'W_h')

W_xz = theano.shared(random_matrix(recurrent_size, word_embedding_size), 'W_xz')  # C
W_hz = theano.shared(random_matrix(recurrent_size, recurrent_size), 'W_hz')  #
W_xr = theano.shared(random_matrix(recurrent_size, word_embedding_size), 'W_xr')  #
W_hr = theano.shared(random_matrix(recurrent_size, recurrent_size), 'W_hr')  #


def rnn_step(x, h_prev, W_x, W_h, W_xz, W_hz, W_xr, W_hr):  # C

    r = theano.tensor.math.sigmoid(theano.tensor.dot(W_xr, x) + theano.tensor.dot(W_hr, h_prev))  # C
    z = theano.tensor.math.sigmoid(theano.tensor.dot(W_xz, x) + theano.tensor.dot(W_hz, h_prev))  # C
    _h = theano.tensor.tanh(theano.tensor.dot(W_x, x) + theano.tensor.dot(W_h, r * h_prev))  # C
    return z * h_prev + (1.0 - z) * _h  # E: in theano * stands for elemement wise mutiplication
'''def rnn_step(x, h_prev, W_x, W_h): #implements simple RNN recurrence step
    return theano.tensor.tanh(theano.tensor.dot(W_h, h_prev) + theano.tensor.dot(W_x, x))'''
context_vector, other_info = theano.scan(
            rnn_step,
            sequences = input_vectors,
            outputs_info=numpy.zeros(recurrent_size),
            non_sequences=[W_x, W_h, W_xz, W_hz, W_xr, W_hr]  #C
            #non_sequences = [W_x, W_h]
        )
context_vector = context_vector[-1]

W_output = theano.shared(random_matrix(n_classes, recurrent_size), 'W_output')
#W_output = theano.shared(random_matrix(2, recurrent_size), 'W_output')
activations = theano.tensor.dot(W_output, context_vector)

predicted_class = theano.tensor.argmax(activations)
output = theano.tensor.special.softmax([activations])[0]
cost = -theano.tensor.log(output[target_class]) #We use cross-entropy: It works better with multiple classes
learning_rate = theano.tensor.fscalar('learning_rate') #e.g. could be sentiment level
#learning_rate = .1 #
#learning_rate = .01
updates = [ #now reducing cost so using '-' for gradient descent, not '+' for ascent
    #make sure all the new trainable parameters added:
    (word_embeddings, word_embeddings - learning_rate*theano.tensor.grad(cost, word_embeddings)),
    (W_output, W_output - learning_rate*theano.tensor.grad(cost, W_output)),
    (W_x, W_x - learning_rate*theano.tensor.grad(cost, W_x)),
    (W_h, W_h - learning_rate * theano.tensor.grad(cost, W_h)),
    (W_xz, W_xz - learning_rate * theano.tensor.grad(cost, W_xz)),  # C
    (W_hz, W_hz - learning_rate * theano.tensor.grad(cost, W_hz)),  # C
    (W_xr, W_xr - learning_rate * theano.tensor.grad(cost, W_xr)),  # C
    (W_hr, W_hr - learning_rate * theano.tensor.grad(cost, W_hr)),  # C
]

train = theano.function([input_indices, target_class, learning_rate], [cost, predicted_class], updates=updates, allow_input_downcast = True)
#train = theano.function([input_indices, target_class], [cost, predicted_class], updates=updates, allow_input_downcast = True)
test = theano.function([input_indices, target_class], [cost, predicted_class], allow_input_downcast = True)

def read_dataset(path):
    """Read a dataset, where the first column contains a real-valued score,
    followed by a tab and a string of words.
    """
    dataset = []
    with open(path, "r") as f:
        for line in f:
            assert len(line) >= 4
            assert '\t' in line
            line_parts = line.strip().split("\t") #E the parts
            assert line_parts[0] == "<s>" or "<s>" in line_parts[1]
            dataset.append((line_parts[0], line_parts[1]))
            if "</s>" in line_parts[0]:
                sentenceEndFound = True

    assert sentenceEndFound
    return dataset

id2word  = []

def create_dictionary(sentences): #this is indexing! we need to convert all words to their ID-s
    counter = collections.Counter() #Python's class that can count
    for sentence in sentences:
        for word in sentence:
            counter.update([word])

    word2id = collections.OrderedDict() #Python's class that can map words to ID-s
    word2id["<unk>"] = 0    #We reserve this for "uknown words" that we may encounter in the future
    word2id["<s>"] = 1 #Marks beginning of the sentence
    word2id["</s>"] = 2 #Marks the end of the sentence

    id2word.append("<unk>")
    id2word.append("<s>")
    id2word.append("</s>")

    word_count_list = counter.most_common() #For every word, we create an entry in  'word2id'
    for (word, count) in word_count_list: #so it can map them to their ID-s
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word.append(word)

    for word in word2id: #Verifying that our mapping between words and their id-s is consistent
        assert id2word[word2id[word]] == word

    return word2id

def sentence2ids(words, word2id): #Converts a word sequence (sentence) into a list of ID-s
    ids = []
    for word in words:
        if word in word2id: #This is how it was in "shuffle-detection-template.py"
            ids.append(word2id[word])
        else:
            ids.append(word2id["<unk>"])
    return ids


path_train = "shuffled-geo-train-lm.txt"
path_test = "shuffled-geo-test-lm.txt"
sentences_train = read_dataset(path_train)
sentences_test = read_dataset(path_test)

word2id = create_dictionary([sentence.split() for label, sentence in sentences_train+sentences_test])
n_words = len(word2id)  # Important to set it c

data_train = [(word2id[label], sentence2ids(sentence.split(), word2id)) for label, sentence in sentences_train]
data_test = [(word2id[label], sentence2ids(sentence.split(), word2id)) for label, sentence in sentences_test]
random.shuffle(data_train)

def generate_sentence(starting_text): #E explain high level how works
         number_of_words_generated = 0
         already_generated_text = "<s> " + starting_text + " "  #E rename to 'already generated?'
         sv =  sentence2ids(already_generated_text.split(), word2id)
         while  True:
              cost, predicted_class = test(sv, 0)  #any class label ok to give here, so we are passing 0.
              number_of_words_generated += 1
              if predicted_class == word2id["</s>"] or number_of_words_generated > 100 or predicted_class >= len(id2word):
                   break
              word = id2word[predicted_class] #E how to convert id-already_generated_text back to words
              sv.append(predicted_class)
              already_generated_text += word + " "
         print (already_generated_text)

for epoch in range(100):


        cost_sum = 0.0
        correct = 0
        count = 0
        learning_rate = 0.03
        for target_class, sentence in data_train:
            count += 1
            cost, predicted_class = train(sentence, target_class, learning_rate)
            #cost, predicted_class = train(sentence, target_class)
            cost_sum += cost
            if predicted_class == target_class:
                correct += 1
        print("Epoch: " + str(epoch) + "\tAverage Cost: " + str(cost_sum/count) + "\tAccuracy: " + str(float(correct)/count))
        if float(correct)/count > .50:
            learning_rate  = 0.003
        #print ("Epoch: " + str(epoch) + "\tCost: " + str(cost_sum) + "\tAccuracy: " + str(float(correct)/count))
        cost_sum2 = 0.0
        correct2 = 0
        for target_class, sentence in data_test:
              cost, predicted_class = test(sentence, target_class)
              cost_sum2 += cost
              if predicted_class == target_class:
                correct2 += 1
        print ("\t\t\t\t\t\t\tTest_cost: " + str(cost_sum2/len(data_test)) + "\tTest_accuracy: " + str(float(correct2)/len(data_test)))
        generate_sentence("there is")
        generate_sentence("there are")
        generate_sentence("the")
        generate_sentence("at least")
        generate_sentence("there is 1")
        generate_sentence("each")
        generate_sentence("one")


