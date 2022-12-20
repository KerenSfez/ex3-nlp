import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data_loader
import pickle
from matplotlib import pyplot

# --------------------------------- Constants ---------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------- Helper methods and classes ------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster
    when a GPU with cuda is available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on
    the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or
    evaluation can be executed later on.
    :param epoch: Epoch
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with
    save_model
    :param model: should be the same model as the one which was saved in the
    path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved
    in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------- Data utilities ------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the
    words consisting the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    average = np.zeros(embedding_dim)
    known_count = 0

    for w in sent.text:
        if w in word_to_vec:
            average += word_to_vec[w]
            known_count += 1

    if known_count != 0:
        average /= known_count
        return average

    return average


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is
    placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    o_h = np.zeros(size, dtype=np.float64)
    o_h[ind] = 1
    return o_h


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and
    returns the average one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """

    index = list()

    for word in sent.text:
        index.append(word_to_ind[word])

    numerator = get_one_hot(len(word_to_ind), index)

    denominator = np.sum(numerator)

    return numerator/denominator


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {words_list[i]: i for i in range(len(words_list))}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a
    list containing the words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the
    representation of the sentence
    """
    sentences = sent.text[:min(len(sent.text), seq_len)]

    embed = list()
    for i in range(len(sentences)):
        word = sentences[i]
        if word in word_to_vec:
            embed.append(word_to_vec[word])
        else:
            embed.append(np.zeros(embedding_dim))

    while len(embed) < seq_len:
        embed.append(np.zeros(embedding_dim))

    return np.array(embed)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of
    SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input
        datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get
    iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all
        sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader. \
            SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset. \
                get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind":
                                     get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec":
                                     create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec":
                                     create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func,
                                                self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size,
                                              shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the
        datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class
                         for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x,
        ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ----------------------------------- Models ----------------------------------

def predict_helper(obj, x):
    probabilities = torch.sigmoid(obj.forward(x)).detach().numpy() \
        .flatten()
    predictions = np.zeros((probabilities >= 0.5).shape)
    predictions[(probabilities >= 0.5)] = 1

    return predictions


class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the
    exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=embedding_dim, bidirectional=True,
                            batch_first=True, hidden_size=hidden_dim,
                            num_layers=n_layers)
        self.linear = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        return

    def forward(self, text):
        hid_lstm = self.LSTM(text)[1][0]

        return self.linear(self.dropout(torch.cat((hid_lstm[-2, :, :],
                                                   hid_lstm[-1, :, :]),
                                                  dim=1)))

    def predict(self, text):
        return predict_helper(self, text)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

        return

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return predict_helper(self, x)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the
    labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value -
            (<number of accurate predictions> / <number of examples>)
    """

    return np.mean(preds == y)


def data_training(model, data_iterator, criterion, optimizer=None):
    loss, acc, labs = list(), list(), list()

    for element in data_iterator:
        data = element[0]
        label = element[1]

        prediction = model(data.float())
        current_loss = criterion(prediction.flatten(), label)

        if optimizer:
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

        loss.append(current_loss.item())

        prediction = torch.sigmoid(prediction)
        probabilities = prediction.detach().numpy().flatten()
        label_prediction = np.zeros(probabilities.shape)
        label_prediction[probabilities >= 0.5] = 1
        labs.extend(label_prediction)

        acc.append(binary_accuracy(label_prediction, label.detach().numpy()))

    labs = np.array(labs).flatten()
    return np.mean(acc), np.mean(loss), labs


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training
    of the given model, and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for
    the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    return data_training(model, data_iterator, criterion, optimizer)[:2]


def evaluate(model, data_iterator, criterion, train_fun=False):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :param train_fun: If called in train function or not
    :return: tuple of (average loss over all examples, average accuracy over
    all examples)
    """
    if train_fun:
        return data_training(model, data_iterator, criterion)

    return data_training(model, data_iterator, criterion)[:2]


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter
    and return all of the models predictions as a numpy ndarray or torch
    tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = list()

    for element in data_iter:
        predictions.extend(model.predict(element[0]))

    predictions = np.array(predictions)
    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization
    should be done using the Adam optimizer with all parameters but learning
    rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    eval_loss_hist, train_loss_hist, eval_acc_hist, train_acc_hist = \
        list(), list(), list(), list()

    count = 0
    while count < n_epochs:
        train_acc, train_loss, = \
            train_epoch(model, data_manager.get_torch_iterator(),
                        torch.optim.Adam(params=model.parameters(),
                                         lr=lr, weight_decay=weight_decay),
                        nn.BCEWithLogitsLoss())
        eval_acc, eval_loss = evaluate(model,
                                       data_manager.get_torch_iterator(
                                           data_subset=VAL),
                                       nn.BCEWithLogitsLoss())

        eval_loss_hist.append(eval_loss)
        train_loss_hist.append(train_loss)
        eval_acc_hist.append(eval_acc)
        train_acc_hist.append(train_acc)

        count += 1

    return eval_loss_hist, train_loss_hist, eval_acc_hist, train_acc_hist


def graphs(loss, acc, first_label, second_label, model_name, ylabel):
    pyplot.plot(loss, label=first_label)
    pyplot.plot(acc, label=second_label)

    pyplot.title(model_name)
    pyplot.xlabel("Epoch Number")

    pyplot.ylabel(ylabel)
    pyplot.legend()

    pyplot.show()


def get_graphs(h, name):
    graphs(h[1], h[0], "Train Loss", "Validation Loss",
           name, "Loss")
    graphs(h[3], h[2], "Train Accuracy", "Validation Accuracy",
           name, "Accuracy Value")


def get_rares_and_negs(dm):
    rares = data_loader.get_rare_words_examples(dm.sentiment_dataset.
                                                get_test_set(),
                                                dm.sentiment_dataset)
    negs = data_loader.get_negated_polarity_examples(dm.sentiment_dataset.
                                                     get_test_set())

    return rares, negs


def get_accuracies(dm, predication_lab, rares, negs):
    rares_acc = np.mean(dm.get_labels(TEST)[rares] == predication_lab[rares])
    negs_acc = np.mean(dm.get_labels(TEST)[negs] == predication_lab[negs])

    return rares_acc, negs_acc


def t_model_helper(model, name, dm, weight_decay, learning_rate, ep_num):
    h = train_model(model, dm, ep_num, learning_rate, weight_decay)

    test_acc, test_loss, predication_lab = evaluate(model,
                                                    dm.get_torch_iterator(
                                                        data_subset=TEST),
                                                    nn.BCEWithLogitsLoss(),
                                                    train_fun=True)

    get_accuracies(dm, predication_lab, *get_rares_and_negs(dm))

    get_graphs(h, name)


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model
    with one hot representation.
    """
    t_model_helper(LogLinear(DataManager(batch_size=64).get_input_shape()[0]),
                   "Log Linear - One hot embedding",
                   DataManager(batch_size=64),
                   0.0001, 0.01, 20)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model
    with word embeddings
    representation.
    """
    t_model_helper(LogLinear(DataManager(W2V_AVERAGE, batch_size=64,
                                         embedding_dim=300).
                             get_input_shape()[0]),
                   "Log Linear - Word2Vector embedding",
                   DataManager(W2V_AVERAGE, batch_size=64, embedding_dim=300),
                   0.0001, 0.01, 20)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    t_model_helper(LSTM(300, 100, 2, 0.5),
                   "LSTM - Word2Vector",
                   DataManager(W2V_SEQUENCE, batch_size=64, embedding_dim=300),
                   0.0001, 0.001, 4)


if __name__ == '__main__':
    print("****-------- LOG Linear with one hot --------****\n\n")
    train_log_linear_with_one_hot()

    print("****-------- LOG Linear with w2v --------****\n\n")
    train_log_linear_with_w2v()

    print("****-------- LSTM with w2v --------****\n\n")
    train_lstm_with_w2v()
