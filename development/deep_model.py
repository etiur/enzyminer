import numpy as np
np.random.seed(10)
from tensorflow.random import set_seed
set_seed(2)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, Embedding, LSTM, MaxPooling1D, Conv1D
from tensorflow.keras.layers import concatenate, GlobalMaxPooling1D, AveragePooling1D, Flatten, Layer
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adamax, Adam
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from kerastuner.tuners import Hyperband
import argparse
from blist import blist
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report as class_re
from sklearn.model_selection import StratifiedKFold
from collections import namedtuple
import warnings
plt.switch_backend('agg')
warnings.filterwarnings("error", category=FutureWarning)


def arg_parse():
    parser = argparse.ArgumentParser(description="generate deep learning models")
    parser.add_argument("-p", "--protein_dataset", required=False,
                        help="The dataset with the protein sequences in excel",
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx")
    parser.add_argument("-v", "--prot2vec", required=False, help="The prot2vec encoding file",
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/data/protVec_100d_3grams.csv")
    parser.add_argument("-l", "--log_folder", required=False, help="The folder name for the tensorboard logs",
                        default="tensorflow_logs")
    parser.add_argument("-d", "--plot_dir", required=False, help="the name of the directory for the plots",
                        default="metrics_plot")
    parser.add_argument("-c", "--csv_dir", required=False, help="the name of the directory for the metrics",
                        default="metrics_csv")
    parser.add_argument("-e", "--epochs", required=False, type=int, help="the number of epochs to run the fit",
                        default=90)
    parser.add_argument("-r", "--restart", required=False, help="Indicate the topology to restart with",
                        choices=("cnn", "rnn", "inception", "rnn_cnn", "rnn_inception"), default=None)
    parser.add_argument("-o", "--only", required=False, help="Indicate which topology to train", nargs="+",
                        choices=("cnn", "rnn", "inception", "rnn_cnn", "rnn_inception"), default=None)
    parser.add_argument("-t", "--train_model", required=False, action="store_true",
                        help="If train the models using the search hyperparameters")
    args = parser.parse_args()

    return [args.protein_dataset, args.prot2vec, args.log_folder, args.plot_dir, args.csv_dir, args.epochs, args.restart,
            args.train_model, args.only]


class GetConfig:
    """
    Get the configuration of a model
    """
    def __init__(self, model, path, name):
        """
        Get the configurations of the model

        Parameters
        ----------
        model: Model
            A keras model object
        path: str
            Name of the folder to keep the csv
        name: str
            The csv file name
        """
        self.config = model.get_config()["layers"]
        self.layers = {f'{self.config[i]["class_name"]}_{i}': self.config[i]["config"] for i in range(len(self.config))}
        self.optimizer = model.optimizer.get_config()
        self.path = path
        self.name = name

    def dataframe(self):
        """
        Converts the configurations into dataframes
        Returns
        -------
        dataframe: DataFrame
            A pandas dataframe object
        """
        new_dict = {}
        for key, value in self.layers.items():
            if "Dense" in key:
                new_dict[key] = [value.get("units", 0), value.get('activation', 0), 0]
            elif "Bidirectional" in key:
                new_dict[key] = [value["layer"]["config"].get("units", 0), value["layer"]["config"].get('dropout', 2),
                                 value["layer"]["config"].get('recurrent_dropout', 2)]
            elif "Dropout" in key:
                new_dict[key] = [value.get("rate", 2), 6, 7]
            elif 'MaxPooling1D' in key or 'AveragePooling1D' in key:
                new_dict[key] = [value.get('pool_size', (0,))[0], value.get('strides', (0,))[0], value.get('padding', 0)]

            elif 'Conv1D' in key:
                new_dict[key] = [value.get('filters', 0), value.get('kernel_size', (0,))[0], value.get('padding', 0)]

            elif 'Embedding' in key:
                new_dict[key] = [value.get('output_dim', 0), value.get('input_dim', 0), 10]
            else:
                new_dict[key] = [value.get("units", 0), 9, 10]

        new_dict["optimizer"] = [self.optimizer.get("name", 0), 10, 10]

        return pd.DataFrame(new_dict)

    def to_csv(self):
        """
        Write the dataframe to CSV format
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        frame = self.dataframe()
        frame.to_csv(f"{self.path}/{self.name}", header=False)


class MyAttention(Layer):
    """
    My own implementation of the attention mechanism
    """
    def __init__(self, **kwargs):
        super(MyAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(MyAttention, self).build(input_shape)

    def call(self, x, **kwargs):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        return super(MyAttention, self).get_config()


class GenerateScore:
    """
    Generate different classification scores from the probabilities
    """
    def __init__(self, y_true_train, y_pred_train, y_test_true, y_test_pred, fold):
        """
        Initialize the GenerateScore class

        Parameters
        ----------
        y_true_train: array
            The true labels for the train set
        y_pred_train array
            The predicted labels for the train set
        y_test_true: array
            The true labels for the test set
        y_test_pred: array
            The predicted labels for the test
        fold: int, optional
            The current fold number
        """
        self.y_true_train = np.argmax(y_true_train, axis=1)
        self.y_pred_train = np.argmax(y_pred_train, axis=1)
        self.y_true_test = np.argmax(y_test_true, axis=1)
        self.y_pred_test = np.argmax(y_test_pred, axis=1)
        self.fold = fold

    def scores(self):
        """
        Generate all the score from the predictions and the true labels

        Returns
        -------
        every_score: namedtuple
            A namedtuple that contains every score
        """
        target_names = ["class 0", "class 1"]
        name_tuple = namedtuple("score", ["test_metrics", "train_metrics", "tr_report", "te_report"])
        matrix = namedtuple("confusion_matrix", ["tn", "fp", "fn", "tp"])

        # Training scores
        train_confusion = confusion_matrix(self.y_true_train, self.y_pred_train)
        train_confusion = matrix(*train_confusion.ravel())
        tr_report = class_re(self.y_true_train, self.y_pred_train, target_names=target_names, output_dict=True)
        tr_report = pd.DataFrame(tr_report).transpose()
        tr_report.columns = [f"{x}_{self.fold}" for x in tr_report.columns]
        train_mat = matthews_corrcoef(self.y_true_train, self.y_pred_train)
        train_metrics = {"mcc": train_mat, "tn": train_confusion.tn, "fp": train_confusion.fp, "fn": train_confusion.fn,
                         "tp": train_confusion.tp}
        train_metrics = pd.DataFrame(train_metrics, index=[f"model_{self.fold}"])

        # Test scores
        test_confusion = confusion_matrix(self.y_true_test, self.y_pred_test)
        test_confusion = matrix(*test_confusion.ravel())
        test_mat = matthews_corrcoef(self.y_true_test, self.y_pred_test)
        te_report = class_re(self.y_true_test, self.y_pred_test, target_names=target_names, output_dict=True)
        te_report = pd.DataFrame(te_report).transpose()
        te_report.columns = [f"{x}_{self.fold}" for x in te_report.columns]
        test_metrics = {"mcc": test_mat, "tn": test_confusion.tn, "fp": test_confusion.fp, "fn": test_confusion.fn,
                        "tp": test_confusion.tp}
        test_metrics = pd.DataFrame(test_metrics, index=[f"model_{self.fold}"])
        # keeping all the scores
        score = [test_metrics, train_metrics, tr_report, te_report]
        every_score = name_tuple(*score)

        return every_score


class Encode:
    """
    A class that performs one hot encoding and prot2vec encoding of the sequences
    """

    def __init__(self, protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                 vec="/gpfs/home/bsc72/bsc72661/feature_extraction/data/protVec_100d_3grams.csv"):
        """
        Initialize the Encoding class

        Parameters
        ___________
        protein_dataset: str, optional
            The path to the protein sequences
        vec: str, optional
            The path to the prot2vec encoding
        """
        self.sequences = pd.read_excel(protein_dataset, engine='openpyxl', usecols=["enzyme", "seq", "label"],
                                       index_col=0)
        self.protvec = pd.read_csv(vec, sep="\t", index_col=0)
        length = [len(seq) for seq in self.sequences["seq"]]
        self.maxlen = int(sum(length) / len(length) + 100)
        self.categorical_label = to_categorical(self.sequences["label"])

    def _padding_seq(self, seq):
        """
        Pads the sequences that are shorter than maxlen

        parameters
        ___________
        seq: str
            The protein sequence

        returns
        ________
        seq: str
            Padded protein sequence
        """
        if len(seq) >= self.maxlen:
            return seq[:self.maxlen]
        else:
            dif = self.maxlen - len(seq)
            seq_list = list(seq)
            for i in range(dif):
                seq_list.append("Z")
            return "".join(seq_list)

    def _split_in3(self, seq):
        """
        Splits in 3 the sequences

        parameters
        ___________
        seq: str
            A protein sequence

        returns
        ________
        seq_matrix: DataFrame object
            A dataframe of dimension seq length X 100
        """
        seq = self._padding_seq(seq)
        split = [seq[i:i + 3] for i in range(0, len(seq), 3)]
        try:
            seq_matrix = self.protvec.loc[split]
        except FutureWarning:
            mat = []
            for words in split:
                try:
                    embedding = self.protvec.loc[words]
                    mat.append(embedding)
                except KeyError:
                    random_emb = pd.Series(np.random.rand(100), index=self.protvec.columns)
                    mat.append(random_emb)
            seq_matrix = pd.concat(mat, axis=1).transpose()

        return seq_matrix.to_numpy()

    def prot2vec(self):
        """
        Returns a matrix with prot2vec encoding of the sequences
        """
        prot_list = blist()
        for seq in self.sequences["seq"]:
            matrix = self._split_in3(seq)
            prot_list.append(matrix)

        return np.array(prot_list)

    def tokenized(self):
        """
        Encode the sequences 1D, each aa is an index
        """
        tokenizer = Tokenizer(char_level=True, lower=False)
        tokenizer.fit_on_texts(self.sequences["seq"].values)
        sequences = tokenizer.texts_to_sequences(self.sequences["seq"].values)
        data = pad_sequences(sequences, maxlen=self.maxlen)

        return data

    def one_hot(self):
        """
        one-hot encode the sequences 2D
        """
        encoder = OneHotEncoder(sparse=False, drop="first")
        data = self.tokenized()
        data0 = data[0].reshape(len(data[0]), 1)
        transformed = encoder.fit_transform(data0)
        matrix = np.zeros((len(data), transformed.shape[0], transformed.shape[1]))
        for i in range(len(data)):
            matrix[i] = encoder.transform(data[i].reshape(len(data[i]), 1))
        return matrix

    @staticmethod
    def processing(data, label, state=20, split=0.2):
        """
        Process and split the training data

        Parameters
        ____________
        data: list
            A list of the training data
        categorical_label: array[labels], optional
            An array of categorical labels
        state: int, optional
            To make the splitting reproducible
        split: float, optional
            The size of the split test set
        """
        # split the dataset
        x_train, x_val, y_train, y_val = train_test_split(data, label, stratify=label,
                                                          random_state=state, test_size=split)

        return x_train, x_val, y_train, y_val

    @staticmethod
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False


class DeepNetworkSearch:
    """
    A class that generates the different deep learning topologies
    """
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None, x_train_hot=None, x_test_hot=None,
                 protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                 csv_dir="metrics_csv", epochs=80):
        """
        Initialize the DeepNetwork class

        parameters
        ___________
        x_train: list
            A list of the prot2vec training data
        x_test: list
            A list of the prot2vec testing data
        y_train: list
            A list of the training labels
        y_test: list
            A list of the testing labels
        x_train_hot: list
            A list of the one-hot encoded training data
        x_test_hot: list
            A list of the one-hot encoded testing data
        protein_dataset: str
            The path to the dataset
        csv_dir: str, optional
            forlder for the csv
        epochs: int
            The number of training epochs
        """
        self.x_train_vec = x_train
        self.x_test_vec = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_train_hot = x_train_hot
        self.x_test_hot = x_test_hot
        sequences = pd.read_excel(protein_dataset, engine='openpyxl', usecols=["enzyme", "seq", "label"], index_col=0)
        length = [len(seq) for seq in sequences["seq"]]
        self.maxlen = int(sum(length) / len(length) + 100)
        self.shape = self.x_train_vec.shape[1:]
        self.csv = csv_dir
        self.epoch = epochs

    def sequential_rnn(self, hp):
        """
        A sequential deep network of RNN layers
        """
        # hyperparameters
        emb = hp.Int("emb_output", min_value=10, max_value=90, step=10)
        lstm_1 = hp.Int("LSTM_1", min_value=32, max_value=250, step=32, default=64)
        lstm_2 = hp.Int("LSTM_2", min_value=32, max_value=250, step=32, default=64)
        drop_lstm1 = hp.Float("dropout_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        drop_lstm2 = hp.Float("dropout_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop1 = hp.Float("recurrent_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop2 = hp.Float("recurrent_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        units_1 = hp.Int("units_1", min_value=32, max_value=150, step=32, default=64)
        drop = hp.Float("dropout", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        units_2 = hp.Int("units_2", min_value=32, max_value=150, step=32, default=64)
        # model
        model = Sequential()
        model.add(Embedding(21, emb, mask_zero=True, input_length=self.maxlen))
        #model.add(Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1,
        #                             recurrent_dropout=recurrent_drop1)))
        if hp.Choice("attention", ["attention", "no_attention"]) == "no_attention":
            model.add(Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2)))
        else:
            model.add(Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2,
                                         return_sequences=True)))
            model.add(MyAttention())
        # model.add(Dense(units_1, activation="relu"))
        # model.add(Dropout(drop))
        model.add(Dense(units_2, activation="relu"))
        model.add((Dense(2, activation="softmax")))

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.001, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if hp.Choice('optimizer', ['adam', 'adamax']) == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def sequential_cnn(self, hp):
        """
        A sequential deep network of CNN layers
        parameters
        __________
        shape: int, optional
            The input shape of the cnn layer
        """
        # hyperparameters
        conv_f1 = hp.Int("filter1", min_value=32, max_value=250, step=32, default=64)
        kernel_1 = hp.Int("kernel1", min_value=1, max_value=8, step=2, default=6)
        conv_f2 = hp.Int("filter2", min_value=32, max_value=250, step=32, default=64)
        kernel_2 = hp.Int("kernel2", min_value=1, max_value=8, step=2, default=6)
        conv_f3 = hp.Int("filter3", min_value=32, max_value=250, step=32, default=64)
        kernel_3 = hp.Int("kernel3", min_value=1, max_value=8, step=2, default=6)
        units_1 = hp.Int("units_1", min_value=32, max_value=150, step=32, default=64)
        drop_hp = hp.Float("dropout_hp", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        pool_1 = hp.Int("pool1", min_value=1, max_value=3, step=1, default=1)
        pool_2 = hp.Int("pool2", min_value=1, max_value=3, step=1, default=1)
        pool_3 = hp.Int("pool3", min_value=1, max_value=3, step=1, default=1)
        # model
        model = Sequential()
        model.add(Conv1D(conv_f1, kernel_1, activation="relu", input_shape=self.shape))
        model.add(MaxPooling1D(pool_1))
        model.add(Conv1D(conv_f2, kernel_2, activation="relu"))
        model.add(MaxPooling1D(pool_2))
        # model.add(Conv1D(conv_f3, kernel_3, activation="relu"))
        # model.add(MaxPooling1D(pool_3))
        if hp.Choice('sequential_pooling', ['global', 'flatten', "attention"]) == 'global':
            model.add(GlobalMaxPooling1D())
        elif hp.Choice('sequential_pooling', ['global', 'flatten', "attention"]) == 'attention':
            model.add(MyAttention())
        else:
            model.add(Flatten())
        model.add(Dense(units_1, activation="relu"))
        model.add(Dropout(drop_hp))
        model.add(Dense(2, activation="softmax"))
        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if hp.Choice('optimizer', ['adam', 'adamax']) == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def inception_cnn(self, hp):
        """
        An inception model of CNN

        parameters
        __________
        hp: HyperParameters
            Used for the keras Tuner to find the optimal hyperparameters
        """
        # hyperparameters
        conv_f1 = hp.Int("filter1", min_value=32, max_value=250, step=32, default=64)
        kernel_1 = hp.Int("kernel1", min_value=1, max_value=8, step=2, default=6)
        conv_f2 = hp.Int("filter2", min_value=32, max_value=250, step=32, default=64)
        kernel_2 = hp.Int("kernel2", min_value=1, max_value=8, step=2, default=6)
        conv_f3 = hp.Int("filter3", min_value=32, max_value=250, step=32, default=64)
        kernel_3 = hp.Int("kernel3", min_value=1, max_value=8, step=2, default=6)
        conv_f4 = hp.Int("filter4", min_value=32, max_value=250, step=32, default=64)
        kernel_4 = hp.Int("kernel4", min_value=1, max_value=8, step=2, default=6)
        conv_f5 = hp.Int("filter5", min_value=32, max_value=250, step=32, default=64)
        kernel_5 = hp.Int("kernel5", min_value=1, max_value=8, step=2, default=6)
        conv_f6 = hp.Int("filter6", min_value=32, max_value=250, step=32, default=64)
        kernel_6 = hp.Int("kernel6", min_value=1, max_value=8, step=2, default=6)
        units_1 = hp.Int("units_1", min_value=32, max_value=150, step=32, default=64)
        drop_hp = hp.Float("dropout_hp", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        pool_1 = hp.Int("pool1", min_value=1, max_value=3, step=1, default=1)
        pool_2 = hp.Int("pool2", min_value=1, max_value=3, step=1, default=1)
        # model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        branch_a_cnn = Conv1D(conv_f1, kernel_1, activation="relu", strides=2, padding="same")(input_cnn)
        # branch_b_cnn = Conv1D(conv_f2, kernel_2, activation="relu", padding="same")(input_cnn)
        # branch_b_cnn = Conv1D(conv_f3, kernel_3, activation="relu", strides=2, padding="same")(branch_b_cnn)
        average_pool_c = AveragePooling1D(pool_1, strides=2, padding="same")(input_cnn)
        branch_c_cnn = Conv1D(conv_f4, kernel_4, activation="relu", padding="same")(average_pool_c)
        # branch_d_cnn = Conv1D(conv_f5, kernel_5, activation="relu", padding="same")(input_cnn)
        # branch_d_cnn = Conv1D(conv_f6, kernel_6, activation="relu", strides=2, padding="same")(branch_d_cnn)
        inception = concatenate([branch_a_cnn, branch_c_cnn])
        if hp.Choice('inception_pooling', ['global', 'flatten', "attention"]) == 'global':
            inception = MaxPooling1D(pool_2)(inception)
            flat = GlobalMaxPooling1D()(inception)
        elif hp.Choice('inception_pooling', ['global', 'flatten', "attention"]) == 'attention':
            flat = MyAttention()(inception)
        else:
            inception = MaxPooling1D(pool_2)(inception)
            flat = Flatten()(inception)
        # dense1 = Dense(units_1, activation="relu")(flat)
        # drop = Dropout(drop_hp)(flat)
        output = Dense(2, activation="softmax")(flat)
        model = Model(input_cnn, output)
        # compile
        learning = PolynomialDecay(initial_learning_rate=0.001, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if hp.Choice('optimizer', ['adam', 'adamax']) == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def search(self, mode):
        """
        Search for the hyperparameters

        Parameters
        ___________
        mode: str
            The type of network topology to train
        log_folder: str, optional
            The name of the log folder
        """
        call_list = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=15)]
        call_list_rnn = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=10)]
        if mode == "rnn":
            tuner = Hyperband(self.sequential_rnn, objective="val_categorical_crossentropy", max_epochs=self.epoch-60,
                              seed=50, overwrite=True)
            tuner.search(self.x_train_hot, self.y_train, shuffle=True, validation_data=(self.x_test_hot, self.y_test),
                         epochs=self.epoch, callbacks=call_list_rnn, batch_size=40)
        elif mode == "cnn":
            tuner = Hyperband(self.sequential_cnn, objective="val_categorical_crossentropy", max_epochs=self.epoch-55,
                              seed=40, overwrite=True)
            tuner.search(self.x_train_vec, self.y_train, shuffle=True, validation_data=(self.x_test_vec, self.y_test),
                         epochs=self.epoch, callbacks=call_list, batch_size=40)
        elif mode == "inception":
            tuner = Hyperband(self.inception_cnn, objective="val_categorical_crossentropy", max_epochs=self.epoch-55,
                              seed=12, overwrite=True)
            tuner.search(self.x_train_vec, self.y_train, shuffle=True, validation_data=(self.x_test_vec, self.y_test),
                         epochs=self.epoch, callbacks=call_list, batch_size=40)
        K.clear_session()
        return tuner

    def analyse_search(self, x_train, x_test, mode, log_folder="tensorflow_logs", fold=1, model=None):
        """
        analyse the hyperparameters search results

        Parameters
        ___________
        x_train: array
            The dataset for the training
        x_test: array
            The dataset for evaluation
        mode: str
            The type of network topology to train
        log_folder: str, optional
            The name of the log folder
        fold: int
            The current fold number
        model: Model
            The keras model object
        """
        if not os.path.exists(f"{self.csv}/{mode}"):
            os.makedirs(f"{self.csv}/{mode}")
        if model is None:
            tuner = self.search(mode)
            best_hps = tuner.get_best_hyperparameters()[0]
            # writing the best hyperparameters in file
            best = pd.Series(best_hps.values)
            best.to_csv(f"{self.csv}/{mode}/hps_{fold}.csv", header=False)
            model = tuner.hypermodel.build(best_hps)
            config = GetConfig(model, f"model/{mode}", f"hps_{fold}.csv")
            config.to_csv()
        if not os.path.exists(f"{log_folder}_{mode}"):
            os.makedirs(f"{log_folder}_{mode}")
        call_list = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=15)]
        if mode != "rnn":
            history = model.fit(self.x_train_vec, self.y_train, shuffle=True,
                                validation_data=(self.x_test_vec, self.y_test), epochs=self.epoch, callbacks=call_list,
                                batch_size=40)
        else:
            history = model.fit(self.x_train_hot, self.y_train, shuffle=True,
                                validation_data=(self.x_test_hot, self.y_test), epochs=self.epoch, callbacks=call_list,
                                batch_size=40)
        history = pd.DataFrame(history.history)
        history.index = [f"model_{fold}" for _ in range(len(history))]
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        return history, model, train_pred, test_pred

    def iterative(self, mode, steps=2):
        """
        iteratively runs the model

        Parameters
        ___________
        mode: str
            The type of network topology to train
        steps: int, optional
            How many iterations of the model training to perform
        log_folder: str, optional
            The name of the log folder
        state: int, optional
            The rnadom state used for dataset splitting
        """

        # build the model
        tuner = self.search(mode)
        best_hps = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.x_train_hot, self.y_train, shuffle=True,
                            validation_data=(self.x_test_hot, self.y_test), epochs=self.epoch)
        history_list = []
        model_list = []
        history_list.append(history)
        model_list.append(model)
        if mode == "rnn":
            for i in range(steps):
                pred = model.predict(self.x_train_hot)
                self.x_train_hot = [np.append(self.x_train_hot[i], (pred[i])) for i in range(len(self.x_train_hot))]
                self.x_train_hot = np.array(self.x_train_hot)
                # rebuild the model with the best hyperparameters and train again
                tuner = self.search(mode)
                best_hps = tuner.get_best_hyperparameters()[0]
                model = tuner.hypermodel.build(best_hps)
                history = model.fit(self.x_train_hot, self.y_train, shuffle=True,
                                    validation_data=(self.x_test_hot, self.y_test), epochs=self.epoch)
                history_list.append(history)
                model_list.append(model)

        return history_list, model_list


class MultiInputNetworkSearch:
    """
    Generate a multinput netowrk
    """
    def __init__(self, x_train, y_train, x_train_hot,
                 protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                 csv_dir="metrics_csv", epochs=80):
        """
        Initialize the DeepNetwork class

        parameters
        ___________
        x_train: list
            A list of the prot2vec training data
        y_train: list
            A list of the training labels
        x_train_hot: list
            A list of the one-hot encoded training data
        protein_dataset: str
            The path to the dataset
        csv_dir: str: optional
            The name of the folder for the csv
        epochs: int, optional
            The number of training epochs
        """
        self.x_train_vec = x_train
        self.y_train = y_train
        self.x_train_hot = x_train_hot
        sequences = pd.read_excel(protein_dataset, engine='openpyxl', usecols=["enzyme", "seq", "label"], index_col=0)
        length = [len(seq) for seq in sequences["seq"]]
        self.maxlen = int(sum(length) / len(length) + 100)
        self.shape = self.x_train_vec.shape[1:]
        self.csv = csv_dir
        self.epoch = epochs

    def rnn_cnn(self, hp):
        """
        parameters
        __________
        hp: HyperParameters
            Used for the keras Tuner to find the optimal hyperparameters
        """
        # cnn hyperparameters
        conv_f1 = hp.Int("filter1", min_value=32, max_value=250, step=32, default=64)
        kernel_1 = hp.Int("kernel1", min_value=1, max_value=8, step=2, default=6)
        conv_f2 = hp.Int("filter2", min_value=32, max_value=250, step=32, default=64)
        kernel_2 = hp.Int("kernel2", min_value=1, max_value=8, step=2, default=6)
        conv_f3 = hp.Int("filter3", min_value=32, max_value=250, step=32, default=64)
        kernel_3 = hp.Int("kernel3", min_value=1, max_value=8, step=2, default=6)
        units_1 = hp.Int("units_1", min_value=32, max_value=150, step=32, default=64)
        drop_hp = hp.Float("dropout_hp", min_value=0.0, max_value=0.6, default=0.0, step=0.1)
        pool_1 = hp.Int("pool1", min_value=1, max_value=3, step=1, default=1)
        pool_2 = hp.Int("pool2", min_value=1, max_value=3, step=1, default=1)
        pool_3 = hp.Int("pool3", min_value=1, max_value=3, step=1, default=1)
        # cnn model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        conv1 = Conv1D(conv_f1, kernel_1, activation="relu")(input_cnn)
        max1 = MaxPooling1D(pool_1)(conv1)
        conv2 = Conv1D(conv_f2, kernel_2, activation="relu")(max1)
        cnn_network = MaxPooling1D(pool_2)(conv2)
        # cnn_network = Conv1D(conv_f3, kernel_3, activation="relu")(max2)
        # cnn_network = MaxPooling1D(pool_3)(cnn_network)
        if hp.Choice('rnn_cnn_pooling', ['global', 'flatten', "attention"]) == 'global':
            flat = GlobalMaxPooling1D()(cnn_network)
        elif hp.Choice('rnn_cnn_pooling', ['global', 'flatten', "attention"]) == 'attention':
            flat = MyAttention()(cnn_network)
        else:
            flat = Flatten()(cnn_network)
        # rnn hyperparameters
        emb = hp.Int("emb_output", min_value=10, max_value=90, step=10)
        lstm_1 = hp.Int("LSTM_1", min_value=32, max_value=250, step=32, default=64)
        lstm_2 = hp.Int("LSTM_2", min_value=32, max_value=250, step=32, default=64)
        drop_lstm1 = hp.Float("dropout_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        drop_lstm2 = hp.Float("dropout_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop1 = hp.Float("recurrent_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop2 = hp.Float("recurrent_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)

        # rnn network
        input_rnn = Input(name="rnn_input", shape=self.maxlen)
        rnn1 = Embedding(21, emb, mask_zero=True)(input_rnn)
        # rnn1 = Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1, recurrent_dropout=recurrent_drop1
        #                           ))(embed)
        if hp.Choice("attention_rnn_cnn", ["attention", "no_attention"]) == "no_attention":
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2))(rnn1)
        else:
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2,
                                      return_sequences=True))(rnn1)
            rnn2 = MyAttention()(rnn2)
        # concatenate and generate the model
        concat = concatenate([flat, rnn2], axis=-1)
        # dense1 = Dense(units_1, activation="relu")(concat)
        drop = Dropout(drop_hp)(concat)
        output = Dense(2, activation="softmax")(drop)
        model = Model([input_cnn, input_rnn], output)

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.001, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if hp.Choice('optimizer', ['adam', 'adamax']) == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def rnn_inception(self, hp):
        """

        Parameters
        ----------
        hp: HyperParameters
            The HyperParameters class in keras tuner
        """
        # hyperparameters
        conv_f1 = hp.Int("filter1", min_value=32, max_value=250, step=32, default=64)
        kernel_1 = hp.Int("kernel1", min_value=1, max_value=8, step=2, default=6)
        conv_f2 = hp.Int("filter2", min_value=32, max_value=250, step=32, default=64)
        kernel_2 = hp.Int("kernel2", min_value=1, max_value=8, step=2, default=6)
        conv_f3 = hp.Int("filter3", min_value=32, max_value=250, step=32, default=64)
        kernel_3 = hp.Int("kernel3", min_value=1, max_value=8, step=2, default=6)
        conv_f4 = hp.Int("filter4", min_value=32, max_value=250, step=32, default=64)
        kernel_4 = hp.Int("kernel4", min_value=1, max_value=8, step=2, default=6)
        conv_f5 = hp.Int("filter5", min_value=32, max_value=250, step=32, default=64)
        kernel_5 = hp.Int("kernel5", min_value=1, max_value=8, step=2, default=6)
        conv_f6 = hp.Int("filter6", min_value=32, max_value=250, step=32, default=64)
        kernel_6 = hp.Int("kernel6", min_value=1, max_value=8, step=2, default=6)
        units_1 = hp.Int("units_1", min_value=32, max_value=150, step=32, default=64)
        drop_hp = hp.Float("dropout_hp", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        pool_1 = hp.Int("pool1", min_value=1, max_value=3, step=1, default=1)
        pool_2 = hp.Int("pool2", min_value=1, max_value=3, step=1, default=1)
        # model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        branch_a_cnn = Conv1D(conv_f1, kernel_1, activation="relu", strides=2, padding="same")(input_cnn)
        # branch_b_cnn = Conv1D(conv_f2, kernel_2, activation="relu", padding="same")(input_cnn)
        # branch_b_cnn = Conv1D(conv_f3, kernel_3, activation="relu", strides=2, padding="same")(branch_b_cnn)
        average_pool_c = AveragePooling1D(pool_1, strides=2, padding="same")(input_cnn)
        branch_c_cnn = Conv1D(conv_f4, kernel_4, activation="relu", padding="same")(average_pool_c)
        # branch_d_cnn = Conv1D(conv_f5, kernel_5, activation="relu", padding="same")(input_cnn)
        # branch_d_cnn = Conv1D(conv_f6, kernel_6, activation="relu", strides=2, padding="same")(branch_d_cnn)
        inception = concatenate([branch_a_cnn,  branch_c_cnn, ])
        inception = MaxPooling1D(pool_2)(inception)
        if hp.Choice('rnn_inception_pooling', ['global', 'flatten', "attention"]) == 'global':
            flat = GlobalMaxPooling1D()(inception)
        elif hp.Choice('rnn_inception_pooling', ['global', 'flatten', "attention"]) == 'attention':
            flat = MyAttention()(inception)
        else:
            flat = Flatten()(inception)

        # rnn hyperparameters
        emb = hp.Int("emb_output", min_value=10, max_value=90, step=10)
        lstm_1 = hp.Int("LSTM_1", min_value=32, max_value=250, step=32, default=64)
        lstm_2 = hp.Int("LSTM_2", min_value=32, max_value=250, step=32, default=64)
        drop_lstm1 = hp.Float("dropout_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        drop_lstm2 = hp.Float("dropout_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop1 = hp.Float("recurrent_lstm1", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        recurrent_drop2 = hp.Float("recurrent_lstm2", min_value=0.0, max_value=0.5, default=0.0, step=0.1)
        # rnn network
        input_rnn = Input(name="rnn_input", shape=self.maxlen)
        rnn1 = Embedding(21, emb, mask_zero=True)(input_rnn)
        # rnn1 = Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1, recurrent_dropout=recurrent_drop1
        #                           ))(embed)
        if hp.Choice("attention_rnn_inception", ["attention", "no_attention"]) == "no_attention":
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2))(rnn1)
        else:
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_drop2,
                                      return_sequences=True))(rnn1)
            rnn2 = MyAttention()(rnn2)
        # concatenate and generate the model
        concat = concatenate([flat, rnn2], axis=-1)
        # dense1 = Dense(units_1, activation="relu")(concat)
        drop = Dropout(drop_hp)(concat)
        output = Dense(2, activation="softmax")(drop)
        model = Model([input_cnn, input_rnn], output)

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.001, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if hp.Choice('optimizer', ['adam', 'adamax']) == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def search(self, mode):
        """
        Search for the best hyper parameters
        """
        call_list = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=10)]
        if mode == "rnn_cnn":
            tuner = Hyperband(self.rnn_cnn, objective="val_categorical_crossentropy", max_epochs=self.epoch-60,
                              seed=10, overwrite=True)
            tuner.search({"rnn_input": self.x_train_hot, "cnn_input": self.x_train_vec}, self.y_train, shuffle=True,
                         validation_split=0.17, epochs=self.epoch, callbacks=call_list, batch_size=40)
        elif mode == "rnn_inception":
            tuner = Hyperband(self.rnn_inception, objective="val_categorical_crossentropy", max_epochs=self.epoch-60,
                              seed=2, overwrite=True)
            tuner.search({"rnn_input": self.x_train_hot, "cnn_input": self.x_train_vec}, self.y_train, shuffle=True,
                         validation_split=0.17, epochs=self.epoch, callbacks=call_list, batch_size=40)
        K.clear_session()
        return tuner

    def analyse_search(self, x_train, x_test, mode, log_folder="tensorflow_logs", fold=1, model=None):
        """
        analyse the hyperparameters search results

        Parameters
        ___________
        x_train: dict{array1, array2}
            The dataset for the training
        x_test: dict{array1, array2}
            The dataset for evaluation
        mode: str
            The type of network topology to train
        log_folder: str, optional
            The name of the log folder
        model: Model
            A keras model object
        """
        if not os.path.exists(f"{self.csv}/{mode}"):
            os.makedirs(f"{self.csv}/{mode}")
        if not os.path.exists(f"model/{mode}"):
            os.makedirs(f"model/{mode}")
        if model is None:
            tuner = self.search(mode)
            best_hps = tuner.get_best_hyperparameters()[0]
            # writing the best hyperparameters in file
            best = pd.Series(best_hps.values)
            best.to_csv(f"{self.csv}/{mode}/hps_{fold}.csv", header=False)
            model = tuner.hypermodel.build(best_hps)
            config = GetConfig(model, f"model/{mode}", f"hps_{fold}.csv")
            config.to_csv()
        if not os.path.exists(f"{log_folder}_{mode}"):
            os.makedirs(f"{log_folder}_{mode}")
        call_list = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=15)]
        history = model.fit({"rnn_input": self.x_train_hot, "cnn_input": self.x_train_vec}, self.y_train,
                            shuffle=True, epochs=self.epoch, callbacks=call_list, validation_split=0.17, batch_size=40)
        history = pd.DataFrame(history.history)
        history.index = [f"model_{fold}" for _ in range(len(history))]
        # concatenate both metrics
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        return history, model, train_pred, test_pred


class DeepNetwork:
    """
    A class that generates the different deep learning topologies
    """
    def __init__(self, shape,
                 protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                 epochs=80):
        """
        Initialize the DeepNetwork class

        parameters
        ___________
        x_train: list
            A list of the prot2vec training data
        x_test: list
            A list of the prot2vec testing data
        y_train: list
            A list of the training labels
        y_test: list
            A list of the testing labels
        x_train_hot: list
            A list of the one-hot encoded training data
        x_test_hot: list
            A list of the one-hot encoded testing data
        protein_dataset: str
            The path to the dataset
        csv_dir: str, optional
            forlder for the csv
        epochs: int
            The number of training epochs
        """
        sequences = pd.read_excel(protein_dataset, engine='openpyxl', usecols=["enzyme", "seq", "label"], index_col=0)
        length = [len(seq) for seq in sequences["seq"]]
        self.maxlen = int(sum(length) / len(length) + 100)
        self.shape = shape
        self.epoch = epochs

    def sequential_rnn(self, emb, lstm_1, lstm_2, drop_lstm1, drop_lstm2, recurrent_1, recurrent_2, units_1, drop,
                       units_2, attention=None, adam=None):
        """
        A sequential deep network of RNN layers
        """

        # model
        model = Sequential()
        model.add(Embedding(21, emb, mask_zero=True, input_length=self.maxlen))
        model.add(Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1,
                                     recurrent_dropout=recurrent_1)))
        if not attention:
            model.add(Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2)))
        else:
            model.add(Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2,
                                         return_sequences=True)))
            model.add(MyAttention())
        model.add(Dense(units_1, activation="relu"))
        model.add(Dropout(drop))
        model.add(Dense(units_2, activation="relu"))
        model.add((Dense(2, activation="softmax")))

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if not adam:
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def sequential_cnn(self, conv_f1, kernel_1, conv_f2, kernel_2, conv_f3, kernel_3, units_1, drop, pool_1, pool_2,
                       pool_3, pooling, optimizer):
        """
        A sequential deep network of CNN layers
        parameters
        __________
        shape: int, optional
            The input shape of the cnn layer
        """
        # model
        model = Sequential()
        model.add(Conv1D(conv_f1, kernel_1, activation="relu", input_shape=self.shape))
        model.add(MaxPooling1D(pool_1))
        model.add(Conv1D(conv_f2, kernel_2, activation="relu"))
        model.add(MaxPooling1D(pool_2))
        model.add(Conv1D(conv_f3, kernel_3, activation="relu"))
        model.add(MaxPooling1D(pool_3))
        if pooling == 'global':
            model.add(GlobalMaxPooling1D())
        elif pooling == 'attention':
            model.add(MyAttention())
        else:
            model.add(Flatten())
        model.add(Dense(units_1, activation="relu"))
        model.add(Dropout(drop))
        model.add(Dense(2, activation="softmax"))
        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if optimizer == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def inception_cnn(self, conv_f1, kernel_1, conv_f2, kernel_2, conv_f3, kernel_3, conv_f4, kernel_4, conv_f5,
                      kernel_5, conv_f6, kernel_6, units_1, drop_hp, pool_1, pool_2, pooling, optimizer):
        """
        An inception model of CNN

        parameters
        __________
        hp: HyperParameters
            Used for the keras Tuner to find the optimal hyperparameters
        """
        # model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        branch_a_cnn = Conv1D(conv_f1, kernel_1, activation="relu", strides=2, padding="same")(input_cnn)
        branch_b_cnn = Conv1D(conv_f2, kernel_2, activation="relu", padding="same")(input_cnn)
        branch_b_cnn = Conv1D(conv_f3, kernel_3, activation="relu", strides=2, padding="same")(branch_b_cnn)
        average_pool_c = AveragePooling1D(pool_1, strides=2, padding="same")(input_cnn)
        branch_c_cnn = Conv1D(conv_f4, kernel_4, activation="relu", padding="same")(average_pool_c)
        branch_d_cnn = Conv1D(conv_f5, kernel_5, activation="relu", padding="same")(input_cnn)
        branch_d_cnn = Conv1D(conv_f6, kernel_6, activation="relu", strides=2, padding="same")(branch_d_cnn)
        inception = concatenate([branch_a_cnn, branch_b_cnn, branch_c_cnn, branch_d_cnn])
        if pooling == 'global':
            inception = MaxPooling1D(pool_2)(inception)
            flat = GlobalMaxPooling1D()(inception)
        elif pooling == 'attention':
            flat = MyAttention()(inception)
        else:
            inception = MaxPooling1D(pool_2)(inception)
            flat = Flatten()(inception)
        dense1 = Dense(units_1, activation="relu")(flat)
        drop = Dropout(drop_hp)(dense1)
        output = Dense(2, activation="softmax")(drop)
        model = Model(input_cnn, output)
        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if optimizer == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model


class MultiInputNetwork:
    """
    Generate a multinput netowrk
    """
    def __init__(self, shape,
                 protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                 epochs=80):
        """
        Initialize the DeepNetwork class

        parameters
        ___________
        x_train: list
            A list of the prot2vec training data
        y_train: list
            A list of the training labels
        x_train_hot: list
            A list of the one-hot encoded training data
        protein_dataset: str
            The path to the dataset
        epochs: int, optional
            The number of training epochs
        """
        sequences = pd.read_excel(protein_dataset, engine='openpyxl', usecols=["enzyme", "seq", "label"], index_col=0)
        length = [len(seq) for seq in sequences["seq"]]
        self.maxlen = int(sum(length) / len(length) + 100)
        self.shape = shape
        self.epoch = epochs

    def rnn_cnn(self, conv_f1, kernel_1, conv_f2, kernel_2, conv_f3, kernel_3, units_1, drop, pool_1, pool_2,
                pool_3, pooling, emb, lstm_1, lstm_2, drop_lstm1, drop_lstm2, recurrent_1, recurrent_2,
                attention=None, optimizer=None):

        # cnn model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        conv1 = Conv1D(conv_f1, kernel_1, activation="relu")(input_cnn)
        max1 = MaxPooling1D(pool_1)(conv1)
        conv2 = Conv1D(conv_f2, kernel_2, activation="relu")(max1)
        max2 = MaxPooling1D(pool_2)(conv2)
        cnn_network = Conv1D(conv_f3, kernel_3, activation="relu")(max2)
        cnn_network = MaxPooling1D(pool_3)(cnn_network)
        if pooling == 'global':
            flat = GlobalMaxPooling1D()(cnn_network)
        elif pooling == 'attention':
            flat = MyAttention()(cnn_network)
        else:
            flat = Flatten()(cnn_network)

        # rnn network
        input_rnn = Input(name="rnn_input", shape=self.maxlen)
        embed = Embedding(21, emb, mask_zero=True)(input_rnn)
        rnn1 = Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1, recurrent_dropout=recurrent_1
                                  ))(embed)
        if not attention:
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2))(rnn1)
        else:
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2,
                                      return_sequences=True))(rnn1)
            rnn2 = MyAttention()(rnn2)
        # concatenate and generate the model
        concat = concatenate([flat, rnn2], axis=-1)
        dense1 = Dense(units_1, activation="relu")(concat)
        drop = Dropout(drop)(dense1)
        output = Dense(2, activation="softmax")(drop)
        model = Model([input_cnn, input_rnn], output)

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if optimizer == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model

    def rnn_inception(self, conv_f1, kernel_1, conv_f2, kernel_2, conv_f3, kernel_3, conv_f4, kernel_4, conv_f5,
                      kernel_5, conv_f6, kernel_6, units_1, drop_hp, pool_1, pool_2, pooling,
                      emb, lstm_1, lstm_2, drop_lstm1, drop_lstm2, recurrent_1, recurrent_2,
                      attention=None, optimizer=None):

        # model
        input_cnn = Input(shape=self.shape, name="cnn_input")
        branch_a_cnn = Conv1D(conv_f1, kernel_1, activation="relu", strides=2, padding="same")(input_cnn)
        branch_b_cnn = Conv1D(conv_f2, kernel_2, activation="relu", padding="same")(input_cnn)
        branch_b_cnn = Conv1D(conv_f3, kernel_3, activation="relu", strides=2, padding="same")(branch_b_cnn)
        average_pool_c = AveragePooling1D(pool_1, strides=2, padding="same")(input_cnn)
        branch_c_cnn = Conv1D(conv_f4, kernel_4, activation="relu", padding="same")(average_pool_c)
        branch_d_cnn = Conv1D(conv_f5, kernel_5, activation="relu", padding="same")(input_cnn)
        branch_d_cnn = Conv1D(conv_f6, kernel_6, activation="relu", strides=2, padding="same")(branch_d_cnn)
        inception = concatenate([branch_a_cnn, branch_b_cnn, branch_c_cnn, branch_d_cnn])
        inception = MaxPooling1D(pool_2)(inception)
        if pooling == 'global':
            flat = GlobalMaxPooling1D()(inception)
        elif pooling == 'attention':
            flat = MyAttention()(inception)
        else:
            flat = Flatten()(inception)
        # rnn network
        input_rnn = Input(name="rnn_input", shape=self.maxlen)
        embed = Embedding(21, emb, mask_zero=True)(input_rnn)
        rnn1 = Bidirectional(LSTM(lstm_1, return_sequences=True, dropout=drop_lstm1, recurrent_dropout=recurrent_1
                                  ))(embed)
        if attention == "no_attention":
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2))(rnn1)
        else:
            rnn2 = Bidirectional(LSTM(lstm_2, dropout=drop_lstm2, recurrent_dropout=recurrent_2,
                                      return_sequences=True))(rnn1)
            rnn2 = MyAttention()(rnn2)
        # concatenate and generate the model
        concat = concatenate([flat, rnn2], axis=-1)
        dense1 = Dense(units_1, activation="relu")(concat)
        drop = Dropout(drop_hp)(dense1)
        output = Dense(2, activation="softmax")(drop)
        model = Model([input_cnn, input_rnn], output)

        # compile
        learning = PolynomialDecay(initial_learning_rate=0.0008, end_learning_rate=0.0001, decay_steps=80, power=0.8)
        if optimizer == 'adamax':
            opti = Adamax(learning_rate=learning)
        else:
            opti = Adam(learning_rate=learning)
        model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=[CategoricalAccuracy(name="accuracy"),
                                                                                CategoricalCrossentropy()])
        return model


def plotting(history, plot_dir="metrics_plot", mode="cnn", fold=1):
    """
    A function that plots the losses and the accuracy of the 2 datasets

    Parameters
    ___________
    history: history object
        history object from the fitted keras models
    plot_dir: str, optional
        The folder to keep the images
    mode: str, optional
        The Network topology
    state: int, optional
        The random state used for dataset splitting
    """
    if not os.path.exists(f"{plot_dir}/{mode}"):
        os.makedirs(f"{plot_dir}/{mode}")

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss'][3:]
    val_loss = history['val_loss'][3:]
    entropy = history["categorical_crossentropy"][3:]
    val_entropy = history["val_categorical_crossentropy"][3:]
    epochs = range(1, len(acc) + 1)
    epochs_loss = range(4, len(acc) + 1)
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'{plot_dir}/{mode}/acc_{fold}.png', transparent=True, dpi=800)
    plt.figure()
    plt.plot(epochs_loss, loss, 'ro', label='Training loss')
    plt.plot(epochs_loss, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{plot_dir}/{mode}/loss_{fold}.png', transparent=True, dpi=800)
    plt.figure()
    plt.plot(epochs_loss, entropy, 'ro', label='Training cross entropy')
    plt.plot(epochs_loss, val_entropy, 'b', label='Validation cross entropy')
    plt.title('Training and validation categorical crossentropy')
    plt.xlabel('epochs')
    plt.ylabel('crossentropy')
    plt.legend()
    plt.savefig(f'{plot_dir}/{mode}/entropy_{fold}.png', transparent=True, dpi=800)
    plt.close("all")


def train(protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
          vec="/gpfs/home/bsc72/bsc72661/feature_extraction/data/protVec_100d_3grams.csv",
          log_folder="tensorflow_logs", mode="cnn", plot_dir="metrics_plot", csv_dir="metrics_csv", epochs=80):
    """
    Train the model

    Parameters
    ----------
    protein_dataset: str, optional
        Path to the sequence dataset
    vec: str, optional
        Path to the prot2vec encoding
    log_folder: str, optional
        The folder name for the logs from tensorboard
    mode: str, optional
        The network topology
    plot_dir: str, optional
        The name of the folder for the plots
    csv_dir: str, optional
        The folder for the csv
    epochs: int
        The number of epochs to train
    """
    encoding = Encode(protein_dataset, vec)
    prot_list = encoding.prot2vec()
    one_hot = encoding.tokenized()
    categorical_label = encoding.categorical_label
    sequences = encoding.sequences
    history_list = []
    test_metric_list = []
    train_metric_list = []
    tr_report_list = []
    te_report_list = []
    if not os.path.exists(f"{csv_dir}/{mode}"):
        os.makedirs(f"{csv_dir}/{mode}")
    # reserve a test set
    indices = [i for i in range(len(one_hot))]
    np.random.shuffle(indices)
    one_hot = one_hot[indices]
    categorical_label = categorical_label[indices]
    sequences = sequences.iloc[indices]
    count = 1
    kf = StratifiedKFold(n_splits=6)
    for train_index, test_index in kf.split(one_hot, sequences["label"]):
        # shuffling more the indices since kfolds does not do that very well
        x_train_vec, x_test_vec = prot_list[train_index], prot_list[test_index]
        y_train, y_test = categorical_label[train_index], categorical_label[test_index]
        x_train_hot, x_test_hot = one_hot[train_index], one_hot[test_index]

        # create a validation set
        x_train2_hot, x_val_hot, y_train2_hot, y_val_hot = encoding.processing(x_train_hot, y_train, split=0.17)
        x_train_2_vec, x_val_vec, y_train_2_vec, y_val_vec = encoding.processing(x_train_vec, y_train, split=0.17)

        if mode == "rnn_cnn" or mode == "rnn_inception":
            train_data = {"rnn_input": x_train_hot, "cnn_input": x_train_vec}
            test_data = {"rnn_input": x_test_hot, "cnn_input": x_test_vec}
            network = MultiInputNetworkSearch(x_train_vec, y_train, x_train_hot, protein_dataset, csv_dir, epochs)
            history, model, train_pred, test_pred = network.analyse_search(train_data, test_data,
                                                                           mode=mode, log_folder=log_folder, fold=count)
        else:
            network = DeepNetworkSearch(x_train_2_vec, x_val_vec, y_train_2_vec, y_val_vec, x_train2_hot, x_val_hot,
                                        protein_dataset, csv_dir=csv_dir, epochs=epochs)
            if mode != "rnn":
                history, model, train_pred, test_pred = network.analyse_search(x_train_vec, x_test_vec,
                                                                               mode=mode, log_folder=log_folder,
                                                                               fold=count)
            else:
                history, model, train_pred, test_pred = network.analyse_search(x_train_hot, x_test_hot,
                                                                               mode=mode, log_folder=log_folder,
                                                                               fold=count)
        # plot the results
        plotting(history, plot_dir, mode, count)
        # classification metrics
        score = GenerateScore(y_train, train_pred, y_test, test_pred, count)
        every_score = score.scores()
        history_list.append(history)
        test_metric_list.append(every_score.test_metrics)
        train_metric_list.append(every_score.train_metrics)
        tr_report_list.append(every_score.tr_report)
        te_report_list.append(every_score.te_report)
        count += 1

    # generating the dataframes from the different folds
    test_metric_list = pd.concat(test_metric_list)
    train_metric_list = pd.concat(train_metric_list)
    tr_report_list = pd.concat(tr_report_list, axis=1)
    te_report_list = pd.concat(te_report_list, axis=1)
    history = pd.concat(history_list)
    # writing to csv_files
    history.to_csv(f"{csv_dir}/{mode}/history_metrics.csv", header=False)
    test_metric_list.to_csv(f"{csv_dir}/{mode}/test_metrics.csv", header=True)
    train_metric_list.to_csv(f"{csv_dir}/{mode}/train_metrics.csv", header=True)
    tr_report_list.to_csv(f"{csv_dir}/{mode}/tr_reports.csv", header=True)
    te_report_list.to_csv(f"{csv_dir}/{mode}/te_reports.csv", header=True)


def train_from_hps(protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
                   vec="/gpfs/home/bsc72/bsc72661/feature_extraction/data/protVec_100d_3grams.csv",
                   mode="cnn", plot_dir="metrics_plot", csv_dir="metrics_csv", epochs=80):
    """
    Train the model from the best hyperparameters

    Parameters
    ----------
    protein_dataset: str, optional
        Path to the sequence dataset
    vec: str, optional
        Path to the prot2vec encoding
    mode: str, optional
        The network topology
    plot_dir: str, optional
        The name of the folder for the plots
    csv_dir: str, optional
        The folder for the csv
    epochs: int
        The number of epochs to train
    """
    np.random.seed(10)
    encoding = Encode(protein_dataset, vec)
    prot_list = encoding.prot2vec()
    one_hot = encoding.tokenized()
    categorical_label = encoding.categorical_label
    sequences = encoding.sequences
    history_list = []
    test_metric_list = []
    train_metric_list = []
    tr_report_list = []
    te_report_list = []
    eval_list = []
    if not os.path.exists(f"{csv_dir}/{mode}"):
        os.makedirs(f"{csv_dir}/{mode}")
    # reserve a test set
    count = 1
    indices = [i for i in range(len(one_hot))]
    np.random.shuffle(indices)
    one_hot = one_hot[indices]
    categorical_label = categorical_label[indices]
    sequences = sequences.iloc[indices]
    kf = StratifiedKFold(n_splits=6)
    for train_index, test_index in kf.split(one_hot, sequences["label"]):
        x_train_vec, x_test_vec = prot_list[train_index], prot_list[test_index]
        y_train, y_test = categorical_label[train_index], categorical_label[test_index]
        x_train_hot, x_test_hot = one_hot[train_index], one_hot[test_index]
        # create a validation set
        x_train2_hot, x_val_hot, y_train2_hot, y_val_hot = encoding.processing(x_train_hot, y_train, split=0.10)
        x_train_2_vec, x_val_vec, y_train_2_vec, y_val_vec = encoding.processing(x_train_vec, y_train, split=0.10)
        # read the best hyperparameters
        hps = pd.read_csv(f"{csv_dir}/{mode}/hps_{count}.csv", header=None, names=["col"])
        hps = list(hps["col"].values)
        for i in range(len(hps)):
            if hps[i].isnumeric():
                hps[i] = int(hps[i])
            elif encoding.is_float(hps[i]) and not hps[i].isnumeric():
                hps[i] = float(hps[i])
        if "adam" in hps:
            del hps[hps.index("adam")+1:]
        elif "adamax" in hps:
            del hps[hps.index("adamax")+1:]
        # gathering the data for multi-input networks
        train_data = {"rnn_input": x_train_hot, "cnn_input": x_train_vec}
        test_data = {"rnn_input": x_test_hot, "cnn_input": x_test_vec}
        call_list = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=20)]

        # instatiate the netwroks
        network = MultiInputNetwork(x_train_vec.shape[1:], protein_dataset, epochs)
        deep = DeepNetwork(x_train_vec.shape[1:], protein_dataset, epochs)
        if mode == "rnn_cnn":
            model = network.rnn_cnn(*hps)
            history = model.fit(train_data, y_train, epochs=epochs, shuffle=True, validation_split=0.1,
                                callbacks=call_list)
            metrics = model.evaluate(test_data, y_test)
            train_pred = model.predict(train_data)
            test_pred = model.predict(test_data)
        elif mode == "rnn_inception":
            model = network.rnn_inception(*hps)
            history = model.fit(train_data, y_train, epochs=epochs, shuffle=True, validation_split=0.1,
                                callbacks=call_list)
            metrics = model.evaluate(test_data, y_test, return_dict=True)
            train_pred = model.predict(train_data)
            test_pred = model.predict(test_data)
        elif mode == "rnn":
            model = deep.sequential_rnn(*hps)
            history = model.fit(x_train2_hot, y_train2_hot, epochs=epochs, shuffle=True, callbacks=call_list,
                                validation_data=(x_val_hot, y_val_hot))
            metrics = model.evaluate(x_test_hot, y_test, return_dict=True)
            train_pred = model.predict(x_train_hot)
            test_pred = model.predict(x_test_hot)
        elif mode == "cnn":
            model = deep.sequential_cnn(*hps)
            history = model.fit(x_train_2_vec, y_train_2_vec, epochs=epochs, shuffle=True, callbacks=call_list,
                                validation_data=(x_val_vec, y_val_vec))
            metrics = model.evaluate(x_test_vec, y_test, return_dict=True)
            train_pred = model.predict(x_train_vec)
            test_pred = model.predict(x_test_vec)
        else:
            model = deep.inception_cnn(*hps)
            history = model.fit(x_train_2_vec, y_train_2_vec, epochs=epochs, shuffle=True, callbacks=call_list,
                                validation_data=(x_val_vec, y_val_vec))
            metrics = model.evaluate(x_test_vec, y_test, return_dict=True)
            train_pred = model.predict(x_train_vec)
            test_pred = model.predict(x_test_vec)

        # plot the results
        metrics = pd.Series(metrics)
        eval_list.append(metrics)
        history = pd.DataFrame(history.history)
        history.index = [f"model_{count}" for _ in range(len(history))]
        plotting(history, plot_dir, mode, count)
        # classification metrics
        score = GenerateScore(y_train, train_pred, y_test, test_pred, count)
        every_score = score.scores()
        history_list.append(history)
        test_metric_list.append(every_score.test_metrics)
        train_metric_list.append(every_score.train_metrics)
        tr_report_list.append(every_score.tr_report)
        te_report_list.append(every_score.te_report)
        count += 1

    # generating the dataframes from the different folds
    eval_list = pd.concat(eval_list, axis=1)
    eval_list.columns = [f"model_{i}" for i in range(len(eval_list.columns))]
    test_metric_list = pd.concat(test_metric_list)
    train_metric_list = pd.concat(train_metric_list)
    tr_report_list = pd.concat(tr_report_list, axis=1)
    te_report_list = pd.concat(te_report_list, axis=1)
    history = pd.concat(history_list)
    # writing to csv_files
    eval_list.to_csv(f"{csv_dir}/{mode}/evaluation.csv", header=False)
    history.to_csv(f"{csv_dir}/{mode}/history_metrics.csv", header=False)
    test_metric_list.to_csv(f"{csv_dir}/{mode}/test_metrics.csv", header=True)
    train_metric_list.to_csv(f"{csv_dir}/{mode}/train_metrics.csv", header=True)
    tr_report_list.to_csv(f"{csv_dir}/{mode}/tr_reports.csv", header=True)
    te_report_list.to_csv(f"{csv_dir}/{mode}/te_reports.csv", header=True)


def train_all(protein_dataset="/gpfs/home/bsc72/bsc72661/feature_extraction/data/sequences.xlsx",
              vec="/gpfs/home/bsc72/bsc72661/feature_extraction/data/protVec_100d_3grams.csv",
              log_folder="tensorflow_logs", plot_dir="metrics_plot", csv_dir="metrics_csv", epochs=80, restart=None,
              train_model=False, only=None):
    """
    Train the all network topologies

    Parameters
    ----------
    protein_dataset: str, optional
        Path to the sequence dataset
    vec: str, optional
        Path to the prot2vec encoding
    log_folder: str, optional
        The folder name for the logs from tensorboard
    plot_dir: str, optional
        The name of the folder for the plots
    csv_dir: str, optional
        The folder for the csv
    epochs: int
        The number of epochs to train
    restart: str, optional
        Indicate the topology to restart the training
    train_model: bool, optional
        True if you want to retrain the models using the search hyperparameters
    only: list[str], optional
        Twhich topologis to train
    """
    modes = ["cnn", "rnn", "inception", "rnn_cnn", "rnn_inception"]
    if restart:
        modes = modes[modes.index(restart):]
    if only:
        modes = only
    for mode in modes:
        if not train_model:
            train(protein_dataset, vec, log_folder, mode, plot_dir, csv_dir, epochs)
        else:
            train_from_hps(protein_dataset, vec, mode, plot_dir, csv_dir, epochs)


def main():
    protein_dataset, prot2vec, log_folder, plot_dir, csv_dir, epochs, restart, train_model, only = arg_parse()
    train_all(protein_dataset, prot2vec, log_folder, plot_dir, csv_dir, epochs, restart, train_model, only)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
