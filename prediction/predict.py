import joblib
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial import distance
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import os
from os.path import dirname


def arg_parse():
    # Argument parsers
    parser = argparse.ArgumentParser(description="Make predictions")
    # main required arguments
    parser.add_argument("-fo", "--feature_out", required=False,
                        help="Include the directory path where the features are stored",
                        default="/gpfs/projects/bsc72/ruite/feature_extraction/power9/filtered_features")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    parser.add_argument("-c", "--csv_name", required=False,
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/results/common_doamin.csv",
                        help="The name of the csv file for the ensemble prediction")
    parser.add_argument("-if", "--input_fasta", required=False,
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/data/large.fasta",
                        help="The name of the input fasta file")
    parser.add_argument("-ps", "--positive_sequences", required=False,
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/results/positive.fasta",
                        help="The name for the fasta file with the positive sequences")
    parser.add_argument("-ns", "--negative_sequences", required=False,
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/results/negative.fasta",
                        help="The name for the fasta file with negative sequences")
    args = parser.parse_args()

    return [args.feature_out, args.number_similar_samples, args.csv_name, args.input_fasta, args.positive_sequences,
            args.negative_sequences]


class EnsembleVoting:
    """
    A class to perform ensemble voting
    """

    def __init__(self, feature_out="/gpfs/projects/bsc72/ruite/feature_extraction/power9/filtered_features"):
        """
        Initialize the class EnsembleVoting

        Parameters
        ____________
        feature_out: str
            The path to the directory where the new extracted feature files are
        """
        self.filtered_out = feature_out
        self.learning = "/gpfs/home/bsc72/bsc72661//feature_extraction/data/esterase_binary.xlsx"
        self.models = "/gpfs/home/bsc72/bsc72661/feature_extraction/models"

    def scale_transform(self, file_path, feature_set):
        """
        A class to scale the new extracted features according to the features used for training the models

        Parameters
        ___________
        file_path: str
            The path to the feature files
        feature_set: str
            The name of the feature set of the features used for training
        """
        # predict ch2_20 features
        scaling = MinMaxScaler()
        X_svc = pd.read_excel(f"{self.learning}", index_col=0, sheet_name=f"{feature_set}", engine='openpyxl')
        new_svc = pd.read_csv(f"{file_path}", index_col=0)

        if X_svc.isnull().values.any():
            X_svc.dropna(axis=1, inplace=True)
            X_svc.drop(["go"], axis=1, inplace=True)

        old_svc = scaling.fit_transform(X_svc)
        transformed_x = scaling.transform(new_svc)

        return transformed_x, old_svc

    def predicting(self):
        """
        Make predictions on new samples
        """
        svc = {}
        ridge = {}
        knn = {}

        # load the saved models
        svc_20 = joblib.load(f"{self.models}/svc_20.pkl")
        svc_80 = joblib.load(f"{self.models}/svc_80.pkl")
        ridge_20 = joblib.load(f"{self.models}/ridge_20.pkl")
        ridge_40 = joblib.load(f"{self.models}/ridge_20.pkl")
        ridge_80 = joblib.load(f"{self.models}/ridge_80.pkl")
        knn_20 = joblib.load(f"{self.models}/knn_20.pkl")
        knn_70 = joblib.load(f"{self.models}/knn_70.pkl")

        # predict ch2_20 features
        transformed_x, old_svc = self.scale_transform(f"{self.filtered_out}/svc_features.csv", "ch2_20")

        pred_svc_20 = svc_20.predict(transformed_x)
        pred_svc_80 = svc_80.predict(transformed_x)
        pred_ridge_20 = ridge_20.predict(transformed_x)
        pred_ridge_40 = ridge_40.predict(transformed_x)
        pred_ridge_80 = ridge_80.predict(transformed_x)

        svc[20] = pred_svc_20
        svc[80] = pred_svc_80
        ridge[20] = pred_ridge_20
        ridge[40] = pred_ridge_40
        ridge[80] = pred_ridge_80

        # predict random_30 features
        transformed_x_knn, old_knn = self.scale_transform(f"{self.filtered_out}/knn_features.csv", "random_30")

        pred_knn_20 = knn_20.predict(transformed_x_knn)
        pred_knn_80 = knn_70.predict(transformed_x_knn)

        knn[20] = pred_knn_20
        knn[80] = pred_knn_80

        return svc, ridge, knn, transformed_x, old_svc, transformed_x_knn, old_knn

    def vote(self, *args):
        """
        Hard voting for the ensembles

        Parameters
        ___________
        args: list[arrays]
            A list of prediction arrays
        """
        vote_ = []
        index = []
        if len(args) == 2:
            mean = np.mean(args, axis=0)
            for s, x in enumerate(mean):
                if x == 1 or x == 0:
                    vote_.append(int(x))
                else:
                    vote_.append(args[-1][s])
                    index.append(s)  # keep the index of non unanimous predictions

        elif len(args) % 2 == 1:
            mean = np.mean(args, axis=0)
            for s, x in enumerate(mean):
                if x == 1 or x == 0:
                    vote_.append(int(x))
                elif x > 0.5:
                    vote_.append(1)
                    index.append(s)
                else:
                    vote_.append(0)
                    index.append(s)

        else:
            mean = np.mean(args, axis=0)
            for s, x in enumerate(mean):
                if x == 1 or x == 0:
                    vote_.append(int(x))
                elif x > 0.5:
                    vote_.append(1)
                    index.append(s)
                elif x < 0.5:
                    vote_.append(0)
                    index.append(s)
                else:
                    vote_.append(args[-1][s])
                    index.append(s)

        return vote_, index


class ApplicabilityDomain():
    """
    A class that looks for the applicability domain
    """

    def __init__(self):
        """
        Initialize the class
        """
        self.x_train = None
        self.x_test = None
        self.thresholds = None
        self.test_names = None
        self.pred = []
        self.dataframe = None
        self.n_insiders = []

    def fit(self, x_train):
        """
        A function to calculate the training sample threshold for the applicability domain

        Parameters
        ___________
        x_train: pandas Dataframe object
        """
        self.x_train = x_train
        distances = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in self.x_train])
        distances_sorted = [np.sort(d[0]) for d in distances]
        d_no_ii = [d[1:] for d in distances_sorted]  # not including the distance with itself
        k = int(round(pow(len(self.x_train), 1 / 3)))

        d_means = [np.mean(d[:k][0]) for d in d_no_ii]  # medium values
        Q1 = np.quantile(d_means, .25)
        Q3 = np.quantile(d_means, .75)
        d_ref = Q3 + 1.5 * (Q3 - Q1)  # setting the reference value
        n_allowed = []
        all_allowed = []
        for i in d_no_ii:
            d_allowed = [d for d in i if d <= d_ref]  # keeping the distances that are smaller than the ref value
            all_allowed.append(d_allowed)
            n_allowed.append(len(d_allowed))  # calculating the density (number of distances kept) per sample

        # selecting minimum value not 0:
        min_val = [np.sort(n_allowed)[i] for i in range(len(n_allowed)) if np.sort(n_allowed)[i] != 0]

        # replacing 0's with the min val
        n_allowed = [n if n != 0 else min_val[0] for n in n_allowed]
        all_d = [sum(all_allowed[i]) for i, d in enumerate(d_no_ii)]
        self.thresholds = np.divide(all_d, n_allowed)  # threshold computation
        self.thresholds[np.isinf(self.thresholds)] = min(self.thresholds)  # setting to the minimum value where infinity
        return self.thresholds

    def predict(self, x_test):
        """
        A function to find those samples that are within the training samples' threshold

        Parameters
        ___________
        x_test: pandas Dataframe object
        """
        self.x_test = x_test
        self.test_names = ["sample_{}".format(i) for i in range(self.x_test.shape[0])]
        # calculating the distance of test with each of the training samples
        d_train_test = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in self.x_test])
        for i in d_train_test:  # for each sample
            # saving indexes of training with distance < threshold
            idxs = [j for j, d in enumerate(i[0]) if d <= self.thresholds[j]]
            self.n_insiders.append(len(idxs))  # for each test sample see how many training samples is within the AD

        return self.n_insiders

    def filter(self, prediction, index, min_num=1,
               path_name="/gpfs/home/bsc72/bsc72661/feature_extraction/results/filtered_predictions.csv"):
        """
        Filter those predictions that has less than min_num training samples that are within the AD
        prediction: array
            An array of the predictions
        index: array
            The index of those predictions that were not unanimous between different models
        path_name: str, optional
            The path for the csv file
        min_num: int, optional
            The minimun number of training samples within the AD of the test samples
        """
        # filter the predictions and names  based on the specified number of similar training samples
        filtered_pred = [d[0] for x, d in enumerate(zip(prediction, self.n_insiders)) if d[1] >= min_num and x not in index]
        filtered_names = [d[0] for y, d in enumerate(zip(self.test_names, self.n_insiders)) if d[1] >= min_num and y not in index]
        filtered_n_insiders = [d for s, d in enumerate(self.n_insiders) if d >= min_num and s not in index]
        pred = pd.Series(filtered_pred, index=filtered_names)
        n_applicability = pd.Series(filtered_n_insiders, index=filtered_names)
        self.pred = pd.concat([pred, n_applicability], axis=1)
        self.pred.columns = ["prediction", "AD_number"]
        self.pred.to_csv(path_name, header=True)
        return self.pred

    def extract(self, test_fasta, pred=None,
                positive_fasta="/gpfs/home/bsc72/bsc72661/feature_extraction/results/positive.fasta",
                negative_fasta="/gpfs/home/bsc72/bsc72661/feature_extraction/results/negative.fasta"):
        """
        A function to extract those test fasta sequences that passed the filter
        test_fasta: str
            The path to the test fasta sequences
        pred: pandas Dataframe, optional
            Predictions
        positive_fasta: str, optional
            The new filtered fasta file with positive predictions
        negative_fasta
            The new filtered fasta file with negative sequences
        """
        if pred is not None:
            self.pred = pred
        # separating the records according to if the prediction is positive or negative
        with open(test_fasta) as inp:
            record = SeqIO.parse(inp, "fasta")
            p = 0
            positive = []
            negative = []
            for ind, seq in enumerate(record):
                try:
                    if int(self.pred.index[p].split("_")[1]) == ind:
                        seq.id = f"{seq.id}-AD#%${self.pred['AD_number'][p]}"
                        if self.pred["prediction"][p] == 1:
                            positive.append(seq)
                        else:
                            negative.append(seq)
                        p += 1
                except IndexError:
                    break
        # writing the positive and negative fasta sequences to different files
        if not os.path.exists(dirname(positive_fasta)):
            os.makedirs(dirname(positive_fasta))
        if not os.path.exists(dirname(negative_fasta)):
            os.makedirs(dirname(negative_fasta))
        with open(positive_fasta, "w") as pos:
            positive = sorted(positive, reverse=True, key=lambda x: int(x.id.split("#%$")[1]))
            fasta_pos = FastaIO.FastaWriter(pos, wrap=None)
            fasta_pos.write_file(positive)
        with open(negative_fasta, "w") as neg:
            negative = sorted(negative, reverse=True, key=lambda x: int(x.id.split("#%$")[1]))
            fasta_neg = FastaIO.FastaWriter(neg, wrap=None)
            fasta_neg.write_file(negative)


def vote_and_filter(feature_out, input_fasta, min_num=1,
                    csv_name="/gpfs/home/bsc72/bsc72661/feature_extraction/results/common_doamin.csv",
                    positive="/gpfs/home/bsc72/bsc72661/feature_extraction/results/positive.fasta",
                    negative="/gpfs/home/bsc72/bsc72661/feature_extraction/results/negative.fasta"):
    """
    A class that predicts and then filter the results based on the applicability domain of the model

    Parameters
    ___________
    feature_out: str,
        Path to the directory of the newly generated feature files
    input_fasta: str
        The path to the test fasta sequences
    csv_name: str, optional
        The name for the ensemble prediction file
    min_num: int, optional
        The number of similar training samples
    positive_fasta: str, optional
        The new filtered fasta file with positive predictions
    negative_fasta
        The new filtered fasta file with negative sequences
    """
    ensemble = EnsembleVoting(feature_out)
    # predictions
    svc, ridge, knn, new_svc, X_svc, new_knn, X_knn = ensemble.predicting()
    svc_voting, svc_index = ensemble.vote(svc[20], svc[80])
    ridge_voting, ridge_index = ensemble.vote(ridge[20], ridge[40], ridge[80])
    knn_voting, knn_index = ensemble.vote(knn[20], knn[80])
    all_voting, all_index = ensemble.vote(svc[20], svc[80], ridge[20], ridge[40], ridge[80], knn[20], knn[80])

    # applicability domain
    domain_svc = ApplicabilityDomain()
    domain_svc.fit(X_svc)
    domain_svc.predict(new_svc)
    pred_svc = domain_svc.filter(all_voting, all_index, min_num,
                                 "/gpfs/home/bsc72/bsc72661/feature_extraction/results/svc_domain.csv")

    domain_knn = ApplicabilityDomain()
    domain_knn.fit(X_knn)
    domain_knn.predict(new_knn)
    pred_knn = domain_knn.filter(all_voting, all_index,  min_num,
                                 "/gpfs/home/bsc72/bsc72661/feature_extraction/results/knn_domain.csv")

    if not os.path.exists(dirname(csv_name)):
        os.makedirs(dirname(csv_name))
    name_set = set(pred_svc.index).intersection(pred_knn.index)
    name_set = sorted(name_set, key=lambda x: int(x.split("_")[1]))
    common_domain = pred_svc.loc[name_set]
    common_domain.to_csv(csv_name, header=True)
    domain_knn.extract(input_fasta, common_domain, positive, negative)


def main():
    feature_out, min_num, csv_name, input_fasta, positive, negative = arg_parse()
    vote_and_filter(feature_out, input_fasta, min_num, csv_name, positive, negative)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
