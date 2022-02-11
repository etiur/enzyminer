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
    parser.add_argument("-fo", "--filtered_out", required=False, help="The directory for the filtered features",
                        default="filtered_features")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path")
    parser.add_argument("-rs", "--res_dir", required=False,
                        default="results", help="The name for the folder where to store the prediction results")
    parser.add_argument("-st", "--strict", required=False, action="store_false",
                        help="To use a strict voting scheme or not, default to true")
    parser.add_argument("-v", "--value", required=False, default=0.8, type=float, choices=(0.8, 0.7, 0.5),
                        help="The voting threshold to be considered positive")
    args = parser.parse_args()

    return [args.filtered_out, args.number_similar_samples, args.fasta_file, args.res_dir, args.strict, args.value]


class EnsembleVoting:
    """
    A class to perform ensemble voting
    """

    def __init__(self, feature_out="filtered_features"):
        """
        Initialize the class EnsembleVoting

        Parameters
        ____________
        feature_out: str
            The path to the directory where the new extracted feature files are
        """
        self.filtered_out = feature_out
        self.learning = "/gpfs/projects/bsc72/ruite/enzyminer/data/esterase_binary.xlsx"
        self.models = "/gpfs/projects/bsc72/ruite/enzyminer/models"

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
        ridge_40 = joblib.load(f"{self.models}/ridge_40.pkl")
        ridge_80 = joblib.load(f"{self.models}/ridge_80.pkl")
        knn_20 = joblib.load(f"{self.models}/knn_20.pkl")
        knn_90 = joblib.load(f"{self.models}/knn_90.pkl")

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
        pred_knn_90 = knn_90.predict(transformed_x_knn)

        knn[20] = pred_knn_20
        knn[90] = pred_knn_90

        return svc, ridge, knn, transformed_x, old_svc, transformed_x_knn, old_knn

    def vote(self, val=0.5, *args):
        """
        Hard voting for the ensembles

        Parameters
        ___________
        args: list[arrays]
            A list of prediction arrays
        """
        vote_ = []
        index = []
        mean = np.mean(args, axis=0)
        for s, x in enumerate(mean):
            if x == 1 or x == 0:
                vote_.append(int(x))
            elif x > val:
                vote_.append(1)
                index.append(s)
            elif x <= val:
                vote_.append(0)
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
        path_to_esterase = "/gpfs/projects/bsc72/ruite/enzyminer/data/esterase_binary.xlsx"
        x_svc = pd.read_excel(f"{path_to_esterase}", index_col=0, sheet_name=f"ch2_20", engine='openpyxl')
        self.training_names = x_svc.index
        self.ad_indices = []

    def fit(self, x_train):
        """
        A function to calculate the training sample threshold for the applicability domain

        Parameters
        ___________
        x_train: pandas Dataframe object
        """
        self.x_train = x_train
        # for each of the training sample calculate the distance to the other samples
        distances = np.array([distance.cdist(np.array(x).reshape(1, -1), self.x_train) for x in self.x_train])
        distances_sorted = [np.sort(d[0]) for d in distances]
        d_no_ii = [d[1:] for d in distances_sorted]  # not including the distance with itself, which is 0
        k = int(round(pow(len(self.x_train), 1 / 3)))
        d_means = [np.mean(d[:k]) for d in d_no_ii]  # medium values, np.mean(d[:k][0])
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
            idxs = [self.training_names[j] for j, d in enumerate(i[0]) if d <= self.thresholds[j]]
            self.n_insiders.append(len(idxs))  # for each test sample see how many training samples is within the AD
            idxs = "_".join(idxs)
            self.ad_indices.append(idxs)

        return self.n_insiders, self.ad_indices

    def _sort_models_vote(self, svc, knn, ridge, idx, filtered_names, min_num=1):
        """
        Parameters
        ----------
        svc: dict
            Predictions from SVCs
        knn: dict
            Predictions from KNNs
        ridge: dict
            Prediction from ridge classifiers
        idx: list[int]
            Indices where the votes did not agree
        filtered_names: list[str]
            Names of the test samples after the filtering
        min_num: int
            The minimum number to be considered of the same applicability domain

        Returns
        --------
        object: pd.Dataframe
            The predictions of each of the models kept in the dataframe
        """
        results = {"s20": None, "s80": None, "r20": None, "r40": None, "r80": None,
                   "k20": None, "k90": None}
        svc_20 = [d[0] for x, d in enumerate(zip(svc[20], self.n_insiders)) if d[1] >= min_num and x not in idx]
        svc_80 = [d[0] for x, d in enumerate(zip(svc[80], self.n_insiders)) if d[1] >= min_num and x not in idx]
        ridge_20 = [d[0] for x, d in enumerate(zip(ridge[20], self.n_insiders)) if d[1] >= min_num and x not in idx]
        ridge_40 = [d[0] for x, d in enumerate(zip(ridge[40], self.n_insiders)) if d[1] >= min_num and x not in idx]
        ridge_80 = [d[0] for x, d in enumerate(zip(ridge[80], self.n_insiders)) if d[1] >= min_num and x not in idx]
        knn_20 = [d[0] for x, d in enumerate(zip(knn[20], self.n_insiders)) if d[1] >= min_num and x not in idx]
        knn_90 = [d[0] for x, d in enumerate(zip(knn[90], self.n_insiders)) if d[1] >= min_num and x not in idx]
        results["s20"] = svc_20
        results["s80"] = svc_80
        results["r20"] = ridge_20
        results["r40"] = ridge_40
        results["r80"] = ridge_80
        results["k20"] = knn_20
        results["k90"] = knn_90
        return pd.DataFrame(results, index=filtered_names)

    def filter(self, prediction, index, svc, knn, ridge, min_num=1, path_name="filtered_predictions.parquet",
               strict=True):
        """
        Filter those predictions that have less than min_num training samples that are within the AD

        Parameters
        ___________
        prediction: array
            An array of the predictions
        index: array
            The index of those predictions that were not unanimous between different models
        path_name: str, optional
            The path for the csv file
        min_num: int, optional
            The minimun number of training samples within the AD of the test samples
        """
        if strict:
            idx = index[:]
        else:
            idx = []
        # filter the predictions and names  based on the specified number of similar training samples
        filtered_indices = [d[0] for x, d in enumerate(zip(self.ad_indices, self.n_insiders)) if d[1] >= min_num and x not in idx]
        filtered_pred = [d[0] for x, d in enumerate(zip(prediction, self.n_insiders)) if d[1] >= min_num and x not in idx]
        filtered_names = [d[0] for y, d in enumerate(zip(self.test_names, self.n_insiders)) if d[1] >= min_num and y not in idx]
        filtered_n_insiders = [d for s, d in enumerate(self.n_insiders) if d >= min_num and s not in idx]
        name_training_sample = pd.Series(filtered_indices, index=filtered_names)
        pred = pd.Series(filtered_pred, index=filtered_names)
        n_applicability = pd.Series(filtered_n_insiders, index=filtered_names)
        models = self._sort_models_vote(svc, knn, ridge, idx, filtered_names, min_num)
        self.pred = pd.concat([pred, n_applicability, name_training_sample, models], axis=1)
        self.pred.columns = ["prediction", "AD_number", "AD_names"] + list(models.columns)
        self.pred.to_parquet(path_name)
        return self.pred

    def separate_negative_positive(self, fasta_file, pred=None):
        """
        Parameters
        ______________
        fasta_file: str
            The input fasta file
        pred: list, optional
            The predictions

        Return
        ________
        positive: list[Bio.SeqIO]
        negative: list[Bio.SeqIO]
        """
        if pred is not None:
            self.pred = pred
        # separating the records according to if the prediction is positive or negative
        if dirname(fasta_file) != "":
            base = dirname(fasta_file)
        else:
            base = "."
        with open(f"{base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            p = 0
            positive = []
            negative = []
            for ind, seq in enumerate(record):
                try:
                    if int(self.pred.index[p].split("_")[1]) == ind:
                        col = self.pred[self.pred.columns[3:]].iloc[p]
                        mean = round(sum(col) / len(col), 2)
                        col = [f"{col.index[i]}-{d}" for i, d in enumerate(col)]
                        seq.id = f"{seq.id}-{'+'.join(col)}-#%$prob:{mean}#%$AD:{self.pred['AD_number'][p]}"
                        if self.pred["prediction"][p] == 1:
                            positive.append(seq)
                        else:
                            negative.append(seq)
                        p += 1
                except IndexError:
                    break

        return positive, negative

    def find_max_ad(self, pred1, pred2):
        """
        find the maximum applicability domain of the 2 preds
        """
        assert len(pred1) == len(pred2), "Both predictions has different length"
        ad = []
        pred = pred1.copy()
        for idx in pred1.index:
            if pred1["AD_number"].loc[idx] <= pred2["AD_number"].loc[idx]:
                ad.append(f'{pred1["AD_number"].loc[idx]}-svc')
            else:
                ad.append(f'{pred2["AD_number"].loc[idx]}-knn')
        pred["AD_number"] = ad
        return pred

    def extract(self, fasta_file, pred1=None, pred2=None, positive_fasta="positive.fasta",
                negative_fasta="negative.fasta", res_dir="results"):
        """
        A function to extract those test fasta sequences that passed the filter

        Parameters
        ___________
        fasta_file: str
            The path to the test fasta sequences
        pred: pandas Dataframe, optional
            Predictions
        positive_fasta: str, optional
            The new filtered fasta file with positive predictions
        negative_fasta: str, optional
            The new filtered fasta file with negative sequences
        res_dir: str, optional
            The folder where to keep the prediction results
        """
        if pred2 is not None:
            pred1 = self.find_max_ad(pred1, pred2)
        positive, negative = self.separate_negative_positive(fasta_file, pred1)
        # writing the positive and negative fasta sequences to different files
        with open(f"{res_dir}/{positive_fasta}", "w") as pos:
            positive = sorted(positive, reverse=True, key=lambda x: (float(x.id.split("#%$")[1].split(":")[1]),
                                                                     int(x.id.split("#%$")[2].split(":")[1].split("-")[0])))
            fasta_pos = FastaIO.FastaWriter(pos, wrap=None)
            fasta_pos.write_file(positive)
        with open(f"{res_dir}/{negative_fasta}", "w") as neg:
            negative = sorted(negative, reverse=True, key=lambda x: int(x.id.split("#%$")[2].split(":")[1].split("-")[0]))
            fasta_neg = FastaIO.FastaWriter(neg, wrap=None)
            fasta_neg.write_file(negative)


def vote_and_filter(feature_out, fasta_file, min_num=1, res_dir="results", strict=True, val=0.8):
    """
    A class that predicts and then filter the results based on the applicability domain of the model

    Parameters
    ___________
    feature_out: str,
        Path to the directory of the newly generated feature files
    fasta_file: str
        The path to the test fasta sequences
    csv_name: str, optional
        The name for the ensemble prediction file
    min_num: int, optional
        The number of similar training samples
    res_dir: str, optional
        The folder where to keep the prediction results
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ensemble = EnsembleVoting(feature_out)
    # predictions
    svc, ridge, knn, new_svc, X_svc, new_knn, X_knn = ensemble.predicting()
    all_voting, all_index = ensemble.vote(val, *svc.values(), *ridge.values(), *knn.values())
    # applicability domain for scv
    domain_svc = ApplicabilityDomain()
    domain_svc.fit(X_svc)
    domain_svc.predict(new_svc)
    # return the prediction after the applicability domain filter of SVC
    pred_svc = domain_svc.filter(all_voting, all_index, svc, knn, ridge, min_num, f"{res_dir}/svc_domain.parquet",
                                 strict)
    domain_svc.extract(fasta_file, pred_svc, positive_fasta=f"positive_svc.fasta",
                       negative_fasta=f"negative_svc.fasta", res_dir=res_dir)
    # applicability domain for KNN
    domain_knn = ApplicabilityDomain()
    domain_knn.fit(X_knn)
    domain_knn.predict(new_knn)
    # return the prediction after the applicability domain filter of KNN
    pred_knn = domain_knn.filter(all_voting, all_index,  svc, knn, ridge, min_num, f"{res_dir}/knn_domain.parquet",
                                 strict)
    domain_knn.extract(fasta_file, pred_knn, positive_fasta=f"positive_knn.fasta",
                       negative_fasta=f"negative_knn.fasta", res_dir=res_dir)
    # Then filter again to see which sequences are within the AD of both algorithms since it is an ensemble classifier
    name_set = set(pred_svc.index).intersection(set(pred_knn.index))
    name_set = sorted(name_set, key=lambda x: int(x.split("_")[1]))
    knn_set = pred_knn.loc[name_set]
    common_domain = pred_svc.loc[name_set]
    common_domain.to_parquet(f"{res_dir}/common_domain.parquet")
    # the positive sequences extracted will have the AD of the SVC
    domain_knn.extract(fasta_file, common_domain, knn_set, res_dir=res_dir)


def main():
    feature_out, min_num, fasta_file, res_dir, strict, value = arg_parse()
    vote_and_filter(feature_out, fasta_file, min_num, res_dir, strict, value)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
