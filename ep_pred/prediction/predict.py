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
    parser.add_argument("-v", "--value", required=False, default=1, type=float, choices=(1, 0.8, 0.7, 0.5),
                        help="The voting threshold to be considered positive")
    args = parser.parse_args()

    return [args.filtered_out, args.number_similar_samples, args.fasta_file, args.res_dir, args.value]


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
        self.learning = "/gpfs/projects/bsc72/ruite/enzyminer/data/final_features.xlsx"
        self.models = "/gpfs/projects/bsc72/ruite/enzyminer/final_models"

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
        # extract the features
        transformed_x_knn, old_knn = self.scale_transform(f"{self.filtered_out}/knn_features.csv", "xgboost_30")
        transformed_x_svc, old_svc = self.scale_transform(f"{self.filtered_out}/svc_features.csv", "ch2_30")
        transformed_x_ridge, old_ridge = self.scale_transform(f"{self.filtered_out}/ridge_features.csv", "random_20")
        # load the saved models
        models = os.listdir(self.models)
        for mod in models:
            if "svc" in mod:
                mud = joblib.load(f"{self.models}/{mod}")
                pred = mud.predict(transformed_x_svc)
                svc[mod.strip(".pkl")] = pred
            elif "ridge" in mod:
                mud = joblib.load(f"{self.models}/{mod}")
                pred = mud.predict(transformed_x_ridge)
                ridge[mod.strip(".pkl")] = pred
            elif "knn" in mod:
                mud = joblib.load(f"{self.models}/{mod}")
                pred = mud.predict(transformed_x_knn)
                knn[mod.replace(".pkl", "")] = pred

        return svc, ridge, knn, transformed_x_svc, old_svc, transformed_x_knn, old_knn, transformed_x_ridge, old_ridge

    def vote(self, val=1, *args):
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
            elif x >= val:
                vote_.append(1)
                index.append(s)
            elif x < val:
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
        path_to_esterase = "/gpfs/projects/bsc72/ruite/enzyminer/data/final_features.xlsx"
        x_svc = pd.read_excel(f"{path_to_esterase}", index_col=0, sheet_name=f"ch2_30", engine='openpyxl')
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
        d_means = [np.mean(d[:k]) for d in d_no_ii]  # mean values, np.mean(d[:k][0])
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

    def _filter_models_vote(self, svc, knn, ridge, filtered_names, min_num=1):
        """
        Eliminate the models individual predictions of sequences that did not pass the applicability domain threshold

        Parameters
        ----------
        svc: dict
            Predictions from SVCs
        knn: dict
            Predictions from KNNs
        ridge: dict
            Prediction from ridge classifiers
        filtered_names: list[str]
            Names of the test samples after the filtering
        min_num: int
            The minimum number to be considered of the same applicability domain

        Returns
        --------
        object: pd.Dataframe
            The predictions of each of the models kept in the dataframe, the columns are the different model predictions
            and the rows the different sequences
        """
        results = {}
        for s, pred in svc.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv
        for s, pred in ridge.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv
        for s, pred in knn.items():
            sv = [d[0] for x, d in enumerate(zip(pred, self.n_insiders)) if d[1] >= min_num]
            results[s] = sv

        return pd.DataFrame(results, index=filtered_names)

    def filter(self, prediction, svc, knn, ridge, min_num=1, path_name="filtered_predictions.parquet"):
        """
        Filter those predictions that have less than min_num training samples

        Parameters
        ___________
        prediction: array
            An array of the predictions
        svc: dict[array]
            The prediction of the SVC models
        knn: dict[array]
            The predictions of the different Knn models
        ridge: dict[array]
            The predictions of the different ridge models
        index: array
            The index of those predictions that were not unanimous between different models
        path_name: str, optional
            The path for the csv file
        min_num: int, optional
            The minimun number of training samples within the AD of the test samples
        """
        # filter the predictions and names based on the if it passed the threshold of similar samples
        filtered_pred = [d[0] for x, d in enumerate(zip(prediction, self.n_insiders)) if d[1] >= min_num]
        filtered_names = [d[0] for y, d in enumerate(zip(self.test_names, self.n_insiders)) if d[1] >= min_num]
        filtered_n_insiders = [d for s, d in enumerate(self.n_insiders) if d >= min_num]
        # Turn the different arrays into pandas Series or dataframes
        pred = pd.Series(filtered_pred, index=filtered_names)
        n_applicability = pd.Series(filtered_n_insiders, index=filtered_names)
        models = self._filter_models_vote(svc, knn, ridge, filtered_names, min_num)
        # concatenate all the objects
        self.pred = pd.concat([pred, n_applicability, models], axis=1)
        self.pred.columns = ["prediction", "AD_number"] + list(models.columns)
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

    def find_max_ad(self, pred1, pred2, pred3):
        """
        find the maximum applicability domain of the 2 preds
        parameters
        ___________
        pred1: array
            svc predictions
        pred2: array
            knn predictions
        pred3: array
            ridge predictions
        """
        assert len(pred1) == len(pred2) == len(pred3), "The predictions have different lengths"
        ad = []
        pred = pred1.copy()
        for idx in pred1.index:
            if pred3["AD_number"].loc[idx] <= pred1["AD_number"].loc[idx] >= pred2["AD_number"].loc[idx]:
                ad.append(f'{pred1["AD_number"].loc[idx]}-svc')
            elif pred1["AD_number"].loc[idx] <= pred2["AD_number"].loc[idx] >= pred3["AD_number"].loc[idx]:
                ad.append(f'{pred2["AD_number"].loc[idx]}-knn')
            else:
                ad.append(f'{pred3["AD_number"].loc[idx]}-ridge')
        pred["AD_number"] = ad
        return pred

    def extract(self, fasta_file, pred1=None, pred2=None, pred3=None, positive_fasta="positive.fasta",
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
        if pred2 is not None and pred3 is not None:
            pred1 = self.find_max_ad(pred1, pred2, pred3)
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


def vote_and_filter(feature_out, fasta_file, min_num=1, res_dir="results", val=1):
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
    svc, ridge, knn, new_svc, X_svc, new_knn, X_knn, new_ridge, X_ridge = ensemble.predicting()
    all_voting, all_index = ensemble.vote(val, *svc.values(), *ridge.values(), *knn.values())
    # applicability domain for scv
    domain_svc = ApplicabilityDomain()
    domain_svc.fit(X_svc)
    domain_svc.predict(new_svc)
    # return the prediction after the applicability domain filter of SVC (the filter depends on the feature space)
    pred_svc = domain_svc.filter(all_voting, svc, knn, ridge, min_num, f"{res_dir}/svc_domain.parquet")
    domain_svc.extract(fasta_file, pred_svc, positive_fasta=f"positive_svc.fasta",
                       negative_fasta=f"negative_svc.fasta", res_dir=res_dir)
    # applicability domain for KNN
    domain_knn = ApplicabilityDomain()
    domain_knn.fit(X_knn)
    domain_knn.predict(new_knn)
    # return the prediction after the applicability domain filter of KNN (so different features will produce different samples to be included)
    pred_knn = domain_knn.filter(all_voting, svc, knn, ridge, min_num, f"{res_dir}/knn_domain.parquet")
    domain_knn.extract(fasta_file, pred_knn, positive_fasta=f"positive_knn.fasta",
                       negative_fasta=f"negative_knn.fasta", res_dir=res_dir)

    # applicability domain for ridge
    domain_ridge = ApplicabilityDomain()
    domain_ridge.fit(X_knn)
    domain_ridge.predict(new_knn)
    # return the prediction after the applicability domain filter of Ridge (We save the different versions)
    pred_ridge = domain_knn.filter(all_voting, svc, knn, ridge, min_num, f"{res_dir}/ridge_domain.parquet")
    domain_ridge.extract(fasta_file, pred_ridge, positive_fasta=f"positive_ridge.fasta",
                       negative_fasta=f"negative_ridge.fasta", res_dir=res_dir)
    # Then filter again to see which sequences are within the AD of the 3 algorithms since it is an ensemble classifier
    name_set = set(pred_svc.index).intersection(set(pred_knn.index), set(pred_ridge.index))
    name_set = sorted(name_set, key=lambda x: int(x.split("_")[1]))
    knn_set = pred_knn.loc[name_set]
    ridge_set = pred_ridge.loc[name_set]
    common_domain = pred_svc.loc[name_set]
    common_domain.to_parquet(f"{res_dir}/common_domain.parquet")
    # the positive sequences extracted will have the AD of the SVC
    domain_knn.extract(fasta_file, common_domain, knn_set, ridge_set,res_dir=res_dir)


def main():
    feature_out, min_num, fasta_file, res_dir, value = arg_parse()
    vote_and_filter(feature_out, fasta_file, min_num, res_dir, value)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
