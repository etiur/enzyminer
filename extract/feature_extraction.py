import argparse
import os
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import shlex
from subprocess import Popen
import time
import pandas as pd
from mpi4py import MPI, futures
import glob


def arg_parse():
    parser = argparse.ArgumentParser(description="extract features using possum and ifeatures")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=False)
    parser.add_argument("-p", "--pssm_dir", help="The pssm files directory's path", required=False)
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_dir")
    parser.add_argument("-IF", "--ifeature", required=False, help="Path to the iFeature programme")
    parser.add_argument("-PO", "--possum", required=False, help="A path to the possum programme")
    parser.add_argument("-if", "--ifeature_out", required=False, help="The directory where the ifeature features are",
                        default="/gpfs/projects/bsc72/ruite/feature_extraction/power9/ifeature")
    parser.add_argument("-po", "--possum_out", required=False, help="The directory for the possum extractions",
                        default="/gpfs/projects/bsc72/ruite/feature_extraction/power9/possum")
    parser.add_argument("-fo", "--filtered_out", required=False, help="The directory for the filtered features",
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/filtered_features")
    parser.add_argument("-on", "--filter_only", required=False, help="true if you already have the features",
                        action="store_true")
    parser.add_argument("-fl", "--file", required=False, help="The file to restart the extraction with")
    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.fasta_dir, args.ifeature, args.possum, args.ifeature_out,
            args.possum_out, args.filtered_out, args.filter_only, args.file]


class ExtractFeatures:
    """
    A class to extract features using Possum and iFeatures
    """

    def __init__(self, fasta_file, pssm_dir=None, fasta_dir="fasta_dir", ifeature=None, ifeature_out="ifeature", possum=None,
                 possum_out="possum"):
        """
        Initialize the ExtractFeatures class

        Parameters
        ___________
        fasta_file: str
            The fasta file to be analysed
        pssm_dir: str, optional
            The directory of the generated pssm files
        fasta_dir: str, optional
            The directory to store the new fasta files
        ifeature: str, optional
            A path to the iFeature programme
        ifeature_out: str, optional
            A directory for the extraction results from iFeature
        possum: str, optional
            A path to the POSSUM programme
        possum_out: str, optional
            A directory for the extraction results from possum
        """
        self.fasta_file = fasta_file
        if not pssm_dir:
            self.pssm_dir = "/gpfs/home/bsc72/bsc72661/feature_extraction/pssm_files/pssm"
        else:
            self.pssm_dir = pssm_dir
        self.fasta_dir = fasta_dir
        if not ifeature:
            self.ifeature = "~/feature_extraction//iFeature/iFeature.py"
        else:
            self.ifeature = ifeature
        if not possum:
            self.possum = "~/feature_extraction/POSSUM_Toolkit/possum_standalone.pl"
        else:
            self.possum = possum
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out

    def _batch_iterator(self, iterator, batch_size):
        entry = True  # Make sure we loop once
        while entry:
            batch = []
            while len(batch) < batch_size:
                try:
                    entry = next(iterator)
                except StopIteration:
                    entry = None
                if entry is None:
                    # End of file
                    break
                batch.append(entry)
            if batch:
                yield batch

    def _separate_bunch(self):
        """
        A class that separates the fasta files into smaller fasta files

        parameters
        ___________
        num: int
            The number of files to separate the original fasta_file
        """

        if not os.path.exists(f"{self.fasta_dir}"):
            os.makedirs(f"{self.fasta_dir}")
        with open(self.fasta_file) as inp:
            record = SeqIO.parse(inp, "fasta")
            record_list = list(record)
            if len(record_list) > 20_000:
                record_list.clear()
                for i, batch in enumerate(self._batch_iterator(record, 20_000)):
                    filename = f"group_{i+1}.fasta"
                    record_list.append(filename)
                    with open(f"{self.fasta_dir}/{filename}", "w") as split:
                        fasta_out = FastaIO.FastaWriter(split, wrap=None)
                        fasta_out.write_file(batch)
                return True
            else:
                if len(self.fasta_file.split("/")) > 1:
                    name = f"{os.path.dirname(self.fasta_file)}/group_1.fasta"
                else:
                    name = "group_1.fasta"
                os.rename(self.fasta_file, name)
                return False

    def feature_extraction(self, fasta_file):
        """
        A function to run the iFeature programme
        """
        num = fasta_file.replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.ifeature_out}"):
            os.makedirs(f"{self.ifeature_out}")

        # writing all the commands for the ifeature
        types = ["APAAC", "CKSAAGP", "CTDD", "Moran", "Geary"]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in types]

        # writing the commands for possum
        types_possum = ["pssm_composition", "tpc", "k_separated_bigrams_pssm", "dp_pssm"]
        command_1_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in types_possum]

        command_2_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t pse_pssm -a 3 -o {self.possum_out}/pse_pssm_3_{num}.csv']
        long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in long]

        # combining all the commands
        command_1_possum.extend(command_2_possum)
        command_1_possum.extend(command_3_possum)
        commands_1.extend(command_1_possum)
        # using shlex.split to parse the strings into lists for Popen class
        proc = [Popen(shlex.split(command), close_fds=False) for command in commands_1]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def run_possum(self, fasta_file):
        """
        A function to run the possum programme
        """
        if not os.path.exists(f"{self.possum_out}"):
            os.makedirs(f"{self.possum_out}")
        num = fasta_file.replace(".fasta", "").split("_")[1]
        # writing the commands for possum
        types_possum = ["pssm_composition", "tpc", "k_separated_bigrams_pssm", "dp_pssm"]
        command_1_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in types_possum]

        command_2_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t pse_pssm -a 3 -o {self.possum_out}/pse_pssm_3_{num}.csv']
        long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {self.fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in long]
        # using shlex.split to parse the strings into lists for Popen class
        command_1_possum.extend(command_2_possum)
        command_1_possum.extend(command_3_possum)
        proc = [Popen(shlex.split(command), close_fds=False) for command in command_1_possum]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def run_extraction(self, restart=None):
        """
        run the ifeature programme in different processes
        """
        if len(self.fasta_file.split("/")) > 1:
            name = f"{os.path.dirname(self.fasta_file)}/group_1.fasta"
        else:
            name = "group_1.fasta"
        if not os.path.exists(name):
            self._separate_bunch()
        file = glob.glob(f"{self.fasta_dir}/group_*.fasta")
        file.sort(key=lambda x: int(x.replace(".fasta", "").split("_")[1]))
        if restart:
            file = file[file.index(restart):]
        for files in file:
            self.feature_extraction(files)


class ReadFeatures:
    """
    A class to read the generated features
    """
    def __init__(self, fasta_file=None, ifeature_out="/gpfs/projects/bsc72/ruite/feature_extraction/power9/ifeature",
                 possum_out="/gpfs/projects/bsc72/ruite/feature_extraction/power9/possum",
                 filtered_out="/gpfs/home/bsc72/bsc72661/feature_extraction/filtered_features", fasta_dir="fasta_dir"):
        """
        Initialize the class ReadFeatures

        Parameters
        ___________
        fasta_file: str
            The name of the fasta file
        ifeature_out: str, optional
            A directory for the extraction results from iFeature
        possum_out: str, optional
            A directory for the extraction results from possum
        filtered_out: str, optional
            A directory to store the filtered features from all the generated features
        """
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out
        self.features = None
        self.learning = "/gpfs/home/bsc72/bsc72661//feature_extraction/data/esterase_binary.xlsx"
        self.filtered_out = filtered_out
        if fasta_file:
            self.name = os.path.basename(fasta_file).replace(".fasta", "")
        self.fasta_dir = fasta_dir

    def read_ifeature(self, length):
        """
        A function to read features from ifeatures
        name: str
            name of the file
        """
        # ifeature features
        amphy = [pd.read_csv(f"{self.ifeature_out}/APAAC_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        amphy = pd.concat(amphy)
        comp_space_aa_group_pairs = [pd.read_csv(f"{self.ifeature_out}/CKSAAGP_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        comp_space_aa_group_pairs = pd.concat(comp_space_aa_group_pairs)
        distribution = [pd.read_csv(f"{self.ifeature_out}/CTDD_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        distribution = pd.concat(distribution)
        list_geary = [pd.read_csv(f"{self.ifeature_out}/Geary_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        geary = pd.concat(list_geary)
        geary.columns = [f"{x}_geary" for x in geary.columns]
        list_moran = [pd.read_csv(f"{self.ifeature_out}/Moran_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        moran = pd.concat(list_moran)
        moran.columns = [f"{x}_moran" for x in moran.columns]
        # concat the features
        all_data = pd.concat([amphy, geary, moran, comp_space_aa_group_pairs, distribution], axis=1)
        return all_data

    def read_possum(self, ID, length):
        """
        This function will read possum features

        Parameters
        ___________
        ID: array
            An array of the indices for the possum features
        """
        # reads features of possum
        dp_pssm = [pd.read_csv(f"{self.possum_out}/dp_pssm_{i+1}.csv") for i in range(length)]
        dp_pssm = pd.concat(dp_pssm)
        dp_pssm.index = ID
        pssm_cc = [pd.read_csv(f"{self.possum_out}/pssm_cc_{i+1}.csv") for i in range(length)]
        pssm_cc = pd.concat(pssm_cc)
        pssm_cc.index = ID
        pssm_composition = [pd.read_csv(f"{self.possum_out}/pssm_composition_{i+1}.csv") for i in range(length)]
        pssm_composition = pd.concat(pssm_composition)
        pssm_composition.index = ID
        tpc = [pd.read_csv(f"{self.possum_out}/tpc_{i+1}.csv") for i in range(length)]
        tpc = pd.concat(tpc)
        tpc.index = ID
        tri_gram_pssm = [pd.read_csv(f"{self.possum_out}/tri_gram_pssm_{i+1}.csv") for i in range(length)]
        tri_gram_pssm = pd.concat(tri_gram_pssm)
        tri_gram_pssm.index = ID
        bigrams_pssm = [pd.read_csv(f"{self.possum_out}/k_separated_bigrams_pssm_{i+1}.csv") for i in range(length)]
        bigrams_pssm = pd.concat(bigrams_pssm)
        bigrams_pssm.index = ID
        pse_pssm_3 = [pd.read_csv(f"{self.possum_out}/pse_pssm_3_{i+1}.csv") for i in range(length)]
        pse_pssm_3 = pd.concat(pse_pssm_3)
        index = pse_pssm_3.columns
        index_3 = [f"{x}_3" for x in index]
        pse_pssm_3.index = ID
        pse_pssm_3.columns = index_3

        # Possum features
        feature = [dp_pssm, bigrams_pssm, pssm_cc, pssm_composition, tpc, tri_gram_pssm, pse_pssm_3]
        everything = pd.concat(feature, axis=1)

        return everything

    def read(self):
        """
        Reads all the features
        """
        file = glob.glob(f"{self.fasta_dir}/group_*.fasta")
        all_data = self.read_ifeature(len(file))
        everything = self.read_possum(all_data.index, len(file))
        # concatenate the features
        self.features = pd.concat([all_data, everything], axis=1)
        return self.features

    def filter_features(self):
        """
        filter the obtained features
        """
        self.read()
        svc = pd.read_excel(f"{self.learning}", index_col=0, sheet_name="ch2_20")
        knn = pd.read_excel(f"{self.learning}", index_col=0, sheet_name="random_30")
        if svc.isnull().values.any():
            svc.dropna(axis=1, inplace=True)
            svc.drop(["go"], axis=1, inplace=True)
        if knn.isnull().values.any():
            knn.dropna(axis=1, inplace=True)
            knn.drop(["go"], axis=1, inplace=True)

        if not os.path.exists(self.filtered_out):
            os.makedirs(self.filtered_out)
        # write the new features to csv
        features_svc = self.features[svc.columns]
        features_knn = self.features[knn.columns]
        features_svc.to_csv(f"{self.filtered_out}/svc_features.csv", columns=features_svc.columns)
        features_knn.to_csv(f"{self.filtered_out}/knn_features.csv", columns=features_knn.columns)


def extract_and_filter(fasta_file=None, pssm_dir=None, fasta_dir=None, ifeature=None, possum=None,
                       ifeature_out="/gpfs/projects/bsc72/ruite/feature_extraction/power9/ifeature",
                       possum_out="/gpfs/projects/bsc72/ruite/feature_extraction/power9/possum",
                       filtered_out="/gpfs/home/bsc72/bsc72661/feature_extraction/filtered_features",
                       filter_only=False, restart=None):
    """
    A function to extract and filter the features

    Parameters
    __________
    fasta_file: str
        The fasta file to be analysed
    pssm_dir: str, optional
        The directory of the generated pssm files
    fasta_dir: str, optional
        The directory to store the new fasta files
    ifeature: str, optional
        A path to the iFeature programme
    ifeature_out: str, optional
        A directory for the extraction results from iFeature
    possum: str, optional
        A path to the POSSUM programme
    possum_out: str, optional
        A directory for the extraction results from possum
    filtered_out: str, optional
        A directory to store the filtered features from all thegenerated features
    """
    # Feature extraction
    if not filter_only:
        extract = ExtractFeatures(fasta_file, pssm_dir, fasta_dir, ifeature, ifeature_out, possum, possum_out)
        extract.run_extraction(restart)
    # feature filtering
    filtering = ReadFeatures(fasta_file, ifeature_out, possum_out, filtered_out)
    filtering.filter_features()


def main():
    fasta_file, pssm_dir, fasta_dir, ifeature, possum, ifeature_out, possum_out, filtered_out, filter_only, \
    file = arg_parse()
    extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature, possum, ifeature_out, possum_out, filtered_out,
                       filter_only, file)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
