import argparse
import os
import shutil

from Bio import SeqIO
from Bio.SeqIO import FastaIO
import shlex
from subprocess import Popen
import time
import pandas as pd
import glob
from os.path import basename, dirname
from multiprocessing import Pool


def arg_parse():
    parser = argparse.ArgumentParser(description="extract features using possum and ifeatures")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=True)
    parser.add_argument("-p", "--pssm_dir", help="The pssm files directory's path", required=False,
                        default="pssm")
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_files")
    parser.add_argument("-id", "--ifeature_dir", required=False, help="Path to the iFeature programme folder",
                        default="/gpfs/projects/bsc72/ruite/enzyminer/iFeature")
    parser.add_argument("-Po", "--possum_dir", required=False, help="A path to the possum programme",
                        default="/gpfs/projects/bsc72/ruite/enzyminer/POSSUM_Toolkit/")
    parser.add_argument("-io", "--ifeature_out", required=False, help="The directory where the ifeature features are",
                        default="ifeature_features")
    parser.add_argument("-po", "--possum_out", required=False, help="The directory for the possum extractions",
                        default="possum_features")
    parser.add_argument("-fo", "--filtered_out", required=False, help="The directory for the filtered features",
                        default="filtered_features")
    parser.add_argument("-on", "--filter_only", required=False, help="true if you already have the features",
                        action="store_true")
    parser.add_argument("-er", "--extraction_restart", required=False, help="The file to restart the extraction with")
    parser.add_argument("-lg", "--long", required=False, help="true you have to restart from the long commands",
                        action="store_true")
    parser.add_argument("-r", "--run", required=False, choices=("possum", "ifeature", "both"), default="both",
                        help="run possum or ifeature extraction")
    parser.add_argument("-n", "--num_thread", required=False, default=100, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.fasta_dir, args.ifeature_dir, args.possum_dir, args.ifeature_out,
            args.possum_out, args.filtered_out, args.filter_only, args.extract_restart, args.long, args.run,
            args.num_thread]


class ExtractFeatures:
    """
    A class to extract features using Possum and iFeatures
    """

    def __init__(self, fasta_file, pssm_dir="pssm", fasta_dir="fasta_files", ifeature_out="ifeature_features",
                 possum_out="possum_features", ifeature_dir="/gpfs/projects/bsc72/ruite/enzyminer/iFeature",
                 thread=12, run="both", possum_dir="/gpfs/projects/bsc72/ruite/enzyminer/POSSUM_Toolkit"):
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
        if dirname(fasta_file) != "":
            self.base = dirname(fasta_file)
        else:
            self.base = "."
        if os.path.exists(f"{self.base}/no_short.fasta"):
            self.fasta_file = f"{self.base}/no_short.fasta"
        else:
            self.fasta_file = fasta_file
        self.pssm_dir = pssm_dir
        self.fasta_dir = fasta_dir
        self.ifeature = f"{ifeature_dir}/iFeature.py"
        self.possum = f"{possum_dir}/possum_standalone.pl"
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out
        self.thread = thread
        self.run = run

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
        with open(f"{self.base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            record_list = list(record)
            if len(record_list) > 10_000:
                record_list.clear()
                for i, batch in enumerate(self._batch_iterator(record, 10_000)):
                    filename = f"group_{i+1}.fasta"
                    with open(f"{self.base}/{filename}", "w") as split:
                        fasta_out = FastaIO.FastaWriter(split, wrap=None)
                        fasta_out.write_file(batch)
                return True
            else:
                shutil.copyfile(self.fasta_file, f"{self.base}/group_1.fasta")
                return False

    def ifeature_long(self, fasta_file):
        """
        Extraction of features for time consuming ifeature features

        Parameters
        ----------
        fasta_file: str
            path to the different fasta files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.ifeature_out}"):
            os.makedirs(f"{self.ifeature_out}")
        ifeature_long = ["Moran", "Geary"]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in ifeature_long]
        # using shlex.split to parse the strings into lists for Popen class
        proc = [Popen(shlex.split(command), close_fds=False) for command in commands_1]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def possum_long(self, fasta_file):
        """
        writing the commands to run the possum features that takes a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.possum_out}"):
            os.makedirs(f"{self.possum_out}")
        possum_long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in possum_long]
        # using shlex.split to parse the strings into lists for Popen class
        proc = [Popen(shlex.split(command), close_fds=False) for command in command_3_possum]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def extraction_long(self, fasta_file):
        """
        writing the commands to run the features that takes a lot of time

        Parameters
        ----------
        fasta_file: str
            path to the different files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.possum_out}"):
            os.makedirs(f"{self.possum_out}")
        if not os.path.exists(f"{self.ifeature_out}"):
            os.makedirs(f"{self.ifeature_out}")
        ifeature_long = ["Moran", "Geary"]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in ifeature_long]
        possum_long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in possum_long]
        # combine the commands
        commands_1.extend(command_3_possum)
        proc = [Popen(shlex.split(command), close_fds=False) for command in commands_1]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def extraction_all(self, fasta_file):
        """
        writting the commands to run the programmes

        Parameters
        ----------
        fasta_file: str
            Path to the different fasta files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.possum_out}"):
            os.makedirs(f"{self.possum_out}")
        if not os.path.exists(f"{self.ifeature_out}"):
            os.makedirs(f"{self.ifeature_out}")
        # ifeature features
        types = ["APAAC", "CKSAAGP", "CTDD"]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in types]
        ifeature_long = ["Moran", "Geary"]
        commands_1_long = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in ifeature_long]
        # possum features
        types_possum = ["pssm_composition", "tpc", "k_separated_bigrams_pssm", "dp_pssm"]
        command_1_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in types_possum]
        command_2_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t pse_pssm -a 3 -o {self.possum_out}/pse_pssm_3_{num}.csv']
        possum_long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in possum_long]

        # combine the commands
        commands_1.extend(command_1_possum)
        commands_1.extend(command_2_possum)
        commands_1.extend(commands_1_long)
        commands_1.extend(command_3_possum)
        proc = [Popen(shlex.split(command), close_fds=False) for command in commands_1]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def extraction_ifeature(self, fasta_file):
        """
        run the ifeature programme iteratively

        Parameters
        ----------
        fasta_file: str
            path to the different fasta_files
        """
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.ifeature_out}"):
            os.makedirs(f"{self.ifeature_out}")
        # ifeature features
        types = ["APAAC", "CKSAAGP", "CTDD"]
        commands_1 = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in types]
        ifeature_long = ["Moran", "Geary"]
        commands_1_long = [
            f"python3 {self.ifeature} --file {fasta_file} --type {prog} --out {self.ifeature_out}/{prog}_{num}.tsv" for
            prog in ifeature_long]
        # combining the commands
        commands_1.extend(commands_1_long)
        proc = [Popen(shlex.split(command), close_fds=False) for command in commands_1]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def extraction_possum(self, fasta_file):
        """
        run the possum programme in different iteratively

        Parameters
        ----------
        fasta_file: str
            Path to the different fasta files
        """
        # possum features
        num = basename(fasta_file).replace(".fasta", "").split("_")[1]
        if not os.path.exists(f"{self.possum_out}"):
            os.makedirs(f"{self.possum_out}")
        types_possum = ["pssm_composition", "tpc", "k_separated_bigrams_pssm", "dp_pssm"]
        command_1_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in types_possum]
        command_2_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t pse_pssm -a 3 -o {self.possum_out}/pse_pssm_3_{num}.csv']
        possum_long = ["pssm_cc", "tri_gram_pssm"]
        command_3_possum = [
            f'perl {self.possum} -i {fasta_file} -p {self.pssm_dir} -t {prog} -o {self.possum_out}/{prog}_{num}.csv' for
            prog in possum_long]
        # combining all the commands
        command_1_possum.extend(command_2_possum)
        command_1_possum.extend(command_3_possum)
        # using shlex.split to parse the strings into lists for Popen class
        proc = [Popen(shlex.split(command), close_fds=False) for command in command_1_possum]
        start = time.time()
        for p in proc:
            p.wait()
        end = time.time()

    def run_extraction_parallel(self, restart=None, long=None):
        """
        Using a pool of workers to run the 2 programmes

        Parameters
        ----------
        restart: str
            The file to restart the programmes with
        long:
            If to run only the longer features
        """
        name = f"{self.base}/group_1.fasta"
        if not os.path.exists(name):
            self._separate_bunch()
        file = glob.glob(f"{self.base}/group_*.fasta")
        file.sort(key=lambda x: int(basename(x).replace(".fasta", "").split("_")[1]))
        if restart:
            file = file[file.index(restart):]
        with Pool(processes=self.thread) as pool:
            if self.run == "both":
                if not long:
                    pool.map(self.extraction_all, file)
                else:
                    pool.map(self.extraction_long, file)
            elif self.run == "possum":
                if not long:
                    pool.map(self.extraction_possum, file)
                else:
                    pool.map(self.possum_long, file)
            elif self.run == "ifeature":
                if not long:
                    pool.map(self.extraction_ifeature, file)
                else:
                    pool.map(self.ifeature_long, file)


class ReadFeatures:
    """
    A class to read the generated features
    """
    def __init__(self, fasta_file, ifeature_out="ifeature_features", possum_out="possum_features",
                 filtered_out="filtered_features"):
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
        self.learning = "/gpfs/projects/bsc72/ruite/enzyminer/data/esterase_binary.xlsx"
        self.filtered_out = filtered_out
        if len(fasta_file.split("/")) > 1:
            self.base = os.path.dirname(fasta_file)
        else:
            self.base = "."

    def read_ifeature(self, length):
        """
        A function to read features from ifeatures
        name: str
            name of the file
        """
        # ifeature features
        amphy = [pd.read_csv(f"{self.ifeature_out}/APAAC_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        comp_space_aa_group_pairs = [pd.read_csv(f"{self.ifeature_out}/CKSAAGP_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        distribution = [pd.read_csv(f"{self.ifeature_out}/CTDD_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        geary = [pd.read_csv(f"{self.ifeature_out}/Geary_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        moran = [pd.read_csv(f"{self.ifeature_out}/Moran_{i+1}.tsv", sep="\t", index_col=0) for i in range(length)]
        # concat features if length > 1 else return the dataframe
        if length > 1:
            moran = pd.concat(moran)
            geary = pd.concat(geary)
            amphy = pd.concat(amphy)
            distribution = pd.concat(distribution)
            comp_space_aa_group_pairs = pd.concat(comp_space_aa_group_pairs)
        else:
            moran = moran[0]
            geary = geary[0]
            amphy = amphy[0]
            distribution = distribution[0]
            comp_space_aa_group_pairs = comp_space_aa_group_pairs[0]
        # change the column names
        geary.columns = [f"{x}_geary" for x in geary.columns]
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
        pssm_cc = [pd.read_csv(f"{self.possum_out}/pssm_cc_{i+1}.csv") for i in range(length)]
        pssm_composition = [pd.read_csv(f"{self.possum_out}/pssm_composition_{i+1}.csv") for i in range(length)]
        tpc = [pd.read_csv(f"{self.possum_out}/tpc_{i+1}.csv") for i in range(length)]
        tri_gram_pssm = [pd.read_csv(f"{self.possum_out}/tri_gram_pssm_{i+1}.csv") for i in range(length)]
        bigrams_pssm = [pd.read_csv(f"{self.possum_out}/k_separated_bigrams_pssm_{i+1}.csv") for i in range(length)]
        pse_pssm_3 = [pd.read_csv(f"{self.possum_out}/pse_pssm_3_{i+1}.csv") for i in range(length)]
        # concat if length > 1 else return the dataframe
        if length > 1:
            dp_pssm = pd.concat(dp_pssm)
            pssm_cc = pd.concat(pssm_cc)
            pssm_composition = pd.concat(pssm_composition)
            tpc = pd.concat(tpc)
            tri_gram_pssm = pd.concat(tri_gram_pssm)
            bigrams_pssm = pd.concat(bigrams_pssm)
            pse_pssm_3 = pd.concat(pse_pssm_3)
        else:
            dp_pssm = dp_pssm[0]
            pssm_cc = pssm_cc[0]
            pssm_composition = pssm_composition[0]
            tpc = tpc[0]
            tri_gram_pssm = tri_gram_pssm[0]
            bigrams_pssm = bigrams_pssm[0]
            pse_pssm_3 = pse_pssm_3[0]

        # change the index and columns
        assert len(dp_pssm) == ID, "Difference in length between possum and ifeature Features "
        dp_pssm.index = ID
        pssm_cc.index = ID
        pssm_composition.index = ID
        tpc.index = ID
        tri_gram_pssm.index = ID
        bigrams_pssm.index = ID
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
        file = glob.glob(f"{self.base}/group_*.fasta")
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
        svc_columns = list(svc.columns)
        knn_columns = list(knn.columns)
        features_svc = self.features[svc_columns]
        features_knn = self.features[knn_columns]
        features_svc.to_csv(f"{self.filtered_out}/svc_features.csv", header=True)
        features_knn.to_csv(f"{self.filtered_out}/knn_features.csv", header=True)


def extract_and_filter(fasta_file=None, pssm_dir="pssm", fasta_dir="fasta_files", ifeature_out="ifeature_features",
                       possum_dir="/gpfs/home/bsc72/bsc72661/feature_extraction/POSSUM_Toolkit",
                       ifeature_dir="/gpfs/projects/bsc72/ruite/enzyminer/iFeature", possum_out="possum_features",
                       filtered_out="filtered_features", filter_only=False, restart=None, long=False, thread=100,
                       run="both"):
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
    thred: int
        The number of poolworkers to use to run the programmes
    run: str
        which programme to run
    """
    # Feature extraction
    if not filter_only:
        extract = ExtractFeatures(fasta_file, pssm_dir, fasta_dir, ifeature_out, possum_out, ifeature_dir, thread, run,
                                  possum_dir)
        extract.run_extraction_parallel(restart, long)

    # feature filtering
    filtering = ReadFeatures(fasta_file, ifeature_out, possum_out, filtered_out)
    filtering.filter_features()


def main():
    fasta_file, pssm_dir, fasta_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, filtered_out, filter_only, \
    extraction_restart, long, run, num_thread = arg_parse()
    extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature_out, possum_dir, ifeature_dir, possum_out,
                       filtered_out, filter_only, extraction_restart, long, num_thread, run)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
