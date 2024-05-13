
import argparse
import glob
import os
from ep_pred.prediction.predict import vote_and_filter
from ep_pred.extract.feature_extraction import extract_and_filter
from ep_pred.extract.generate_pssm import generate_pssm
from subprocess import call
import shlex
from os.path import dirname, basename, abspath
from Bio import SeqIO
from Bio.SeqIO import FastaIO


def arg_parse():
    parser = argparse.ArgumentParser(description="extract features using possum and ifeatures")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=False)
    parser.add_argument("-p", "--pssm_dir", help="The pssm files directory's path", required=False,
                        default="pssm")
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_files")
    parser.add_argument("-id", "--ifeature_dir", required=False, help="Path to the iFeature programme folder",
                        default="ifeatures")
    parser.add_argument("-Po", "--possum_dir", required=False, help="A path to the possum programme",
                        default="POSSUM_Toolkit")
    parser.add_argument("-io", "--ifeature_out", required=False, help="The directory where the ifeature features are",
                        default="ifeature_features")
    parser.add_argument("-po", "--possum_out", required=False, help="The directory for the possum extractions",
                        default="possum_features")
    parser.add_argument("-fo", "--filtered_out", required=False, help="The directory for the filtered features",
                        default="filtered_features")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The path and name of the created database",
                        default="uniref50")
    parser.add_argument("-n", "--num_thread", required=False, default=24, type=int,
                        help="The number of threads to use for the generation of pssm profiles and feature extraction")
    parser.add_argument("-rs", "--res_dir", required=False,
                        default="results", help="The name for the folder where to store the prediction results")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    parser.add_argument("-re", "--restart", required=False, choices=("feature", "predict"),
                        help="From which part of the process to restart with")
    parser.add_argument("-on", "--filter_only", required=False, help="true if you already have the features",
                        action="store_true")
    parser.add_argument("-er", "--extraction_restart", required=False, help="The file to restart the extraction with")
    parser.add_argument("-lg", "--long", required=False, help="true when restarting from the long commands",
                        action="store_true")
    parser.add_argument("-r", "--run", required=False, choices=("possum", "ifeature", "both"), default="both",
                        help="run possum or ifeature extraction")
    parser.add_argument("-st", "--start", required=False, type=int, help="The starting number", default=1)
    parser.add_argument("-en", "--end", required=False, type=int, help="The ending number, not included")
    parser.add_argument("-sp", "--sbatch_path", required=False,
                        help="The folder to keep the run files for generating pssm", default="run_files")
    parser.add_argument("-v", "--value", required=False, default=1, type=float, choices=(1, 0.8, 0.5),
                        help="The voting threshold to be considered positive")
    parser.add_argument("-iter", "--iterations", required=False, default=3, type=int, help="The number of iterations "
                                                                                         "in PSIBlast")
    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.fasta_dir, args.ifeature_dir, args.possum_dir, args.ifeature_out,
            args.possum_out, args.filtered_out, args.dbinp, args.dbout, args.num_thread, args.number_similar_samples,
            args.res_dir, args.restart, args.filter_only, args.extraction_restart, args.long, args.run, args.start,
            args.end, args.sbatch_path, args.value, args.iterations]

class Launcher_EPpred:
    def __init__(self, fasta_file, pssm_dir, fasta_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, filtered_out, dbinp, dbout,
                num_thread, min_num, res_dir, restart, filter_only, extraction_restart, long, run, iterations) -> None:
        
        self.fasta_file = fasta_file
        self.fasta_dir = fasta_dir
        self.pssm = pssm_dir
        self.dbinp = dbinp
        self.dbout = dbout
        self.num_thread = num_thread
        self.possum = possum_dir
        self.run_path = run
        self.iter = iterations
        self.ifeature_dir = ifeature_dir
        self.ifeature_out = ifeature_out
        self.possum_out = possum_out
        self.filtered_out = filtered_out
        self.min_num = min_num
        self.res_dir = res_dir
        self.restart = restart
        self.filter_only = filter_only
        self.extraction_restart = extraction_restart
        self.long = long
        if fasta_file and dirname(fasta_file) != "":
            self.base = dirname(fasta_file)
        else:
            self.base = "."        
    
    def _clean_fasta(self):
        """
        Clean the fasta file
        """
        illegal = f"perl {self.possum}/utils/removeIllegalSequences.pl -i {self.fasta_file} -o {self.base}/no_illegal.fasta"
        short = f"perl {self.possum}/utils/removeShortSequences.pl -i {self.base}/no_illegal.fasta -o {self.base}/no_short.fasta -n 100"
        call(shlex.split(illegal), close_fds=False)
        call(shlex.split(short), close_fds=False)

    def _clean_fasta(self, length=100):
        """
        Clean the fasta file

        Parameters
        ==========
        length: int
            length_threshold

        """
        
        illegal = f"perl {self.possum}/utils/removeIllegalSequences.pl -i {self.fasta_file} -o {self.base}/no_illegal.fasta"
        short = f"perl {self.possum}/utils/removeShortSequences.pl -i {self.base}/no_illegal.fasta -o {self.base}/no_short.fasta -n {length}"
        call(shlex.split(illegal), close_fds=False)
        call(shlex.split(short), close_fds=False)

    def _separate_single(self):
        """
        A function that separates the fasta files into individual files
        Returns
        
        file: iterator
            An iterator that stores the single-record fasta files
        """
        with open(f"{self.base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            count = 1
            # Write the record into new fasta files
            for seq in record:
                with open(f"{self.fasta_dir}/seq_{count}.fsa", "w") as split:
                    fasta_out = FastaIO.FastaWriter(split, wrap=None)
                    fasta_out.write_record(seq)
                count += 1

    def _remove_sequences_from_input(self):
        """
        A function that removes the fasta sequences that psiblast cannot generate pssm files from,
        from the input fasta file. If inside the remove dir there are fasta files them you have to use this function.
        """
        # Search for fasta files that doesn't have pssm files
        fasta_files = list(map(lambda x: basename(x.replace(".fsa", "")), glob.glob(
            f"{abspath('removed_dir')}/seq_*.fsa")))
        difference = sorted(fasta_files, key=lambda x: int(
            x.split("_")[1]), reverse=True)

        if len(difference) > 0 and not os.path.exists(f"{self.base}/no_short_before_pssm.fasta"):
            with open(f"{self.base}/no_short.fasta") as inp:
                record = SeqIO.parse(inp, "fasta")
                record_list = list(record)
                # Eliminate the sequences from the input fasta file and move the single fasta sequences
                # to another folder
                for files in difference:
                    num = int(files.split("_")[1]) - 1
                    del record_list[num]
                    # Rename the input fasta file so to create a new input fasta file with the correct sequences
                os.rename(f"{self.base}/no_short.fasta",
                          f"{self.base}/no_short_before_pssm.fasta")
                with open(f"{self.base}/no_short.fasta", "w") as out:
                    fasta_out = FastaIO.FastaWriter(out, wrap=None)
                    fasta_out.write_file(record_list)
    

    def launch(self):
        """
        A function that launches the pipeline
        """
        
        if not self.restart:
            if not next(os.scandir(f"{self.fasta_dir}"), False):
                self._clean_fasta()
                self._separate_single()
                generate_pssm(num_threads=self.num_thread, fasta_dir=self.fasta_dir, pssm_dir=self.pssm, dbinp=self.dbinp,
                            dbout=self.dbout, num="*", fasta=self.fasta_file, iterations=self.iter)
                self._remove_sequences_from_input()
                self.restart = "feature"
        if self.restart == "feature":
            extract_and_filter(fasta_file=self.fasta_file, pssm_dir=self.pssm, fasta_dir=self.fasta_dir, ifeature_out=self.ifeature_out, 
                               possum_dir=self.possum_dir, ifeature_dir=self.ifeature_dir, possum_out=self.possum_out,
                               filtered_out=self.filtered_out, filter_only=self.filter_only, long=self.long, 
                               thread=self.num_thread, run = self.run)
            self.restart = "predict"
        if self.restart == "predict":
            vote_and_filter(feature_out=self.filtered_out, fasta_file=self.fasta_file, min_num=self.min_num, res_dir=self.res_dir, val=self.value)

def main():
    """
    A function that runs the whole pipeline

    Parameters
    ==========
    fasta_file: str
        The path to the fasta file
    pssm_dir: str
        The path to the pssm directory
    fasta_dir: str
        The path to the fasta directory
    ifeature_dir: str
        The path to the ifeature directory
    possum_dir: str
        The path to the possum directory
    ifeature_out: str
        The path to the ifeature output directory
    possum_out: str
        The path to the possum output directory
    filtered_out: str
        The path to the filtered output directory
    dbinp: str
        The path to the database input directory
    dbout: str
        The path to the database output directory
    num_thread: int
        The number of threads to use
    min_num: int
        The minimum number of sequences to use
    res_dir: str
        The path to the result directory
    restart: str
        The step to restart from
    filter_only: bool
        If True then only the filtering step will be executed
    long: bool
        If True then the long version of the pipeline will be executed
    run: bool
        If True then the pipeline will be executed
    start: int
        The starting iteration
    end: int
        The ending iteration
    sbatch_path: str
        The path to the sbatch file
    value: int
        The value to use for the filtering
    iterations: int
        The number of iterations to use
    """      

    fasta_file, pssm_dir, fasta_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, filtered_out, dbinp, dbout, \
        num_thread, min_num, res_dir, restart, filter_only, long, run, value, iterations = arg_parse()
    l = Launcher_EPpred(fasta_file, pssm_dir, fasta_dir, ifeature_dir, possum_dir, ifeature_out, possum_out, filtered_out, dbinp, dbout,
        num_thread, min_num, res_dir, restart, filter_only, long, run, value, iterations)
    l.launch()

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
