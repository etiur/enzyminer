from Bio.Blast.Applications import NcbimakeblastdbCommandline as makedb
from Bio.Blast.Applications import NcbipsiblastCommandline as psiblast
import argparse
import os
import glob
from os import path
from os.path import basename, dirname, abspath
import time
import logging
from multiprocessing.dummy import Pool
from subprocess import call
import shlex
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import shutil


def arg_parse():
    parser = argparse.ArgumentParser(description="creates a database and performs psiblast")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path", required=False)
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_files")
    parser.add_argument("-p", "--pssm_dir", required=False, help="The directory for the pssm files",
                        default="pssm")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The name for the created database",
                        default="/gpfs/projects/bsc72/ruite/enzyminer/database/uniref50")
    parser.add_argument("-n", "--num_thread", required=False, default=100, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-num", "--number", required=False, help="a number for the files", default="*")
    parser.add_argument("-pa", "--parallel", required=False, help="if run parallel to generate the pssm files",
                        action="store_true")
    parser.add_argument("-Po", "--possum_dir", required=False, help="A path to the possum programme",
                        default="/gpfs/projects/bsc72/ruite/enzyminer/POSSUM_Toolkit/")
    parser.add_argument("-rm", "--remove", required=False, help="To remove the fasta sequences without pssm files",
                        action="store_true")
    args = parser.parse_args()

    return [args.fasta_dir, args.pssm_dir, args.dbinp, args.dbout, args.num_thread, args.number,
            args.parallel, args.fasta_file, args.possum_dir, args.remove]


class ExtractPssm:
    """
    A class to extract pssm profiles from protein sequecnes
    """
    def __init__(self, num_threads=100, fasta_dir="fasta_files", pssm_dir="pssm", dbinp=None,
                 dbout="/gpfs/projects/bsc72/ruite/enzyminer/database/uniref50", fasta=None,
                 possum_dir="/gpfs/projects/bsc72/ruite/enzyminer/POSSUM_Toolkit/"):
        """
        Initialize the ExtractPssm class

        Parameters
        ___________
        fasta: str, optional
            The file to be analysed
        num_threads: int, optional
         The number of threads to use for the generation of pssm profiles
        fasta_dir: str, optional
            The directory of the fasta files
        pssm_dir: str, optional
            The directory for the output pssm files
        dbinp: str, optional
            The path to the protein database
        dbout: str, optional
            The name of the created databse database
        """
        self.fasta_file = fasta
        self.pssm = pssm_dir
        self.fasta_dir = fasta_dir
        self.pssm = pssm_dir
        self.dbinp = dbinp
        self.dbout = dbout
        self.num_thread = num_threads
        self.possum = possum_dir
        if fasta and dirname(fasta) != "":
            self.base = dirname(fasta)
        else:
            self.base = "."

    def makedata(self):
        """
        A function that creates a database for the PSI_blast
        """
        if not path.exists(dirname(self.dbout)):
            os.makedirs(dirname(self.dbout))
        # running the blast commands
        blast_db = makedb(dbtype="prot", input_file=f"{self.dbinp}", out=f"{self.dbout}", title=f"{basename(self.dbout)}")
        stdout_db, stderr_db = blast_db()

        return stdout_db, stderr_db

    def clean_fasta(self):
        """
        Clean the fasta file
        """
        illegal = f"perl {self.possum}/utils/removeIllegalSequences.pl -i {self.fasta_file} -o {self.base}/no_illegal.fasta"
        short = f"perl {self.possum}/utils/removeShortSequences.pl -i {self.base}/no_illegal.fasta -o {self.base}/no_short.fasta -n 100"
        call(shlex.split(illegal), close_fds=False)
        call(shlex.split(short), close_fds=False)

    def separate_single(self):
        """
        A function that separates the fasta files into individual files

        Returns
        _______
        file: iterator
            An iterator that stores the single-record fasta files
        """
        if not os.path.exists(f"{self.fasta_dir}"):
            os.makedirs(f"{self.fasta_dir}")
        with open(f"{self.base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            count = 1
            # write the record into new fasta files
            for seq in record:
                with open(f"{self.fasta_dir}/seq_{count}.fsa", "w") as split:
                    fasta_out = FastaIO.FastaWriter(split, wrap=None)
                    fasta_out.write_record(seq)
                count += 1

    def _check_pssm(self, files):
        """
        Check if the pssm files are correct
        """
        with open(files, "r") as pssm:
            if "PSI" not in pssm.read():
                os.remove(files)

    def fast_check(self, num):
        """
        Accelerates the checking of files
        """
        file = glob.glob(f"{abspath(self.pssm)}/seq_{num}*.pssm")
        with Pool(processes=self.num_thread) as executor:
            executor.map(self._check_pssm, file)

    def generate(self, file=None):
        """
        A function that generates the PSSM profiles
        """
        name = basename(file).replace(".fsa", "")
        if not path.exists(f"{abspath(self.pssm)}/{name}.pssm"):
            psi = psiblast(db=self.dbout, evalue=0.001,
                           num_iterations=3,
                           out_ascii_pssm=f"{abspath(self.pssm)}/{name}.pssm",
                           save_pssm_after_last_round=True,
                           query=file,
                           num_threads=self.num_thread)
            start = time.time()
            stdout_psi, stderr_psi = psi()
            end = time.time()
            print(f"it took {end - start} to finish {name}.pssm")
            return stdout_psi, stderr_psi

    def run_generate(self, num):
        """
        run the generate function
        """
        if not path.exists(f"{abspath(self.pssm)}"):
            os.makedirs(f"{abspath(self.pssm)}")
        self.fast_check(num)
        file = glob.glob(f"{abspath(self.fasta_dir)}/seq_{num}*.fsa")
        file.sort(key=lambda x: int(basename(x).replace(".fsa", "").split("_")[1]))
        for files in file:
            self.generate(files)

    def parallel(self, num):
        """
        A function that run the generate function in parallel
        """
        if not path.exists(f"{abspath(self.pssm)}"):
            os.makedirs(f"{abspath(self.pssm)}")
        self.fast_check(num)
        start = time.time()
        # Using the MPI to parallelize
        file = glob.glob(f"{abspath(self.fasta_dir)}/seq_{num}*.fsa")
        file.sort(key=lambda x: int(basename(x).replace(".fsa", "").split("_")[1]))
        with Pool(processes=self.num_thread) as executor:
            executor.map(self.generate, file)
        end = time.time()
        logging.info(f"it took {end-start} to finish all the files")

    def remove_notpssm_sequences(self):
        """
        A function that removes the fasta sequences that psiblast cannot generate pssm files from,
        from the input fasta file.
        """
        if not os.path.exists("removed_dir"):
            os.makedirs("removed_dir")
        # I search for fasta files that doesn't have pssm files
        pssm_file = set(map(lambda x: basename(x.replace(".pssm", "")), glob.glob(f"{abspath(self.pssm)}/seq_*.pssm")))
        fasta_file = set(map(lambda x: basename(x.replace(".fsa", "")), glob.glob(f"{abspath(self.fasta_dir)}/seq_*.fsa")))
        difference = sorted(list(fasta_file.difference(pssm_file)), key=lambda x: int(x.split("_")[1]), reverse=True)
        if len(difference) > 0:
            with open(f"{self.base}/no_short.fasta") as inp:
                record = SeqIO.parse(inp, "fasta")
                record_list = list(record)
                # I eliminate the sequences from the input fasta file and move the single fasta sequences
                # to another folder
                for files in difference:
                    num = int(files.split("_")[1]) - 1
                    del record_list[num]
                    shutil.move(f"{abspath(self.fasta_dir)}/{files}.fsa", f"{abspath('removed_dir')}/{files}.fsa")
                # I rename the input fasta file so to create a new input fasta file with the correct sequences
                os.rename(f"{self.base}/no_short.fasta", f"{self.base}/no_short_before_pssm.fasta")
                with open(f"{self.base}/no_short.fasta", "w") as out:
                    fasta_out = FastaIO.FastaWriter(out, wrap=None)
                    fasta_out.write_file(record_list)


def generate_pssm(num_threads=100, fasta_dir="fasta_files", pssm_dir="pssm", dbinp=None,
                  dbout="/gpfs/projects/bsc72/ruite/enzyminer/database/uniref50", num="*", parallel=False, fasta=None,
                  possum_dir="/gpfs/projects/bsc72/ruite/enzyminer/POSSUM_Toolkit/", remove=False):
    """
    A function that creates protein databases, generates the pssms and returns the list of files

    fasta: str, optional
        The file to be analysed
    num_threads: int, optional
        The number of threads to use for the generation of pssm profiles
    fasta_dir: str, optional
        The directory of the fasta files
    pssm_dir: str, optional
        The directory for the output pssm files
    dbinp: str, optional
        The path to the protein database
    dbout: str, optional
        The name of the created databse database
    mpi: bool, optional
        If to use MPI or not
    parallel: bool, optional
        True if use parallel to run the generate_pssm
    """
    pssm = ExtractPssm(num_threads, fasta_dir, pssm_dir, dbinp, dbout, fasta, possum_dir)
    if remove:
        pssm.remove_notpssm_sequences()
    else:
        if dbinp and dbout:
            pssm.makedata()
        if fasta and not os.path.exists(f"{fasta_dir}/seq_3.fsa"):
            pssm.clean_fasta()
            pssm.separate_single()
        if not parallel:
            pssm.run_generate(num)
        else:
            pssm.parallel(num)


def main():
    fasta_dir, pssm_dir, dbinp, dbout, num_thread, num, parallel, fasta_file, possum_dir, remove = arg_parse()
    generate_pssm(num_thread, fasta_dir, pssm_dir, dbinp, dbout, num, parallel, fasta_file, possum_dir, remove)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
