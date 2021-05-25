from Bio.Blast.Applications import NcbimakeblastdbCommandline as makedb
from Bio.Blast.Applications import NcbipsiblastCommandline as psiblast
import argparse
import os
import glob
from os import path
from os.path import basename, dirname
from Bio import SeqIO
from mpi4py import MPI, futures
from Bio.SeqIO import FastaIO
import time
import logging
from subprocess import call
import shlex
from multiprocessing import Process


def arg_parse():
    parser = argparse.ArgumentParser(description="creates a database and performs psiblast")
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files")
    parser.add_argument("-p", "--pssm_dir", required=False, help="The directory for the pssm files")
    parser.add_argument("-d", "--dbdir", required=False, help="The directory for the database")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The name for the created database")
    parser.add_argument("-n", "--num_thread", required=False, default=10, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-i", "--fasta_file", required=False, help="The fasta file")
    parser.add_argument("-m", "--mpi", required=False, help="true if using mpi to parallelize", action="store_true")
    parser.add_argument("-PO", "--possum", required=False, help="A path to the possum programme")
    parser.add_argument("-num", "--number", required=False, help="a number for the files", default="*")
    args = parser.parse_args()

    return [args.fasta_file, args.fasta_dir, args.pssm_dir, args.dbdir, args.dbinp, args.dbout, args.num_thread,
            args.mpi, args.possum, args.number]


class ExtractPssm:
    """
    A class to extract pssm profiles from protein sequecnes
    """
    def __init__(self, fasta=None, num_threads=10, fasta_dir=None, pssm_dir=None, dbdir=None, dbinp=None, dbout=None,
                 possum_dir=None):
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
        dbdir: str, optional
            The directory for the protein database
        dbinp: str, optional
            The path to the protein database
        dbout: str, optional
            The name of the created databse database
        """
        self.fasta_file = fasta
        if not fasta_dir:
            self.fasta_dir = "/gpfs/projects/bsc72/ruite/feature_extraction/pssm_files/fasta_files"
        else:
            self.fasta_dir = fasta_dir
        if not pssm_dir:
            self.pssm = "/gpfs/projects/bsc72/ruite/feature_extraction/pssm_files/pssm"
        else:
            self.pssm = pssm_dir
        if not dbinp:
            self.dbinp = "/gpfs/home/bsc72/bsc72661/feature_extraction/uniref50.fasta"
        else:
            self.dbinp = dbinp
        if not dbdir:
            self.dbdir = "/gpfs/home/bsc72/bsc72661/feature_extraction/database"
        else:
            self.dbdir = dbdir
        if not dbout:
            self.dbout = f"{self.dbdir}/uniref50"
        else:
            self.dbout = dbout
        self.num_thread = num_threads
        if not possum_dir:
            self.possum = "/gpfs/home/bsc72/bsc72661/feature_extraction/POSSUM_Toolkit"
        else:
            self.possum = possum_dir

    def makedata(self):
        """
        A function that creates a database for the PSI_blast
        """
        if not path.exists(self.dbdir):
            os.makedirs(self.dbdir)
        # running the blast commands
        blast_db = makedb(dbtype="prot", input_file=f"{self.dbinp}", out=f"{self.dbout}")
        stdout_db, stderr_db = blast_db()

        return stdout_db, stderr_db

    def clean_fasta(self):
        """
        Clean the fasta file
        """
        base = dirname(self.fasta_file)
        illegal = f"perl {self.possum}/removeIllegalSequences.pl -i {self.fasta_file} -o {base}/no_illegal.fasta"
        short = f"perl {self.possum}/removeShortSequences.pl -i {base}/no_illegal.fasta -o {base}/no_short.fasta -n 100"
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
        self.clean_fasta()
        base = basename(self.fasta_file)
        with open(f"{base}/no_short.fasta") as inp:
            record = SeqIO.parse(inp, "fasta")
            count = 0
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

    def fast_check(self):
        """
        Accelerates the checking of files
        """
        pros = []
        file = glob.glob(f"{self.pssm}/*.pssm")
        for prep_pdb in file:
            p = Process(target=self._check_pssm, args=(prep_pdb,))
            p.start()
            pros.append(p)
        for p in pros:
            p.join()

    def generate(self, file=None):
        """
        A function that generates the PSSM profiles
        """
        if not path.exists(f"{self.pssm}"):
            os.makedirs(f"{self.pssm}")
        name = basename(file).replace(".fsa", "")
        if not path.exists(f"{self.pssm}/{name}.pssm"):
            psi = psiblast(db=self.dbout, evalue=0.001,
                           num_iterations=3,
                           out_ascii_pssm=f"{self.pssm}/{name}.pssm",
                           save_pssm_after_last_round=True,
                           query=f"{file}",
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
        if not os.path.exists(f"{self.fasta_dir}/seq_3.fsa"):
            self.separate_single()
        self.fast_check()
        file = glob.glob(f"{self.fasta_dir}/{num}*.fsa")
        file.sort(key=lambda x: int(basename(x).replace(".fsa", "").split("_")[1]))
        for files in file:
            self.generate(files)

    def mpi_parallel(self):
        """
        A function that run the generate function in parallel
        """
        com = MPI.COMM_WORLD
        p = com.Get_size()
        logging.basicConfig(filename="time.log")
        if self.fasta_file:
            self.separate_single()
        start = time.time()
        # Using the MPI to parallelize
        with futures.MPICommExecutor() as executor:
            file = glob.glob(f"{self.fasta_dir}/*.fsa")
            executor.map(self.generate, file)
        end = time.time()
        logging.info(f"it took {end-start} to finish all the files")


def generate_pssm(fasta=None, num_threads=10, fasta_dir=None, pssm_dir=None, dbdir=None, dbinp=None, dbout=None,
                  mpi=False, possum=None, num="*"):
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
    dbdir: str, optional
        The directory for the protein database
    dbinp: str, optional
        The path to the protein database
    dbout: str, optional
        The name of the created databse database
    mpi: bool, optional
        If to use MPI or not
    """
    pssm = ExtractPssm(fasta, num_threads, fasta_dir, pssm_dir, dbdir, dbinp, dbout, possum)
    if dbinp and dbout:
        pssm.makedata()
    if not mpi:
        pssm.run_generate(num)
    else:
        pssm.mpi_parallel()


def main():
    fasta_file, fasta_dir, pssm_dir, dbdir, dbinp, dbout, num_thread, mpi, possum, num = arg_parse()
    generate_pssm(fasta_file, num_thread, fasta_dir, pssm_dir, dbdir, dbinp, dbout, mpi, possum, num)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
