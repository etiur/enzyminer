from Bio.Blast.Applications import NcbimakeblastdbCommandline as makedb
from Bio.Blast.Applications import NcbipsiblastCommandline as psiblast
import argparse
import os
import glob
from os import path
from os.path import basename
import time
import logging
from multiprocessing.dummy import Pool


def arg_parse():
    parser = argparse.ArgumentParser(description="creates a database and performs psiblast")
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files",
                        default="fasta_files")
    parser.add_argument("-p", "--pssm_dir", required=False, help="The directory for the pssm files",
                        default="pssm")
    parser.add_argument("-d", "--dbdir", required=False, help="The directory for the database",
                        default="/gpfs/home/bsc72/bsc72661/feature_extraction/database")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The name for the created database")
    parser.add_argument("-n", "--num_thread", required=False, default=100, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-num", "--number", required=False, help="a number for the files", default="*")
    parser.add_argument("-pa", "--parallel", required=False, help="if run parallel to generate the pssm files",
                        action="store_false")
    args = parser.parse_args()

    return [args.fasta_dir, args.pssm_dir, args.dbdir, args.dbinp, args.dbout, args.num_thread,
            args.number, args.parallel]


class ExtractPssm:
    """
    A class to extract pssm profiles from protein sequecnes
    """
    def __init__(self, num_threads=100, fasta_dir="fasta_files", pssm_dir="pssm", dbinp=None, dbout=None,
                 dbdir="/gpfs/home/bsc72/bsc72661/feature_extraction/database"):
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
        self.pssm = pssm_dir
        self.fasta_dir = fasta_dir
        self.pssm = pssm_dir
        self.dbinp = dbinp
        self.dbdir = dbdir
        if not dbout:
            self.dbout = f"{self.dbdir}/uniref50"
        else:
            self.dbout = dbout
        self.num_thread = num_threads

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
        file = glob.glob(f"{self.pssm}/*.pssm")
        Pool(self.num_thread).map(self._check_pssm, file)

    def generate(self, file=None):
        """
        A function that generates the PSSM profiles
        """
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
        if not path.exists(f"{self.pssm}"):
            os.makedirs(f"{self.pssm}")
        self.fast_check()
        file = glob.glob(f"{self.fasta_dir}/{num}*.fsa")
        file.sort(key=lambda x: int(basename(x).replace(".fsa", "").split("_")[1]))
        for files in file:
            self.generate(files)

    def parallel(self, num):
        """
        A function that run the generate function in parallel
        """
        start = time.time()
        # Using the MPI to parallelize
        file = glob.glob(f"{self.fasta_dir}/{num}*.fsa")
        file.sort(key=lambda x: int(basename(x).replace(".fsa", "").split("_")[1]))
        with Pool(processes=self.num_thread) as executor:
            executor.map(self.generate, file)
        end = time.time()
        logging.info(f"it took {end-start} to finish all the files")


def generate_pssm(num_threads=100, fasta_dir="fasta_files", pssm_dir="pssm", dbdir=None, dbinp=None,
                  dbout=None, num="*", parallel=True):
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
    parallel: bool, optional
        True if use parallel to run the generate_pssm
    """
    pssm = ExtractPssm(num_threads, fasta_dir, pssm_dir, dbinp, dbout, dbdir)
    if dbinp and dbout:
        pssm.makedata()
    if not parallel:
        pssm.run_generate(num)
    else:
        pssm.parallel(num)


def main():
    fasta_dir, pssm_dir, dbdir, dbinp, dbout, num_thread, num, parallel = arg_parse()
    generate_pssm(num_thread, fasta_dir, pssm_dir, dbdir, dbinp, dbout, num, parallel)


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()
