import argparse
import os
from prediction.predict import vote_and_filter
from extract.feature_extraction import extract_and_filter
from extract.generate_pssm import generate_pssm


def arg_parse():
    parser = argparse.ArgumentParser(description="extract features using possum and ifeatures")
    parser.add_argument("-i", "--fasta_file", help="The fasta file path")
    parser.add_argument("-p", "--pssm_dir", help="The pssm files directory's path")
    parser.add_argument("-f", "--fasta_dir", required=False, help="The directory for the fasta files")
    parser.add_argument("-if", "--ifeature", required=False, help="Path to the iFeature programme")
    parser.add_argument("-Po", "--possum", required=False, help="A path to the possum programme")
    parser.add_argument("-ifo", "--ifeature_out", required=False, help="The directory for the ifeature extractions")
    parser.add_argument("-op", "--possum_out", required=False, help="The directory for the possum extractions")
    parser.add_argument("-fo", "--filtered_out", required=False, help="The directory for the filtered features")
    parser.add_argument("-di", "--dbinp", required=False, help="The path to the fasta files to create the database")
    parser.add_argument("-do", "--dbout", required=False, help="The name for the created database")
    parser.add_argument("-n", "--num_thread", required=False, default=12, type=int,
                        help="The number of threads to use for the generation of pssm profiles")
    parser.add_argument("-d", "--dbdir", required=False, help="The directory for the database")
    parser.add_argument("-ps", "--positive_sequences", required=False, defalut="positive.fasta",
                        help="The name for the fasta file with the positive sequences")
    parser.add_argument("-ns", "--negative_sequences", required=False, defalut="negative.fasta",
                        help="The name for the fasta file with negative sequences")
    parser.add_argument("-nss", "--number_similar_samples", required=False, default=1, type=int,
                        help="The number of similar training samples to filter the predictions")
    parser.add_argument("-c", "--csv_name", required=False, defalut="common_doamin.csv",
                        help="The name of the csv file for the ensemble prediction")
    parser.add_argument("-r", "--restart", required=False, choices=("pssm", "feature", "predict"),
                        help="From which part of the process to restart with")
    parser.add_argument("-on", "--filter_only", required=False, help="true if you already have the features",
                        action="store_true")
    parser.add_argument("-fl", "--file", required=False, help="The file to restart the extraction of features with")
    parser.add_argument("-lg", "--long", required=False, help="true you have to restart from the long commands",
                        action="store_true")
    parser.add_argument("-r", "--run", required=False, choices=("possum", "ifeature", "both"), default="both",
                        help="run possum or ifeature extraction")
    parser.add_argument("-st", "--start", required=False, type=int, help="The starting number")
    parser.add_argument("-en", "--end", required=False, type=int, help="The ending number")
    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.fasta_dir, args.ifeature, args.possum, args.ifeature_out,
            args.possum_out, args.filtered_out, args.dbdir, args.dbinp, args.dbout, args.num_thread,
            args.number_similar_samples, args.csv_name, args.positive_sequences, args.negative_sequences, args.restart,
            args.ifeature_feature, args.filter_only, args.file, args.long, args.run, args.start, args.end]


class WriteSh:
    def __init__(self, fasta=None, num_threads=10, fasta_dir=None, pssm_dir=None, dbdir=None, dbinp=None, dbout=None,
                 possum=None):
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
        if not possum:
            self.possum = "/gpfs/home/bsc72/bsc72661/feature_extraction/POSSUM_Toolkit/"
        else:
            self.possum = possum

    def write(self, num):
        with open(f"pssm_{num}.sh", "w") as sh:
            lines = ["#!/bin/bash\n", f"#SBATCH -J pssm_{num}.sh\n", f"#SBATCH --output=pssm_{num}.out\n",
                     f"#SBATCH --error=pssm_{num}.err\n", f"#SBATCH --ntasks={self.num_thread}\n\n",
                     "module purge && module load gcc/7.2.0 blast/2.11.0 impi/2018.1 mkl/2018.1 python/3.7.4\n",
                     "echo 'Start at $(date)'\n", 'echo "-------------------------"\n', "python generate_pssm.py"]

            arguments = f"-f {self.fasta_dir} -p {self.pssm} -d {self.dbdir} -di {self.dbinp} -do {self.dbout} " \
                        f"-n {self.num_thread} -i {self.fasta_file} -PO {self.possum} -num {num}"
            python = f"python generate_pssm.py {arguments}\n"
            lines.append(python)
            lines.append('echo "End at $(date)"\n')
            sh.writelines(lines)

        return f"pssm_{num}.sh"

    def write_all(self, start=None, end=None):
        if start and end:
            for num in range(start, end):
                pssm = self.write(num)
                os.system(f"sbatch {pssm}")
        else:
            num = "*"
            pssm = self.write(num)
            os.system(f"sbatch {pssm}")


def main():
    fasta_file, pssm_dir, fasta_dir, ifeature, possum, ifeature_out, possum_out, filtered_out, dbdir, dbinp, dbout, \
    num_thread, min_num, csv_name, positive, negative, restart, ifeature_feature, filter_only, file, long, \
    run, start, end = arg_parse()
    possum_dir = "/gpfs/home/bsc72/bsc72661/feature_extraction/POSSUM_Toolkit"
    if restart == "pssm":
        # generate_pssm(fasta_file, num_thread, fasta_dir, pssm_dir, dbdir, dbinp, dbout)
        sh = WriteSh(fasta_file, num_thread, fasta_dir, pssm_dir, dbdir, dbinp, dbout, possum_dir)
        sh.write_all(start, end)
        # extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature, ifeature_out, possum, possum_out, filtered_out,
        #                    filter_only, run)
        # vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)
    elif restart == "feature":
        extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature, ifeature_out, possum, possum_out, filtered_out,
                           filter_only, file, long, run)
        vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)
    elif restart == "predict":
        vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)

    
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()