import argparse
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
    args = parser.parse_args()

    return [args.fasta_file, args.pssm_dir, args.fasta_dir, args.ifeature, args.possum, args.ifeature_out,
            args.possum_out, args.filtered_out, args.dbdir, args.dbinp, args.dbout, args.num_thread,
            args.number_similar_samples, args.csv_name, args.positive_sequences, args.negative_sequences, args.restart,
            args.ifeature_feature, args.filter_only, args.file, args.long]


def main():
    fasta_file, pssm_dir, fasta_dir, ifeature, possum, ifeature_out, possum_out, filtered_out, dbdir, dbinp, dbout, \
    num_thread, min_num, csv_name, positive, negative, restart, ifeature_feature, filter_only, file, long = arg_parse()
    if not restart or restart == "pssm":
        generate_pssm(fasta_file, num_thread, fasta_dir, pssm_dir, dbdir, dbinp, dbout)
        extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature, ifeature_out, possum, possum_out, filtered_out,
                           filter_only)
        vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)
    elif restart == "feature":
        extract_and_filter(fasta_file, pssm_dir, fasta_dir, ifeature, ifeature_out, possum, possum_out, filtered_out,
                           filter_only, file, long)
        vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)
    elif restart == "predict":
        vote_and_filter(filtered_out, fasta_file, csv_name, min_num, positive, negative)

    
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()