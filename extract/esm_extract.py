import esm
import torch
import os, argparse
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from glob import glob
import pandas as pd
from os.path import basename


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file")
    parser.add_argument("fasta_file", type=str, help="FASTA file on which to extract representations")
    parser.add_argument("-o", "output_dir", default="pt_features", type=str,
                        help="output directory for extracted representations")
    parser.add_argument("-b", "--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument("-rl", "--repr_layers", type=int, default=[-1], nargs="+",
                        help="layers indices from which to extract representations (0 to num_layers, inclusive)")
    parser.add_argument("-in", "--include", type=str, nargs="+", choices=["mean", "per_tok", "bos", "contacts"],
                        help="specify which representations to return", required=True)
    parser.add_argument("-t", "--truncate", action="store_false",
                        help="Truncate sequences longer than 1024 to match the training setup")
    parser.add_argument("-no", "--nogpu", action="store_true", help="Do not use GPU even if available")

    args = parser.parse_args()

    return [args.fasta_file, args.output_dir, args.toks_per_batch, args.repr_layers, args.include, args.truncate,
            args.nogpu]


def read_esterase_features(labels_path, feature_path="pt_features", num_layers=33, feature="mean_representations",
                           label_type="label"):
    Xs = []
    names = []
    label = pd.read_excel(labels_path, index_col=0, sheet_name="dataset", engine='openpyxl')
    files = glob(f"{feature_path}/*.pt")
    for fil in files:
        names.append(basename(fil.replace(".pt", "")))
        embs = torch.load(fil)
        Xs.append(embs[feature][num_layers])
    Xs = torch.stack(Xs, dim=0).numpy()
    Xs = pd.DataFrame(Xs, index=names)
    Xs = pd.concat([Xs, label[label_type]], axis=1)
    return Xs


def read_features(feature_path="pt_features", num_layers=33, feature="mean_representations"):
    Xs = []
    files = glob(f"{feature_path}/*.pt")
    for fil in files:
        embs = torch.load(fil)
        Xs.append(embs[feature][num_layers])
    Xs = torch.stack(Xs, dim=0).numpy()
    return Xs


def read_sequence(self, filename: str, num=None) -> List[Tuple[str, str]]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    # for fasta files: batch converter needs a sequence of tuples(str, str)
    record = list(SeqIO.parse(filename, "fasta"))
    if not num:
        num = len(record)
    return [(record[i].description, str(record[i].seq)) for i in range(num)]


def remove_insertions(self, sequence: str) -> str:
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(self, filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, self.remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


class CreateFeatures:
    """
    Converts fasta files into features and save them into individual pt files per sequence
    """
    def __init__(self, fasta_file, output_dir="pt_features", toks_per_batch=4096, num_layers=(-1,), include="mean",
                 truncate=True, nogpu=False):
        """
        :param fasta_file: The path to the fasta file
        :param output_dir: The output directory path
        :param toks_per_batch: the maximum batch size
        :param num_layers: which layers to extract the representations from
        :param include: What features to include choices are ["mean", "per_tok", "bos", "contacts"]
        :param truncate: Whether to truncate sequences greater than 1024
        :param nogpu: Not using GPU even available
        """
        # /home/ruite/.cache/torch/checkpoints/esm1b_t33_650M_UR50S.pt
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.fasta_file = fasta_file
        self.output_dir = output_dir
        self.batch_size = toks_per_batch
        self.num_layers = num_layers
        self.include = include
        self.truncate = truncate
        self.nogpu = nogpu
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_esm1b(self):
        self.model.eval()
        if torch.cuda.is_available() and not self.nogpu:
            self.model = self.model.cuda()
            print("Transferred model to GPU")
        dataset = FastaBatchedDataset.from_file(self.fasta_file)
        batches = dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        # separate in batches the fasta files so it is more efficient the feature generation
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self.alphabet.get_batch_converter(), batch_sampler=batches
        )
        print(f"Read {self.fasta_file} with {len(dataset)} sequences")
        return_contacts = "contacts" in self.include
        assert all(-(self.model.num_layers + 1) <= i <= self.model.num_layers for i in self.num_layers)
        repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in self.num_layers]

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
                if torch.cuda.is_available() and not self.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)
                # The model is trained on truncated sequences and passing longer ones in at
                # inference will cause an error. See https://github.com/facebookresearch/esm/issues/21
                if self.truncate:
                    toks = toks[:, :1022]
                # for each batch of sequences generate the pt file
                out = self.model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
                out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                if return_contacts:
                    contacts = out["contacts"].to(device="cpu")
                for i, label in enumerate(labels):
                    output_file = f"{self.output_dir}/{label}.pt"
                    result = {"label": label}
                    # Call clone on tensors to ensure tensors are not views into a larger representation
                    # See https://github.com/pytorch/pytorch/issues/1995
                    if "per_tok" in self.include:
                        # per tok generates 33 tensors (1 per layer) of shape len(aa) X 1280, so each aa has 1280
                        # features
                        result["representations"] = {
                            layer: t[i, 1: len(strs[i]) + 1].clone()
                            for layer, t in representations.items()
                        }
                        # it omits the fist list and then only takes until the length of sequence, because it generates
                        # 1280 representations which is the total representation of the training set however generally
                        # it is shorter
                    if "mean" in self.include:
                        # In mean representations it generates tensors of length 1280, so it is an average of the
                        # aminoacids
                        result["mean_representations"] = {
                            layer: t[i, 1: len(strs[i]) + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }
                    if "bos" in self.include:
                        result["bos_representations"] = {
                            layer: t[i, 0].clone() for layer, t in representations.items()
                        }
                    if return_contacts:
                        result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()
                    # save the files
                    torch.save(result, output_file)


def main():
    fasta_file, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu = arg_parse()
    features = CreateFeatures(fasta_file, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu)
    features.extract_esm1b()
    