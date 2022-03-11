import esm
import torch
import argparse
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import pandas as pd
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file")
    parser.add_argument("input_location", type=str, help="FASTA file or msa folder on which to extract representations")
    parser.add_argument("-o", "output_dir", default="pt_features", type=str,
                        help="output directory for extracted representations")
    parser.add_argument("-b", "--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument("-rl", "--repr_layers", type=int, default=[-1], nargs="+",
                        help="layers indices from which to extract representations (0 to num_layers, inclusive)")
    parser.add_argument("-in", "--include", type=str, nargs="+", choices=["mean", "per_tok", "contacts"],
                        help="specify which representations to return", required=True)
    parser.add_argument("-t", "--truncate", action="store_false",
                        help="Truncate sequences longer than 1024 to match the training setup")
    parser.add_argument("-no", "--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("-m", "--msa", action="store_true", help="Use msa extraction")

    args = parser.parse_args()

    return [args.input_location, args.output_dir, args.toks_per_batch, args.repr_layers, args.include, args.truncate,
            args.nogpu, args.msa]


class Utilities:
    """
    A set of functions to be used
    """
    @staticmethod
    def read_esterase_features(labels_path, feature_path="pt_features", num_layers=33, feature="mean_representations",
                               label_type="label"):
        Xs = []
        names = []
        label = pd.read_excel(labels_path, index_col=0, sheet_name="dataset", engine='openpyxl')
        files = Path(f"{feature_path}").glob("*.pt")
        for fil in files:
            names.append(fil.stem)
            embs = torch.load(fil)
            Xs.append(embs[feature][num_layers])
        Xs = torch.stack(Xs, dim=0).numpy()
        Xs = pd.DataFrame(Xs, index=names)
        Xs = pd.concat([Xs, label[label_type]], axis=1)
        Y = Xs[label_type].copy()
        X = Xs.drop(label_type, axis=1)
        return X, Y

    @staticmethod
    def read_features(feature_path="pt_features", num_layers=33, feature="mean_representations"):
        Xs = []
        files = Path(f"{feature_path}").glob("*.pt")
        for fil in files:
            embs = torch.load(fil)
            Xs.append(embs[feature][num_layers])
        Xs = torch.stack(Xs, dim=0).numpy()
        return Xs

    @staticmethod
    def read_sequence(filename: str, num=None) -> List[Tuple[str, str]]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        # for fasta files: batch converter needs a sequence of tuples(str, str)
        record = list(SeqIO.parse(filename, "fasta"))
        if not num:
            num = len(record)
        return [(record[i].description, str(record[i].seq)) for i in range(num)]

    @staticmethod  # in static methods there is no passed argument
    def _remove_insertions(sequence: str) -> str:
        # This is an efficient way to delete lowercase characters and insertion characters from a string
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(translation)

    @classmethod  # in classmethod the class itself is passed as the 1rst argument of the function
    def read_msa(cls, filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, cls._remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    @classmethod
    def read_msa_from_folder(cls, msa_folder, nseq: int):
        files = Path(f"{msa_folder}").glob("*.a3m")
        msa_data = [cls.read_msa(fil, nseq) for fil in files]
        return msa_data


class CreateFeatures:
    """
    Converts fasta files into features and save them into individual pt files per sequence
    """
    def __init__(self, msa_input="msa", fasta_file=None, esm_output="pt_features", msa_output="msa_features",
                 toks_per_batch=4096, num_layers=(-1,), include="mean", truncate=True, nogpu=False):
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
        self.model, self.alphabet = None, None
        self.fasta_file = fasta_file
        self.esm_output = Path(esm_output)
        self.msa_input = msa_input
        self.msa_output = Path(msa_output)
        self.batch_size = toks_per_batch
        self.num_layers = num_layers
        self.include = include
        self.truncate = truncate
        self.nogpu = nogpu
        self.esm_output.mkdir(parents=True, exist_ok=True)
        self.msa_output.mkdir(parents=True, exist_ok=True)

    def extract_msa(self):
        self.model, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        if torch.cuda.is_available() and not self.nogpu:
            self.model = self.model.cuda()
            print("Transferred model to GPU")
        msa_batch_converter = self.alphabet.get_batch_converter()
        msa_data = Utilities.read_msa_from_folder(self.msa_input, 64)
        msa_batch = msa_batch_converter(msa_data)
        print(f"Parsed {self.msa_input} with {len(msa_data)} files")
        self.extract(msa_batch)

    def extract_esmb1(self):
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.model.eval()
        dataset = FastaBatchedDataset.from_file(self.fasta_file)
        batches = dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        # separate in batches the fasta files so it is more efficient the feature generation
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self.alphabet.get_batch_converter(), batch_sampler=batches
        )
        print(f"Read {self.fasta_file} with {len(dataset)} sequences")
        self.extract(data_loader, batches)

    def extract(self, data_loader, batches=None):
        return_contacts = "contacts" in self.include
        assert all(-(self.model.num_layers + 1) <= i <= self.model.num_layers for i in self.num_layers)
        repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in self.num_layers]
        if torch.cuda.is_available() and not self.nogpu:
            self.model = self.model.cuda()
            print("Transferred model to GPU")
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
                    output_file = self.esm_output/f"{label}.pt"
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
                    if return_contacts:
                        result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()
                    # save the files
                    torch.save(result, output_file)


def extract_pretrained_features(input_location, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu, msa):
    """
    Extract features from pretrained models

    :param input_location: Fasta file or msa folder from which to extract features
    :param output_dir: The output directory
    :param toks_per_batch: the maximum batch size
    :param repr_layers: which layers to extract the representations from
    :param include: What features to include choices are ["mean", "per_tok", "contacts"]
    :param truncate: Whether to truncate sequences greater than 1024
    :param nogpu: Not using GPU even available
    :param msa: Use the msa model or esm1b

    """
    features = CreateFeatures(input_location, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu)
    if msa:
        features.extract_msa()
    else:
        features.extract_esmb1()


def main():
    input_location, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu, msa = arg_parse()
    extract_pretrained_features(input_location, output_dir, toks_per_batch, repr_layers, include, truncate, nogpu, msa)
    