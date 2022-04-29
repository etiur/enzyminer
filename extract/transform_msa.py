"""
The script transforms the 3D features generated from MSA by the pretrained model esm_msa1b_t12_100M_UR50S into
one dimensional features which can be used by classical machine learning models
"""
import numpy as np
import torch


class FeatureTransformer:
    """
    A class that transforms the pretrained representations into features usable to classical machine learning models
    from sklearn
    """
    def __init__(self, msa_features: torch.Tensor):
        self.msa_features = msa_features

    def aac_pssm(self, layer: int = 12) -> torch.Tensor:
        """
        Analog of aac_pssm function from possum which calculates the average of the features across aa sequence,
        in this case across tokens and different proteins in the alignment

        Parameters
        ___________
        layer: int
            The layer of the model where the features come from

        Returns
        ________
        aac_feature: torch.Tensor
            The mean of the features
        """
        aac_feature = self.msa_features["mean_representations"][layer].mean(dim=0)
        return aac_feature

    @staticmethod
    def normalize(matrix):
        # min_max normalization
        element_min = matrix.amin()
        element_max = matrix.amax()
        d_features = (matrix - element_min) / element_max
        return d_features

    def d_fpssm(self, layer: int = 12) -> torch.Tensor:
        """
        Analog of d_fpssm function from possum which normalizes the features with min max,
        then computes the mean

        Parameters
        ___________
        msa_features: torch.Tensor
            Features from the msa pretrained models
        layer: int
            The layer of the model where the features come from

        Returns
        ________
        d_fpssm_feature: torch.Tensor
            The mean of the normalized features

        """
        # Converts negative numbers in 0
        df_matrix = self.msa_features["representations"][layer].clone()
        df_matrix[df_matrix < 0] = 0
        d_features = self.normalize(df_matrix)
        d_fpssm_feature = d_features.mean(dim=1).mean(dim=0)

        return d_fpssm_feature

    @staticmethod
    def pssm_smth(msa_features, smth_size):
        """
        Parameters
        ___________
        msa_features: torch.Tensor
            The features from MSA in 2D of shape (toks, representations)
        smth_matrix: np.array
            A new matrix to hold the smoothed values from msa_features of the same shape
        smth_size: int
            It has to be an odd number
        sequence_length: int
            The length of the toks

        Return
        _______
        smth_matrix: torch.Tensor
            An array of the smoothed features from msa
        """
        # for each position of the matrix it sums few values from msa_features depending on smth_size
        smth_matrix = torch.zeros(msa_features.shape)
        sequence_length = msa_features.shape[0]
        for i in range(sequence_length):
            if i < (smth_size - 1) / 2:
                for j in range(int(i + (smth_size - 1) / 2 + 1)):
                    smth_matrix[i] += msa_features[j]
            elif i >= (sequence_length - (smth_size - 1) / 2):
                for j in range(int(i - (smth_size - 1) / 2), sequence_length):
                    smth_matrix[i] += msa_features[j]
            else:
                for j in range(int(i - (smth_size - 1) / 2), int(i + (smth_size - 1) / 2 + 1)):
                    smth_matrix[i] += msa_features[j]
        return smth_matrix

    def smoothed_pssm(self, layer=12, smooth=7, slide=5):
        msa_matrix = self.msa_features["representations"][layer].clone()
        new_matrix = []
        for seq in msa_matrix:
            new_matrix.append(self.pssm_smth(seq, smth_size=smooth))
        new_matrix = torch.stack(new_matrix)
        new_matrix = new_matrix[:, ::slide, :]
        new_matrix = self.normalize(new_matrix)
        smooth_feature = new_matrix.mean(dim=1).mean(dim=0)

        return smooth_feature


