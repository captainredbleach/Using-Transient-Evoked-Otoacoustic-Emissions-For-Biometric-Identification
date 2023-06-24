import torch
import sklearn.mixture
import numpy
import datasets
from datasets import OAEMatDataset, OAESTFTMatDataset


if __name__ == "__main__":
    dset = OAESTFTMatDataset(r"../datasets/TEOAE_data", type=datasets.DatasetType.TE)
    just_datapoints = [datapoint.numpy().squeeze() for datapoint, _ in dset]
    print(just_datapoints[0].shape)
    gmm_model = sklearn.mixture.GaussianMixture(n_components=179, random_state=2137).fit(just_datapoints)
    print(gmm_model.sample(5))