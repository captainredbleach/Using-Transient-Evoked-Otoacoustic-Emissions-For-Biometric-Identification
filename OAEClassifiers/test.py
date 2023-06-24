import datetime
import pathlib

import numpy as np
import scipy
from torch import nn
import model.arch

import logging
from torch.utils.data import DataLoader
import torch.utils.data
from utils.datasets import OAEMatDataset, OAESTFTMatDataset, OAEAugmentedMatDataset, OAEAugmentedMatDatasetTest, \
    DatasetType
import wandb
import utils
from tqdm import tqdm
import csv
import yaml
logging.basicConfig(filename="program.log", level=logging.INFO)
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryFBetaScore
from torchmetrics.classification import BinarySpecificity

def main():
    torch.manual_seed(2137)
    run = wandb.init(project="OAEExamOutsiderTestResults")
    test_dset = OAEAugmentedMatDatasetTest("/home/lab-user/datasets/project_data/outsider_data/Test",
                                        type=utils.datasets.DatasetType.TE)
    #test_dset = scipy.io.loadmat("/home/lab-user/datasets/project_data/outsider_data/Test/*")
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=True, num_workers=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_load = pathlib.Path("Saved models").glob("*_19_rose-sweep-1*")
    best_model = ""

    metric = BinaryAccuracy(threshold=0.7, num_classes=24).to(device)
    f1_metric = BinaryF1Score(threshold=0.7, num_classes=24).to(device)
    fb_metric = BinaryFBetaScore(beta=1 / (24 ** 2 - 1), threshold=0.7, num_classes=24).to(device)
    precision_metric = BinaryPrecision(threshold=0.7, num_classes=24).to(device)
    recall_metric = BinaryRecall(threshold=0.7, num_classes=24).to(device)
    spec_metric = BinarySpecificity(threshold=0.7, num_classes=24).to(device)
    '''
    metric = BinaryAccuracy(threshold=0.525, num_classes=24).to(device)
    f1_metric = BinaryF1Score(threshold=0.525, num_classes=24).to(device)
    fb_metric = BinaryFBetaScore(beta=1 / (24 ** 2 - 1), threshold=0.525, num_classes=24).to(device)
    precision_metric = BinaryPrecision(threshold=0.525, num_classes=24).to(device)
    recall_metric = BinaryRecall(threshold=0.525, num_classes=24).to(device)
    spec_metric = BinarySpecificity(threshold=0.525, num_classes=24).to(device)
    '''
    tp = 0
    m = nn.LogSigmoid()
    for loaded_model in model_load:
        oae_classifier = torch.load(str(loaded_model)).to(device)
        oae_classifier.eval()
        wandb.watch(oae_classifier)
        print(f"Loaded model: {loaded_model}")
        with torch.no_grad():

            val_acc = 0
            curr_loss = 0
            f1_val = 0
            fb_val = 0
            prec_val = 0
            rec_val = 0
            spec_val = 0
            for i, data in enumerate(test_loader, 0):
                start = datetime.datetime.now()
                inputs, class_names = data
                inputs = torch.nan_to_num(inputs, nan=6.10352e-05, posinf=1, neginf=0)
                inputs = inputs.to(device)
                class_names = class_names.to(device)

                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16,
                                    enabled=True):
                    outputs = oae_classifier(inputs)
                    # lout = torch.logit(torch.exp(outputs.to(torch.float)))
                    # lout = torch.nan_to_num(lout, nan=6.10352e-05, posinf=1, neginf=0)

                o = m(outputs)
                out = o.to(torch.float)
                val_acc += metric(torch.exp(out), class_names).item()
                f1_val += f1_metric(torch.exp(out), class_names).item()
                fb_val += fb_metric(torch.exp(out), class_names).item()
                prec_val += precision_metric(torch.exp(out), class_names).item()
                rec_val += recall_metric(torch.exp(out), class_names).item()
                spec_val += spec_metric(torch.exp(out), class_names).item()
                end = datetime.datetime.now()
                print((end-start).microseconds)


            print({"val_loss": curr_loss / len(test_loader), "val_acc": val_acc / len(test_loader),
                       "F1-score_val": f1_val / len(test_loader), "Fb-score_val": fb_val / len(test_loader),
                       "prec_val": prec_val / len(test_loader), "rec_val": rec_val / len(test_loader),
                       "FRR_val": 1 - prec_val / len(test_loader), "FAR_val": 1 - spec_val / len(test_loader),
                       "TNR_val": spec_val / len(test_loader)})
            wandb.log({"val_loss": curr_loss / len(test_loader), "val_acc": val_acc / len(test_loader),
                       "F1-score_val": f1_val / len(test_loader), "Fb-score_val": fb_val / len(test_loader),
                       "prec_val": prec_val / len(test_loader), "rec_val": rec_val / len(test_loader),
                       "FRR_val": 1 - prec_val / len(test_loader), "FAR_val": 1 - spec_val / len(test_loader),
                       "TNR_val": spec_val / len(test_loader)})




if __name__ == '__main__':
    main()