from torch import nn
import model.arch
import logging
from torch.utils.data import DataLoader
import torch.utils.data
from utils.datasets import OAEMatDataset, OAESTFTMatDataset, OAEAugmentedMatDataset
import wandb
import utils
from tqdm import tqdm
import csv
import yaml
import numpy as np
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryFBetaScore
from torchmetrics.classification import BinarySpecificity



dset_train = OAEAugmentedMatDataset("/home/lab-user/datasets/project_data/time-domain (copy)/train",
                                        type=utils.datasets.DatasetType.TE, noise_path="/home/lab-user/datasets/noises")

dset_val = OAEAugmentedMatDataset("/home/lab-user/datasets/project_data/time-domain (copy)/val",
                                        type=utils.datasets.DatasetType.TE)

logging.basicConfig(filename="program.log", level=logging.INFO)

with open('config.yaml') as file:
    sweep_configuration = yaml.load(file, Loader=yaml.FullLoader)
def main():
    torch.manual_seed(2137)
    run = wandb.init()
    # dset = OAEDataset(r"/home/damian/Nextcloud/Documents/AAU/notes - S2/Project/datasets/teoae_suppresion/RawData")

    # oae_artifact_dset = wandb.Artifact(name="OAE-dataset", type='dataset')
    # oae_artifact_dset.add_file(dset)
    # wandb.log_artifact(dset)

    batch_size = int(wandb.config.batch_size)
    epochs = int(wandb.config.epochs)
    lr = float(wandb.config.lr)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=16,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size, shuffle=True, num_workers=16,
                                             drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    oae_classifier = model.arch.MainOAEClassifierSigmoidRelu().to(device)
    optimizer = torch.optim.Adam(oae_classifier.parameters(), lr=lr, weight_decay=lr)
    loss_function = nn.BCEWithLogitsLoss().to(device)
    wandb.watch(oae_classifier)
    metric = BinaryAccuracy(threshold=0.525, num_classes=24).to(device)
    f1_metric = BinaryF1Score(threshold=0.525, num_classes=24).to(device)
    fb_metric = BinaryFBetaScore(beta=1/(24**2-1), threshold=0.525, num_classes=24).to(device)
    precision_metric = BinaryPrecision(threshold=0.525, num_classes=24).to(device)
    recall_metric = BinaryRecall(threshold=0.525, num_classes=24).to(device)
    spec_metric = BinarySpecificity(threshold=0.525, num_classes=24).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
    m = nn.LogSigmoid()
    for epoch in tqdm(range(0, epochs)):
        print(f"Epoch: {epoch}")
        cur_loss = 0.0
        acc_train = 0
        f1_train = 0
        fb_train = 0
        prec_train = 0
        rec_train = 0
        spec_train = 0
        oae_classifier.train(True)
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs, class_names = data
            #print(f"DEBUG: input: {inputs}, \n"
            #      f" cls: {class_names}")
            inputs = torch.nan_to_num(inputs, nan=6.10352e-05, posinf=1, neginf=0)
            inputs = inputs.to(device)
            class_names = class_names.to(device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=True):
                outputs = oae_classifier(inputs)
                loss = loss_function(outputs, class_names)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            cur_loss += scaler.scale(loss).item() / scaler.get_scale()
            scaler.update()
            o = m(outputs)
            out = o.to(torch.float)
            acc_train += metric(torch.exp(out), class_names).item()
            f1_train += f1_metric(torch.exp(out), class_names).item()
            fb_train += fb_metric(torch.exp(out), class_names).item()
            prec_train += precision_metric(torch.exp(out), class_names).item()
            rec_train += recall_metric(torch.exp(out), class_names).item()
            spec_train += spec_metric(torch.exp(out), class_names).item()


        print({"train_loss": cur_loss / len(train_loader), "train_acc": acc_train / len(train_loader), "F1-score_train": f1_train / len(train_loader), "Fb-score_train": fb_train / len(train_loader), "prec_train": prec_train / len(train_loader), "rec_train": rec_train / len(train_loader)})
        wandb.log({"train_loss": cur_loss / len(train_loader), "train_acc": acc_train / len(train_loader), "F1-score_train": f1_train / len(train_loader), "Fb-score_train": fb_train / len(train_loader), "prec_train": prec_train / len(train_loader), "rec_train": rec_train / len(train_loader), "FRR_train": 1 - prec_train / len(train_loader), "FAR_train": 1 - spec_train / len(train_loader), "TNR_train": spec_train / len(train_loader), "Epoch": epoch})

        # model_artifact = wandb.Artifact(name=f"OAE-Models", type="model")
        # model_artifact.add_file(oae_classifier, name=f"model_{epoch}.bin")
        # wandb.log_artifact(model_artifact)
        oae_classifier.train(False)
        oae_classifier.eval()
        with torch.no_grad():

            val_acc = 0
            curr_loss = 0
            f1_val = 0
            fb_val = 0
            prec_val = 0
            rec_val = 0
            spec_val = 0
            for i, data in enumerate(val_loader, 0):
                inputs, class_names = data
                inputs = torch.nan_to_num(inputs, nan=6.10352e-05, posinf=1, neginf=0)
                inputs = inputs.to(device)
                class_names = class_names.to(device)

                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=True):
                    outputs = oae_classifier(inputs)
                    #lout = torch.logit(torch.exp(outputs.to(torch.float)))
                    #lout = torch.nan_to_num(lout, nan=6.10352e-05, posinf=1, neginf=0)
                    loss = loss_function(outputs, class_names)

                curr_loss += scaler.scale(loss).item() / scaler.get_scale()
                o = m(outputs)
                out = o.to(torch.float)
                val_acc += metric(torch.exp(out), class_names).item()
                f1_val += f1_metric(torch.exp(out), class_names).item()
                fb_val += fb_metric(torch.exp(out), class_names).item()
                prec_val += precision_metric(torch.exp(out), class_names).item()
                rec_val += recall_metric(torch.exp(out), class_names).item()
                spec_val += spec_metric(torch.exp(out), class_names).item()


            print({"val_loss": curr_loss / len(val_loader), "val_acc": val_acc / len(val_loader), "F1-score_val": f1_val / len(val_loader), "Fb -score_val": fb_val / len(val_loader), "prec_val": prec_val / len(val_loader), "rec_val": rec_val / len(val_loader)})
            wandb.log({"val_loss": curr_loss / len(val_loader), "val_acc": val_acc / len(val_loader), "F1-score_val": f1_val / len(val_loader), "Fb-score_val": fb_val / len(val_loader), "prec_val": prec_val / len(val_loader), "rec_val": rec_val / len(val_loader), "FRR_val": 1 - prec_val / len(val_loader), "FAR_val": 1 - spec_val / len(val_loader),"TNR_val": spec_val / len(val_loader), "Epoch": epoch})

        torch.save(oae_classifier, f"Saved models/model_{epoch}_{wandb.run.name}.ckpt")
        wandb.log({"Learning rate": scheduler.optimizer.param_groups[0]['lr'], "Epoch": epoch})
        scheduler.step()



if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='norop')
    wandb.agent(sweep_id, function=main, count=20)
    #main()