from uuid import uuid4
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from timm import create_model
from modules import Net
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import NumpyImagesCSVDataset
from nnAudio.Spectrogram import CQT1992v2

def _splitter(dataset, partition : int):
    train_split = int(partition * len(dataset))
    eval_split = int(len(dataset) - train_split)
    splits = [train_split, eval_split]
    return torch.utils.data.dataset.random_split(dataset, splits)

def _load(directory, model_name):
    model = create_model(model_name, num_classes=1, pretrained=True, in_chans=1)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, 1)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(torch.load(directory))
    return model

def _save(directory, model, epoch):
    torch.save(model.state_dict(), "{}/./{}.pth".format(directory, f"g2net_gernet_m_{epoch}_{uuid4()}"))
                
def update(mode : str, **kwargs):
    correct, running_loss, total, iterations, roc_auc = 0, 0, 0, 0, 0
    for x, y in tqdm(kwargs[mode]):
        h = torch.sigmoid(kwargs['model'](kwargs['transforms'](x.cuda()).unsqueeze(1)))
        loss = kwargs['loss'](h.view(-1), y.cuda().float())
        if mode == 'train':
            kwargs['model'].zero_grad()
            loss.backward()
            kwargs['optim'].step()
            kwargs['scheduler'].step()
        running_loss += loss.cpu().item()
        roc_auc += roc_auc_score(y.cpu().detach().numpy(), h.view(-1).cpu().detach().numpy())
        y_ = torch.where(h >=0.5, 1, 0).float()
        total += y.size(0)
        iterations += 1
        correct += (y_.cpu().view(-1)==y.cpu()).sum().item()
        if iterations%5==0:
            print(correct/total, roc_auc/iterations)
        torch.cuda.empty_cache()
    return float(correct/total), float(running_loss/iterations), float(roc_auc/iterations)

def _weight_calc(distribution):
    return 1./torch.tensor(distribution).float()

def train(**kwargs):
    #torch.distributed.init_process_group(backend='nccl', world_size=N, init_method='...')
    qtransform_params={"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
    model = create_model(kwargs['model'], num_classes=1, pretrained=True, in_chans=1)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, 1)
    model = torch.nn.DataParallel(model.cuda())
    transforms = CQT1992v2(**qtransform_params, verbose=False).cuda()
    dataset = NumpyImagesCSVDataset(kwargs['root'], kwargs['csv'], True, transforms)
    train_split, val_split = _splitter(dataset, 0.9)
    #weights = _weight_calc(dataset.distribution)
    config = {"optim" : torch.optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5, alpha=0.9),
              "loss" : torch.nn.BCELoss().cuda(),
              "train" : DataLoader(train_split, 2048, shuffle=True, num_workers=4),
              "val" :   DataLoader(val_split, 2048, shuffle=True, num_workers=4)}
    config['model'] = model.cuda()
    config["scheduler"] = torch.optim.lr_scheduler.StepLR(config['optim'], step_size=2.4, gamma=0.97)
    config['transforms'] = transforms
    flag = True
    for i in range(kwargs['epochs']):    
        print(f'training {i+1} :\n')
        config['model'].train()
        metrics = update('train', **config)
        if metrics[0] >= 0.78 and flag:
            config['optim'] = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5, nesterov=True)
            config['scheduler'] = torch.optim.lr_scheduler.StepLR(config['optim'], step_size=2.4, gamma=0.97)
            flag = False
            print("Switching Optimizer from RMSprop to SGD")
        print(metrics)
        with torch.no_grad():
            config['model'].eval()
            print(f'evaluating {i+1}:\n')
            print(update('val', **config))
        if (i+1)%5==0:
            _save("checkpoints", config['model'], i+1)

def _submission(**kwargs):   
    qtransform_params={"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
    transforms = CQT1992v2(**qtransform_params, verbose=False).cuda()
    dataset = NumpyImagesCSVDataset(kwargs['root_test'], kwargs['csv'], False, transforms)
    model = _load(kwargs['checkpoint'], kwargs['model'])
    loader = DataLoader(dataset, 512, shuffle=False, num_workers=4)
    df = pd.DataFrame(columns=['id', 'target'])
    model.eval()
    with torch.no_grad():
        for x, name in tqdm(loader):
            h = torch.sigmoid(model(transforms(x.cuda()).unsqueeze(1)))
            df = pd.concat([df, pd.DataFrame.from_dict({"id": list(name), "target": h.view(-1).cpu().tolist()})])
            print(df.head())
    df.to_csv(kwargs['submission_path'], index=False)



    

config = {
           'model' : "gernet_m",
           'root' : "/home/fraulty/g2net_dataset/train",
           'csv' : "/home/fraulty/g2net_dataset/training_labels.csv",
           'root_test' : "/home/fraulty/g2net_dataset/test",
           'epochs' : 5,
           'checkpoint':"/home/fraulty/g2net/checkpoints/g2net_gernet_l.pth",
           'submission_path': "/home/fraulty/g2net/submissions/submission_1_g2net_gernet_l_eval.csv",
         }
train(**config)
#_submission(**config)
