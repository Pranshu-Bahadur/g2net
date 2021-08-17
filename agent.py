from randaugment import RandAugment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from uuid import uuid4
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from timm import create_model
from modules import Net, LSTMClassifier
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
    model = Net(1)#create_model(model_name, num_classes=1, pretrained=True, in_chans=1)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(torch.load(directory))
    return model

def _save(directory, model, epoch):
    torch.save(model.state_dict(), "{}/./{}.pth".format(directory, f"g2net_custom_net_{epoch}_{uuid4()}"))
                
def update(mode : str, **kwargs):
    correct, running_loss, total, iterations, roc_auc = 0, 0, 0, 0, 0
    for x, y in tqdm(kwargs[mode]):
        x = kwargs['transforms'](x.cuda()).unsqueeze(1)
        if mode == 'train':
            seed = torch.randperm(x.size(0)//2)
            x[seed] = torchvision.transforms.functional.hflip(x[seed])
            h = torch.sigmoid(kwargs['model'](x.cuda()))
        else:
            h = torch.sigmoid(kwargs['model'](x.cuda()))
        loss = kwargs['loss'](h.view(-1).cpu(), y.float())
        if mode == 'train':
            kwargs['model'].zero_grad()
            loss.backward()
            kwargs['optim'].step()
            kwargs['scheduler'].step()
        running_loss += loss.cpu().item()
        roc_auc += roc_auc_score(y.cpu().detach().numpy(), h.view(-1).cpu().detach().numpy())
        y_ = torch.where(h >= 0.5, 1, 0).float()
        total += y.size(0)
        iterations += 1
        correct += (y_.view(-1).cpu()==y.cpu()).sum().item()
        if iterations%kwargs['log_step']==0:
            print(correct/total, roc_auc/iterations)
        torch.cuda.empty_cache()
    return float(correct/total), float(running_loss/iterations), float(roc_auc/iterations)

def _weight_calc(distribution):
    return 1./torch.tensor(distribution).float()

def train(**kwargs):
    #torch.distributed.init_process_group(backend='nccl', world_size=N, init_method='...')
    qtransform_params={"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
    
    model = Net(1)# create_model(kwargs['model'], num_classes=1, pretrained=True, in_chans=1)#
    #model.head.fc = torch.nn.Linear(model.classifier.in_features, 1)
    """
    model = Wav2Vec2ForCTC.from_pretrained("techiaith/wav2vec2-xlsr-ft-cy", num_labels=1)
    model.lm_head = torch.nn.Linear(1024, 1)
    """
    model = torch.nn.DataParallel(model.cuda())
    if kwargs['train_checkpoint']:
        model.load_state_dict(torch.load(kwargs['checkpoint']))
    cqt = CQT1992v2(**qtransform_params, verbose=False).cuda()
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), RandAugment(), torchvision.transforms.ToTensor()])
    dataset = NumpyImagesCSVDataset(kwargs['root'], kwargs['csv'], True, transforms)
    train_split, val_split = _splitter(dataset, 0.9)
    config = {"optim" : torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, alpha=0.9),
              "loss" : torch.nn.BCELoss(),
              "train" : DataLoader(train_split, kwargs['batch_size'], shuffle=True, num_workers=4),
              "val" : DataLoader(val_split, kwargs['batch_size'], shuffle=True, num_workers=4),
              "log_step": kwargs['log_step']
              }
    config['model'] = model.cuda()
    config["scheduler"] = torch.optim.lr_scheduler.StepLR(config['optim'], step_size=2.4, gamma=0.97)
    config['transforms'] = cqt
    flag = True
    for i in range(kwargs['epochs']):
        print(f'training {i+1} :\n')
        config['model'].train()
        metrics = update('train', **config)
        if metrics[0] >= kwargs['optim_threshold'] and flag:
            config['optim'] = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
            config['scheduler'] = torch.optim.lr_scheduler.StepLR(config['optim'], step_size=2.4, gamma=0.97)
            flag = False
            print("Switching Optimizer from RMSprop to SGD")
        print(metrics)
        with torch.no_grad():
            #config['model'].eval()
            print(f'evaluating {i+1}:\n')
            print(update('val', **config))
        if (i+1)%5==0:
           _save("checkpoints", config['model'], i+1)

def _submission(**kwargs):
    qtransform_params={"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
    transforms = CQT1992v2(**qtransform_params, verbose=False).cuda()
    dataset = NumpyImagesCSVDataset(kwargs['root_test'], kwargs['csv'], False, transforms)
    model = _load(kwargs['checkpoint'], kwargs['model'])
    loader = DataLoader(dataset, kwargs['batch_size'], shuffle=False, num_workers=4)
    df = pd.DataFrame(columns=['id', 'target'])
    #model.eval()
    with torch.no_grad():
        for x, name in tqdm(loader):
            h = torch.sigmoid(model(transforms(x.cuda()).unsqueeze(1)))
            df = pd.concat([df, pd.DataFrame.from_dict({"id": list(name), "target": h.view(-1).cpu().tolist()})])
            #print(df.head())
    df.to_csv(kwargs['submission_path'], index=False)


config = {
        'model' : "custom_net",
           'root' : "/home/fraulty/g2net_dataset/train",
           'csv' : "/home/fraulty/g2net_dataset/training_labels.csv",
           'root_test' : "/home/fraulty/g2net_dataset/test",
           'epochs' : 15,
           'checkpoint':"/home/fraulty/g2net/checkpoints/g2net_custom_net_15_307f63ce-f8b7-4d20-96c6-e63184a34813.pth",
           'submission_path': "/home/fraulty/g2net/submissions/submission_8_g2net_custom_net_eval.csv",
           'train_checkpoint': False,
           'log_step': 25,
           'batch_size': 2048,
           'optim_threshold': 0.77
         }
#train(**config)
_submission(**config)
