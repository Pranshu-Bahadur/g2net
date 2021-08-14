from timm import create_model
import torch
from tdqm import tdqm
from utils import NumpyImagesCSVDataset

def _splitter(dataset, partition : int):
    train_split = int(partition * len(dataset))
    eval_split = int(len(dataset) - train_split)
    splits = [train_split, eval_split]
    return torch.utils.data.dataset.random_split(dataset, splits)

def train(**kwargs):
    model  = torch.nn.DataParallel(create_model(kwargs['model'], num_classes=2, pretrained=True)).cuda()
    train_split, val_split = _splitter(NumpyImagesCSVDataset(kwargs['root'], kwargs['csv'], True), 0.7)
    config = {  "optim" : torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5, nesterov=True),
                "loss" : torch.nn.BCELoss().cuda(),
                "train" : Loader(train_split, 256, shuffle=True, num_workers=4),
                "val" :   Loader(val_split, 256, shuffle=True, num_workers=4)}
    config["scheduler"] = torch.optim.lr_scheduler.StepLR(config['optim'], 2.4, 0.97)
    for i in range(kwargs['epochs']):
        config['model'].train()
        print(f'{i+1}train:\n')
        print(update('train', **config))
        with torch.no_grad():
            config['model'].eval()
            print(f'{i+1}val:\n')
            print(update('val', **config))
    _save("chekpoints", config['model'])
 



def _save(directory, model):
    torch.save(model.state_dict(), "{}/./{}.pth".format(directory, "effnet_v2s"))
              
    

def update(mode : str, **kwargs):
    correct, running_loss, total, iterations = 0, 0, 0, 0
    for x, y in tdqm(kwargs[mode]):
        h = torch.nn.functional.softmax(kwargs['model'](x.cuda()), dim=1)
        loss = kwargs['loss'](h, y.cuda())
        if mode == 'train':
            kwargs['optim'].zero_grad()
            loss.backward()
            kwargs['optim'].step()
            kwargs['scheduler'].step()
        running_loss += loss.cpu().item()
        total = y.size(0)
        iterations += 1
        correct += (torch.argmax(h, dim=1).cpu()==y.cpu()).sum().item()
        print(correct/total, running_loss/iterations)
    return float(correct/total), float(running_loss/iterations)

