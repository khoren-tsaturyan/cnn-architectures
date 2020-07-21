import torch 
import torchvision 
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torch.utils.data import DataLoader 
from torch import nn 
from torch.nn import functional as F 
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import tarfile
import argparse
import pandas as pd
import copy
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
import torch
from models import *
from dataset import Imagenette
from PIL import Image

def get_data(batch_size,download=False):
    mean = [0.4625, 0.4580, 0.4295]
    std = [0.2351, 0.2287, 0.2372]
    data_transforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
    }

    image_dataset = {
        x: Imagenette('./data',
                            download=download,
                            train=x=='train',
                            transform=data_transforms[x])
        for x in ['train','val']
        }
    
    test_size = int(len(image_dataset['val'])*0.25) 
    val_size = len(image_dataset['val'])-test_size
    test_spit,val_split = torch.utils.data.random_split(image_dataset['val'],[test_size,val_size])
    test_sampler = SubsetRandomSampler(test_spit.indices)
    val_sampler = SubsetRandomSampler(val_split.indices)

    dataloaders = {
        'train': DataLoader(image_dataset['train'],batch_size=batch_size,shuffle=True),
        'val': DataLoader(image_dataset['val'],sampler=val_sampler,batch_size=batch_size),
        'test': DataLoader(image_dataset['val'],sampler=test_sampler,batch_size=1)  
    }

    classes = image_dataset['train'].classes    
    dataset_sizes = {'train': len(image_dataset['train']), 'val':val_size, 'test': test_size}

    return dataloaders,classes,dataset_sizes


def get_model(model_name,num_cls,device,pretrained=False):
    model = eval(f'{model_name}({num_cls})')
    if pretrained:
        model.load_state_dict(torch.load(f'models/{model_name}_model.pth'))
    return model.to(device)

def predict_new_image(img_paths,classes,model):
    model = model.cpu()
    mean = [0.4625, 0.4580, 0.4295]
    std = [0.2351, 0.2287, 0.2372]
    preds = []
    data_transforms =  transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
    
    for img_path in img_paths:
        image = Image.open(img_path)
        image = data_transforms(image)
        image = image.view(1,3,224,224)
        output = model(image)
        pred = torch.argmax(output)
        preds.append(classes[pred.item()])

    return preds

def visualize_logs(logs,save_figure=False,save_dir=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(8, 6)

    ax1.plot(logs['Epoch'],logs['Train_acc'],label='train_accuracy')
    ax1.plot(logs['Epoch'],logs['Val_acc'],label='val_accuracy')
    ax1.legend(loc='best')
    ax1.set_title('Accuraccies')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')

    ax2.plot(logs['Epoch'],logs['Train_loss'],label='train_loss')
    ax2.plot(logs['Epoch'],logs['Val_loss'],label='val_loss')
    ax2.legend(loc='best')
    ax2.set_title('Losses')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    fig.tight_layout(pad=1)
    if save_figure:
        fig.savefig(save_dir)
    plt.show()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def train(dataloaders,dataset_sizes,model,device,epochs,criterion,optimizer,scheduler=None):
    best_loss = np.inf
    best_model = copy.deepcopy(model.state_dict())
    df = pd.DataFrame()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lr =[]
    for epoch in range(1,epochs+1):
        print(f'Epoch {epochs}/{epoch}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                batch_size = inputs.size(0) 
                running_loss += loss.item()*batch_size
                running_corrects += torch.sum(preds == labels.data)

            if scheduler and phase == 'val':
                scheduler.step(running_loss)

            epoch_loss = round(running_loss / dataset_sizes[phase],4)
            epoch_acc = round((running_corrects.double() / dataset_sizes[phase]).item(),4)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                lr.append(get_lr(optimizer))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {} Acc: {}'.format(
                phase, epoch_loss, epoch_acc))
            print()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

    print(f'Best Val Loss.: {best_loss:.4f}, In Epoch: {best_epoch}')

    df['Epoch'] = range(1,epochs+1)
    df['Train_loss'] = train_loss
    df['Train_acc'] = train_acc
    df['Val_loss'] = val_loss
    df['Val_acc'] = val_acc
    df['Lr'] = lr
    df['Batch_Size'] = batch_size
    df['Device'] = device

    model.load_state_dict(best_model)
     
    return model,df


def test(model,test_set,test_size,criterion,device):
    model.eval()  

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(test_set):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)


        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    running_loss = round(running_loss / test_size,4)
    running_corrects = round((running_corrects.double() / test_size).item(),4)
    return running_loss,running_corrects

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-name', type=str, default='VGG13',
                        help='name of the models')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloaders,classes,dataset_sizes = get_data(args.batch_size,download=True)
    model = get_model(args.model_name,len(classes),device,pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True,weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,verbose=True)
    criterion = nn.CrossEntropyLoss()

    model,logs = train(dataloaders,dataset_sizes,model,device,args.epochs,criterion,optimizer,scheduler)
    if args.save_model:
        logs.to_csv(f'logs/{args.model_name}_logs.csv',index=False)
        torch.save(model.state_dict(),f'models/{args.model_name}_model.pth')
        visualize_logs(logs,True,f'logs/{args.model_name}_plot.png')

    test_loss,test_acc = test(model,dataloaders['test'],dataset_sizes['test'],criterion,device)
    print(f'Test Loss {test_loss}, Accuracy{test_acc}')
    
   


if __name__ == '__main__':
    main()