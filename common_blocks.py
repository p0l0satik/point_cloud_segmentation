from src.loader.normals_loader import CustomKitti, CustomKittiProcessing
from src.loader.normals_loader import prep_cross_stacked, prep_long_short_stacked
from  src.utils.metrics import *

from unet import UNet


from src.nets.loss import dice_loss
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import wandb
from datetime import datetime
import os
from torchmetrics import JaccardIndex
from tqdm import tqdm

def get_loaders(batch_size = 4, workers = 12):
    training_data = CustomKitti("/home/polosatik/mnt/kitty/dataset/sequences/00/") 
    validation_data = CustomKitti("/home/polosatik/mnt/kitty/dataset/sequences/00/", mode="val") 
    test_data = CustomKitti("/home/polosatik/mnt/kitty/dataset/sequences/00/", mode="test") 

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    return training_loader, validation_loader, test_loader

def get_loaders_param(func, batch_size = 4, workers = 12):
    training_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/", func) 
    validation_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/",func, mode="val") 
    test_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/", func, mode="test") 

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    return training_loader, validation_loader, test_loader

def get_model_and_optimizer(device, in_ch = 3, num_encoding_blocks=5, patience=3):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
    unet = UNet(
          in_channels=in_ch,
          out_classes=2,
          dimensions=2,
          num_encoding_blocks=num_encoding_blocks,
          normalization='batch',
          upsampling_type='linear',
          padding=True,
          activation='ReLU',
      ).to(device)
    model = unet
      
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.1, 
                                                           patience=patience, 
                                                           threshold=0.01)
    return model, optimizer, scheduler

def one_epoch(device, loader, optimizer, model, criterion=None, train=True):
    running_loss = 0.
    last_loss = 0.
    max_classes = 2 #TODO move to config
    model.train(train)
    dataset_len = len(loader)
    for data in tqdm(loader):
        inputs, labels = data
        inputs = torch.tensor(inputs).to(device=device, dtype=torch.float)
        labels = torch.tensor(labels).to(device=device, dtype=torch.long)

        if train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels) + dice_loss(F.softmax(outputs, dim=1).float(),
                                       F.one_hot(labels, max_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    loss = running_loss / dataset_len 
    return loss

def evaluation(device, test_loader, model,config, chpt=""):
    if chpt != "" :
        model.load_state_dict(torch.load(chpt))

    model.train(False)
    model.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    jaccard = JaccardIndex(num_classes=2)
    mIoU = 0
    Precision = 0
    Recall = 0 
    Dice = 0
    Pixel_accuracy = 0 
    time = 0
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True, activation="softmax")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            tinputs, tlabels = data
            tinputs = torch.tensor(tinputs).to(device=device, dtype=torch.float)
            tlabels = torch.tensor(tlabels).to(device=device, dtype=torch.long)

            # Make predictions for this batch
            starter.record()
            toutputs = model(tinputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            batch_iou = 0

            for i in range(len(toutputs)):
                pred = F.softmax(toutputs, dim=1)[i].permute(1,2,0)[:,:, 1]
                batch_iou += jaccard(pred.cpu(), tlabels[i].cpu().int())

            mIoU += batch_iou/config.batch_size
            time += curr_time
            pixel_accuracy, dice, precision, recall = metric_calculator(tlabels.int(), toutputs)
            Pixel_accuracy += pixel_accuracy
            Dice += dice
            Recall += recall
            Precision += precision

            wandb.log({'test inference': curr_time})
            wandb.log({'batch IoU': batch_iou/config.batch_size})
            wandb.log({'precision': precision})
            wandb.log({'recall': recall})
            wandb.log({'DICE': dice})
            wandb.log({'pixel accuracy': pixel_accuracy})
    mIoU /= len(test_loader)
    mTime = time / len(test_loader)
    Precision /= len(test_loader)
    Dice /= len(test_loader)
    Recall /= len(test_loader)
    Pixel_accuracy /= len(test_loader)

    wandb.log({'mean inference': mTime})
    wandb.log({'mean IoU': mIoU})
    wandb.log({'mean precision': Precision})
    wandb.log({'mean recall': Recall})
    wandb.log({'mean DICE': Dice})
    wandb.log({'mean pixel accuracy': Pixel_accuracy})


def train(device, tl, vl, optimizer, model, config, ):
    best_val, best_train, = 1, 1 
    inf_times = []

    for epoch in range(config.n_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        train_loss = one_epoch(device, tl, model=model, criterion=config.criterion, optimizer=optimizer, train=True)

        with torch.no_grad():
            val_loss = one_epoch(device, vl, model=model, criterion=config.criterion, optimizer=optimizer, train=False)

        if val_loss < best_val:
            best_val = val_loss
            model_path = '{}/{}_{}'.format(config.path, config.run_name, epoch)
            torch.save(model.state_dict(), model_path)
        if train_loss < best_train:
            best_train = train_loss

        print('LOSS train {} valid {}'.format(train_loss, val_loss))

        # TODO - move to logger
        wandb.log({'best train accuracy': best_train})
        wandb.log({'best test accuracy': best_val})

        wandb.log({'current train accuracy': train_loss})
        wandb.log({'current test accuracy': val_loss})

        inp, labels = next(iter(vl))
        inp = torch.tensor(inp).to(device=device, dtype=torch.float)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        out = model(inp)
        ender.record()
        torch.cuda.synchronize()

        curr_time = starter.elapsed_time(ender)
        wandb.log({'current inference': curr_time})

        inf_times.append(curr_time)

        gt = labels[0].cpu().detach().numpy().astype('int')
        pred = F.softmax(out, dim=1)[0].cpu().permute(1,2,0).detach().numpy().squeeze()[:,:, 1].round().astype('int')
        class_labels = {1: "plane"}
        wandb.log({'val view': wandb.Image(pred, caption='Val predict')})
        wandb.log({'val gt': wandb.Image(gt, caption='Val gt')})
        wandb.log({"overlayed ":
            wandb.Image(gt, masks={
                "prediction" : {"mask_data" : pred, "class_labels" : class_labels}})
            }
        )

    wandb.log({'mean inference': np.mean(np.asarray(inf_times))})
    

class Config:
    def __init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.n_epochs = 15
        self.run_name = ""
        self.description = ""
        self.batch_size=4
        self.num_enc_blocks=5
        self.inp_channels = 3
        self.dataset="kitti"
        self.curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    def prepare(self):
        self.path = f"new_chpt/{self.run_name}_{self.curr_time}/"

        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)


        wandb_config = dict(
            batch_size=self.batch_size,
            num_blocks=self.num_enc_blocks, 
            inp_channels = self.inp_channels,
            dataset=self.dataset,
            n_epochs = self.n_epochs,
        )

        wandb.init(
            project="PlaneSegmentation",
            notes=self.description,
            config=wandb_config,
            mode="online"
        )

        name = self.run_name +"_"+ self.curr_time 
        wandb.run.name = name 
        