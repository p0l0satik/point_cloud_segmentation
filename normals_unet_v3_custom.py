from src.loader.normals_loader import CustomKitti, CustomKittiProcessing
from src.loader.normals_loader import prep_long_2_right_up, prep_long_2_half_right_up, prep_stock, prep_cross_aver
from src.nets.custom_unet import UNet

from src.nets.loss import dice_loss
from torch.utils.data import DataLoader
# from unet import UNet
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import wandb
from datetime import datetime

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

def get_model_and_optimizer(device, num_encoding_blocks=5, patience=3):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
    # unet = UNet(
    #       in_channels=3,
    #       out_classes=2,
    #       dimensions=2,
    #       num_encoding_blocks=num_encoding_blocks,
    #       normalization='batch',
    #       upsampling_type='linear',
    #       padding=True,
    #       activation='ReLU',
    #   ).to(device)
    model = UNet(
        in_channels=3,
        out_classes=2, 
        ).to(device)
      
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.1, 
                                                           patience=patience, 
                                                           threshold=0.01)
    return model, optimizer, scheduler

def one_epoch(device, loader, optimizer, model, criterion, train=True):
    running_loss = 0.
    last_loss = 0.
    max_classes = 2 #TODO move to config
    model.train(train)
    dataset_len = len(loader)
    for data in loader:
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

def train(device, tl, vl, optimizer, model, name):
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 20 #TODO move to config
    best_val, best_train, = 1, 1 
    inf_times = []
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        train_loss = one_epoch(device, tl, optimizer, model, criterion, train=True)

        with torch.no_grad():
            val_loss = one_epoch(device, vl, optimizer, model, criterion, train=False)

        if val_loss < best_val:
            best_val = val_loss
            model_path = 'model_{}_{}'.format(name, epoch)
            torch.save(model.state_dict(), model_path)
        if train_loss < best_train:
            best_train = train_loss

        print('LOSS train {} valid {}'.format(train_loss, val_loss))

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
    
        
if __name__ == "__main__":
    wandb_config = dict(
        batch_size=4,
        num_blocks=3, 
        inp_channels = 3,
        dataset="kitti",
    )

    wandb.init(
        project="PlaneSegmentation",
        notes="custom unet test2 with normals",
        config=wandb_config,
        mode="online"
    )
    name = "custom_unet_test2_normals" + datetime.now().strftime('%Y%m%d_%H%M%S') 
    wandb.run.name = name 
    device = "cuda:0"
    tl, vl, test = get_loaders_param(prep_stock)
    model, optimizer, scheduler = get_model_and_optimizer(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    train(device, tl, vl, optimizer, model, name)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    wandb.log({'total train time': np.mean(np.asarray(total_time))})

    
        