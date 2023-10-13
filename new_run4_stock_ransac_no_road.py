from common_blocks import *
from src.loader.normals_loader import *
from SphericalProjectionDataloader.loader.dataset import SphericalProjectionKitti
from pathlib import Path
from torch.utils.data import random_split

class RunConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.run_name = "exp_4_basic_unet_ransac_no_road"
        self.description = "Basic unet + ransac but without road labels"
        self.n_epochs = 30


if __name__ == "__main__":
    config = RunConfig()
    config.prepare()

    device = "cuda:0"

    data = SphericalProjectionKitti(Path("/home/polosatik/mnt/kitti/prep_no_road_ransac"), length=4541)
    generator = torch.Generator().manual_seed(42)
    tl, vl, test = random_split(data, [3700, 541, 300], generator=generator)

    tl = DataLoader(tl, batch_size=4, shuffle=True, num_workers=12)
    vl = DataLoader(vl, batch_size=4, shuffle=True, num_workers=12)
    test = DataLoader(test, batch_size=4, shuffle=True, num_workers=12)

    
    model, optimizer, scheduler = get_model_and_optimizer(device, 
                                                          in_ch=config.inp_channels, 
                                                          num_encoding_blocks=config.num_enc_blocks)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    train(device, tl, vl, optimizer=optimizer, model=model, config=config)
    ender.record()
    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)
    wandb.log({'total train time': np.mean(np.asarray(total_time))})

    evaluation(device=device, test_loader=test, model=model, config=config)