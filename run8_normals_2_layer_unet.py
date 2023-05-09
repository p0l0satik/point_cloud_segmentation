from common_blocks import *
from src.loader.normals_loader import *

class RunConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.run_name = "exp_8_normals_2_layer_unet"
        self.description = "Basic normals built by 1 up and 1 right, but unet has only 2 layers depth"
        self.n_epochs = 15
        self.num_enc_blocks = 2


if __name__ == "__main__":
    config = RunConfig()
    config.prepare()

    device = "cuda:0"
    tl, vl, test = get_loaders_param(prep_stock)
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