import torch
import os
from collections import OrderedDict
from model.MACNet import Params
from model.MACNet import MACNet
def init_model(in_channels,channels,num_half_layer,rs):
    params = Params(in_channels=in_channels, channels=channels,
                             num_half_layer=num_half_layer,rs=rs)
    model = MACNet(params)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params)
    return model
def load_model(model_name,model,device_name):
    out_dir = os.path.join(model_name)
    ckpt_path = os.path.join(out_dir)
    if os.path.isfile(ckpt_path):
        try:
            print('\n existing ckpt detected')
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if device_name=="cpu":
                    name = k[7:]  # remove 'module.' of dataparallel
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        except Exception as e:
            print(e)
            print(f'ckpt loading failed @{ckpt_path}, exit ...')
            exit()

    else:
        print(f'\nno ckpt found @{ckpt_path}')
        exit()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
