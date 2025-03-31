import torch
from .PSTUN import PSTUN

def model_generator(method, scale,device,pretrained_model_path=None):

    if method == 'ours':
        model = PSTUN(in_channels=4,in_feat=128, out_channels=128).to(device)

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
