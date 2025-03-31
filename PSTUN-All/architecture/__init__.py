import torch

from .PSTUN import PSTUN


def model_generator(method, device,dataset,pretrained_model_path=None):
    if dataset == 'chikusei':
        if method == 'ours':
            model = PSTUN(in_channels=4, out_channels=128, in_feat=128).to(device)
        else:
            print(f'Method {method} is not defined !!!!')

    elif dataset == 'xiongan':
        if method == 'ours':
            model = PSTUN(in_channels=4, out_channels=93, in_feat=93).to(device)
        else:
            print(f'Method {method} is not defined !!!!')
    
    else:
        print(f'Dataset {dataset} is not defined !!!!')

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},strict=True)
    return model
