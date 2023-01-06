import argparse
import copy
import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import src.data_loader as module_data
import src.model.metric as module_metric
import src.model.model as module_arch

from src.utils import *


def inference(model, data_loader, tokenizer, device):
    model.eval()
    all_video_names,all_video_features,all_text_features = [],[],[]
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            video_name = batch_data['track_id']
            frames = batch_data['frames'].to(device)
            captions = batch_data['captions']
            motions = batch_data['motion'].to(device)
            reid_feats = batch_data['reid_feats'].to(device)

            captions = tokenizer.tokenize(captions)
            if isinstance(captions, torch.Tensor):
                captions = captions.to(device)
            else:
                captions = {k: v.to(device) for k, v in captions.items()}
    
            rtns = model(frames, captions, motions, reid_feats)
            video_features = rtns[0]
            text_features= rtns[-1]

            all_video_names.append(video_name)
            all_video_features.append(video_features)
            all_text_features.append(text_features)

        all_video_features = torch.cat(all_video_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
    sim_mat = compute_sim_mat(all_text_features, all_video_features)
    torch.save(sim_mat,"sim_mats/sim_mat_soup.pth")
    
def main(args):
    
    config = read_json(args.config)
    record_path = args.record_path
    if record_path is None:
        record_path = Path(config['trainer']['save_dir']) / 'models' \
                                / config['name'] / str(config.get('seed', 1))
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    # test data loader 184 for AIC22
    test_loader_args = copy.deepcopy(config['data_loader']['args'])
    test_loader_args['video_setting']['sample'] = 'uniform'
    test_loader_args['text_setting']['sample'] = 'all'
    test_data_loader = getattr(module_data, config['data_loader']['type'])(
                                        config['data_loader']['test_path'], 
                                        shuffle=False, drop_last=False, 
                                        **test_loader_args)

    tokenizer = Tokenizer(config['arch']['args']['base_setting']['type'])
    model = getattr(module_arch, config['arch']['type'])(
                    device=device, **config['arch']['args'])
    
    # inference with the cross-modal similarity model
    if args.soup:
        print("="*30)
        print("Load checkpoint from {}".format(os.path.join(record_path, 'checkpoint-soup.pth')))
        print("="*30)

        state_dict = torch.load(os.path.join(record_path, 'checkpoint-soup.pth'))
        model.load_state_dict(state_dict)
    else:
        print("="*30)
        print("Load checkpoint from {}".format(os.path.join(record_path, 'checkpoint-latest.pth')))
        print("="*30)

        state_dict = torch.load(os.path.join(record_path, 'checkpoint-latest.pth'))  
        model.load_state_dict(state_dict['state_dict'])
    model = model.to(device)

    print('Inferecing ...')
    inference(model, test_data_loader, tokenizer, device)
   

if __name__ == '__main__':
    SEED = 4
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)

    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default="record/models/cityflow_frozen_cg-wf/1/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--record_path', default="record/model_soup/", type=str)
    args.add_argument('--cpu', dest='cuda', action='store_false')
    args.add_argument('--soup', default=True, type=bool)
    args.set_defaults(cuda=True)
    args = args.parse_args()

    main(args)
