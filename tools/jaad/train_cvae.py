import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import lib.utils as utl
from configs.jaad import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils_cvae import train, val, test
from pprint import pprint

import pdb

import warnings
import copy
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):

    for arg, value in vars(args).items():
        if arg in ['epochs', 'batch_size', 'enc_steps', 'dec_steps', 'input_dim', 'pred_dim', 'seed']:
            print(f"{arg}: {value}")

    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))


    model = build_model(args)
    model = nn.DataParallel(model)
    
    
    # model.load_state_dict(torch.load('/home/aghiya/SGNet.pytorch/base_sgnet.pth'))
    # model.eval()
    model = model.to(device)
    
    # val_gen = utl.build_data_loader(args, 'val')
    
    def find_min_distance_rows(tensor1, tensor2):
        """
        Finds rows from tensor2 that minimize the per-element distance to tensor1.
        
        Parameters:
            tensor1 (torch.Tensor): A tensor of shape [45, 4].
            tensor2 (torch.Tensor): A tensor of shape [45, 20, 4].
            
        Returns:
            torch.Tensor: A tensor of shape [45, 4] containing the rows from tensor2 
                          that minimize the per-element distance to tensor1.
        """
        if not isinstance(tensor1, torch.Tensor):
            tensor1 = torch.tensor(tensor1)
        if not isinstance(tensor2, torch.Tensor):
            tensor2 = torch.tensor(tensor2)
        
        # Scaling factors
        scale_factors = torch.tensor([1920, 1080, 1920, 1080], dtype=tensor1.dtype, device=tensor1.device)
        
        # Scale tensor1
        scaled_tensor1 = (tensor1 * scale_factors).int()  # Shape: [45, 4]
        
        # Expand tensor1 to match the shape of tensor2
        expanded_tensor1 = scaled_tensor1.unsqueeze(1)  # Shape: [45, 1, 4]
        
        # Compute absolute differences and sum across the last dimension
        differences = torch.abs(expanded_tensor1 - tensor2).sum(dim=-1)  # Shape: [45, 20]
        
        # Find the index of the row with the minimal difference
        min_indices = differences.argmin(dim=-1)  # Shape: [45]
        
        # Gather the rows with minimal differences
        minimal_diff_rows = tensor2[torch.arange(45), min_indices]  # Shape: [45, 4]
        
        return minimal_diff_rows.int()
     
    def bbox_denormalize(bbox,W=1280,H=640):
        '''
        normalize bbox value to [0,1]
        :Params:
            bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
        :Return:
            bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
        '''
        new_bbox = copy.deepcopy(bbox)
        new_bbox[..., 0] *= W
        new_bbox[..., 1] *= H
        new_bbox[..., 2] *= W
        new_bbox[..., 3] *= H
        
        return new_bbox
        
    def cxcywh_to_x1y1x2y2(boxes):
        '''
        Params:
            boxes:(Cx, Cy, w, h)
        Returns:
            (x1, y1, x2, y2 or tlbr
        '''
        new_boxes = np.zeros_like(boxes)
        new_boxes[...,0] = boxes[...,0] - boxes[...,2]/2
        new_boxes[...,1] = boxes[...,1] - boxes[...,3]/2
        new_boxes[...,2] = boxes[...,0] + boxes[...,2]/2
        new_boxes[...,3] = boxes[...,1] + boxes[...,3]/2
        return new_boxes
    
    
    def denorm(input_traj, target_traj, cvae_all_dec_traj):
    
        K = cvae_all_dec_traj.shape[2]
        tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
        #import pdb; pdb.set_trace()
        input_traj = np.tile(input_traj[:,-1,:][:,None, None,:], (1, 1, K, 1))
        #import pdb; pdb.set_trace()
        tiled_target_traj += input_traj[...,:4]
        cvae_all_dec_traj += input_traj[...,:4]
        
        tiled_target_traj = bbox_denormalize(tiled_target_traj, W=1920, H=1080)
        cvae_all_dec_traj = bbox_denormalize(cvae_all_dec_traj, W=1920, H=1080)

        tiled_target_traj_xyxy = cxcywh_to_x1y1x2y2(tiled_target_traj)
        cvae_all_dec_traj_xyxy = cxcywh_to_x1y1x2y2(cvae_all_dec_traj)
        
        return tiled_target_traj_xyxy, cvae_all_dec_traj_xyxy
    
    # with torch.set_grad_enabled(False):
        # for batch_idx, data in enumerate(val_gen):
            # if batch_idx == 0:
                # input_traj = data['input_x'].to(device)
                # target_traj = data['target_y'].to(device)
                # input_pose = data['input_x_pose'].to(device)
                # input_angle = data['input_x_angle'].to(device)
                # # target_traj_unnormed = data['target_y_unnormed']
                # all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, input_angle=input_angle, input_pose=input_pose, map_mask=None, targets=target_traj)
                # input_traj_np = input_traj.to('cpu').numpy()
                # target_traj_np = target_traj.to('cpu').numpy()
                # cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
                # tiled_target_traj_xyxy, cvae_all_dec_traj_xyxy = denorm(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj_np[:,-1,:,:,:])
                # # min_distance_tensor = find_min_distance_rows(target_traj_np[0][-1], tiled_target_traj_xyxy[0])
                # # target_traj_unnormed = bbox_denormalize(target_traj[0][-1].cpu(), W=1920, H=1080)
            # else:
                # pass
    # pdb.set_trace()
    
    # with open('base_sgnet_pred.txt', 'w') as f:
        # np_arr = tiled_target_traj.numpy()
        # np.savetxt(f, np_arr, fmt='%.2f')
    
    # with open('ground_truth.txt', 'w') as f:
        # # scale_factors = torch.tensor([1920, 1080, 1920, 1080], dtype=target_traj_unnormed.dtype, device=target_traj_unnormed.device)
        # # scaled_target_traj_unnormed = (target_traj_unnormed[0][-1] * scale_factors) # Shape: [45, 4]
        # # np_arr = scaled_target_traj_unnormed.cpu().numpy()
        # np_arr = target_traj_unnormed + input_traj[0][0].cpu().numpy()
        # np.savetxt(f, np_arr, fmt='%.2f')
    
    # pdb.set_trace()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')
    
    # pdb.set_trace()
    
    print("Number of training samples:", train_gen.__len__())
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    # pdb.set_trace()
    
    

    # train
    min_loss = 1e6
    min_MSE_15 = 10e5
    best_model = None
    best_model_metric = None
    
    epoch_range = range(args.start_epoch, args.epochs + args.start_epoch)
    
    
    with tqdm(total=len(epoch_range), desc="Epochs", unit="epoch") as pbar:
        for epoch in epoch_range:
            # train
            train_goal_loss, train_cvae_loss, train_KLD_loss = train(model, train_gen, criterion, optimizer, device)
            # print('Train Epoch: ', epoch, 'Goal loss: ', train_goal_loss, 'Decoder loss: ', train_dec_loss, 'CVAE loss: ', train_cvae_loss, \
            #     'KLD loss: ', train_KLD_loss, 'Total: ', total_train_loss) 
            print('\nTrain Epoch: {} \t Goal loss: {:.4f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}'.format(
                    epoch, train_goal_loss, train_cvae_loss, train_KLD_loss))


            # val
            val_loss = val(model, val_gen, criterion, device)
            lr_scheduler.step(val_loss)


            # test
            test_loss, MSE_15, MSE_05, MSE_10, FMSE, AIOU, FIOU, CMSE, CFMSE = test(model, test_gen, criterion, device)
            print("Test Loss: {:.4f}".format(test_loss))
            print(f"MSE_05: {MSE_05}, MSE_10: {MSE_10}, MSE_15: {MSE_15}, FMSE: {FMSE}, AIOU: {AIOU}, FIOU: {FIOU}, CMSE: {CMSE}, CFMSE: {CFMSE}")
            
            pbar.update(1)
            
    # torch.save(model.state_dict(), "./base_sgnet.pth")


if __name__ == '__main__':
    main(parse_args())
