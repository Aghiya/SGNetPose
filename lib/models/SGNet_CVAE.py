import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor, JAADPoseFeatureExtractor, JAADAngleFeatureExtractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F

import pdb

class SGNet_CVAE(nn.Module):
    def __init__(self, args):
        super(SGNet_CVAE, self).__init__()
        self.cvae = BiTraPNP(args)
        self.hidden_size = args.hidden_size # GRU hidden size
        self.enc_steps = args.enc_steps # observation step
        self.dec_steps = args.dec_steps # prediction step
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        self.pose_feature_extractor = JAADPoseFeatureExtractor(args)
        # self.angle_feature_extractor = JAADAngleFeatureExtractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        if self.dataset in ['JAAD','PIE']:
            # the predict shift is in pixel
            # self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                     self.pred_dim),
                                                     nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            # the predict shift is in meter
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.addl_drop = nn.Dropout(.1)
        # self.addl_drop_2 = nn.Dropout(.1)
        reduced_size = 4
        # seq 1 - relu
        # seq 2 - relu, lin, relu
        # seq 3 - lin
        # seq 4 - lin reduced to //4
        # seq 5 - lin reduced to //6
        # seq 6 - lin reduced to //8
        # seq 7 - lin reduced to //2
        # seq 8 - relu, lin (already reduced to 128 prior)
        self.addl_process = nn.Sequential(nn.ReLU(inplace=True)
                                          # nn.Linear(self.hidden_size, self.hidden_size//reduced_size)
                                          # ,nn.ReLU(inplace=True)
                                          # ,nn.Linear(self.hidden_size, self.hidden_size)
                                          # ,nn.ReLU(inplace=True)
                                          )
        # self.addl_process_2 = nn.Sequential( nn.ReLU(inplace=True)
                                          # nn.Linear(self.hidden_size, self.hidden_size//reduced_size)
                                          # ,nn.ReLU(inplace=True)
                                          # ,nn.Linear(self.hidden_size, self.hidden_size)
                                          # ,nn.ReLU(inplace=True)
        #                                   )
        self.addl_cell = nn.GRUCell(self.hidden_size//reduced_size, self.hidden_size//reduced_size)
        # self.addl_cell_2 = nn.GRUCell(self.hidden_size//reduced_size, self.hidden_size//reduced_size)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        # self.traj_enc_cell = nn.GRUCell(2 * self.hidden_size + self.hidden_size//4, self.hidden_size)
        # self.traj_enc_cell = nn.GRUCell(3 * self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        # self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        # self.dec_addl_cell = nn.GRUCell(self.hidden_size, self.hidden_size)
        # self.dec_cell = nn.GRUCell(2 * self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4 + self.hidden_size//reduced_size, self.hidden_size)
        # self.dec_cell = nn.GRUCell(3 * self.hidden_size + self.hidden_size//4, self.hidden_size)
    
    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    # def cvae_decoder(self, dec_hidden, goal_for_dec, traj_angle_input, traj_pose_input):
    def cvae_decoder(self, dec_hidden, goal_for_dec, addl_data):
    # def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            
            # make 20 copies of data, then reshape to [x, 512]
            
            # addl_data_2 = torch.cat((traj_angle_input, traj_pose_input), dim=-1).unsqueeze(1).repeat_interleave(20, dim=0)
            addl_data_2 = addl_data.unsqueeze(1).repeat_interleave(20, dim=0)
            addl_data_2 = addl_data_2.view(-1, addl_data_2.shape[-1])
            dec_input = torch.cat((dec_input, addl_data_2), dim=-1)
            
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj

    # def encoder(self, raw_inputs, raw_targets, traj_input, traj_angle_input, traj_pose_input, flow_input=None, start_index = 0):
    def encoder(self, raw_inputs, raw_targets, traj_input, addl_input, flow_input=None, start_index = 0):
    # def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index = 0):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        
        # goal_for_addl_enc = addl_input.new_zeros((addl_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        # traj_enc_addl_hidden = addl_input.new_zeros((addl_input.size(0), self.hidden_size))
        
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            # pdb.set_trace()
            
            # traj_enc_addl_hidden = self.traj_enc_addl_cell(self.enc_drop(torch.cat((addl_input[:,enc_step,:], goal_for_addl_enc), 1)), traj_enc_addl_hidden)
            # traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], traj_enc_addl_hidden, goal_for_enc), 1)), traj_enc_hidden)
            
            # traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], addl_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            addl_data_2 = self.addl_cell(self.addl_drop(addl_input[:, enc_step, :]))
            # pose_data = self.addl_cell(self.addl_drop(traj_pose_input[:,enc_step,:]))
            # angle_data = self.addl_cell_2(self.addl_drop_2(traj_angle_input[:,enc_step,:]))
            # print(addl_data_2.shape)
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            
            # traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], traj_angle_input[:,enc_step,:], traj_pose_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            
            # pdb.set_trace()
            
            enc_hidden = traj_enc_hidden
            # enc_hidden = torch.cat((traj_enc_hidden, traj_enc_addl_hidden), dim=-1)
            
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            # print(enc_hidden.shape, goal_hidden.shape)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:,enc_step,:,:] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            
            if self.training:
                # pdb.set_trace()
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:,:])
                # cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1))/2
            # pdb.set_trace()
            # all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec, traj_angle_input[:,enc_step,:], traj_pose_input[:,enc_step,:])
            # all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec, angle_data, pose_data)
            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec, addl_data_2)
            # all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec, addl_input[:,enc_step,:])
            # all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec, traj_enc_addl_hidden)
            # all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
            
        # convert bbox to ctr
        # all_goal_traj = all_goal_traj[..., :2] + (all_goal_traj[..., 2:] - all_goal_traj[..., :2])/2
        # all_cvae_dec_traj = all_cvae_dec_traj[..., :2] + (all_cvae_dec_traj[..., 2:] - all_cvae_dec_traj[..., :2])/2
            
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities
    
    def forward(self, inputs, input_angle, input_pose, map_mask=None, targets = None, start_index = 0, training=True):  
    # def forward(self, inputs, input_angle, map_mask=None, targets = None, start_index = 0, training=True):    
    # def forward(self, inputs, input_pose, map_mask=None, targets = None, start_index = 0, training=True):
    # def forward(self, inputs, map_mask=None, targets = None, start_index = 0, training=True):
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.dataset in ['JAAD','PIE']:
            traj_input = self.feature_extractor(inputs)
            
            # traj_pose_input = input_pose.flatten(start_dim=2)
            traj_pose_input = input_angle
            traj_pose_input = self.pose_feature_extractor(traj_pose_input)
            traj_pose_input = self.addl_process(traj_pose_input)
            # traj_pose_input = self.addl_drop(traj_pose_input)
            # traj_input = torch.cat((traj_input, traj_pose_input), dim=-1)
            
            # traj_angle_input = self.angle_feature_extractor(input_angle)
            # traj_angle_input = self.addl_process_2(traj_angle_input)
            # traj_input = torch.cat((traj_input, traj_angle_input), dim=-1)

            # all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, traj_angle_input, traj_pose_input)
            # all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, traj_angle_input)
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, traj_pose_input)
            # all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:,start_index:,:] = traj_input_temp
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None, start_index)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities