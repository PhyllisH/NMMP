'''
Re-implement the coordinates transformation through iteration.

Author: Phyllis
Date: 2019-5-31
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from mobilenet_v2 import MobileNetV2
from alexnet import AlexNet
import numpy as np
from utils import rel_mat


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class CNNEncoder(nn.Module):
    """docstring for CNNEncoder"""

    def __init__(self, prev_steps=6, pred_steps=30, n_hid=256, mb2=None):
        super(CNNEncoder, self).__init__()
        self.pred_steps = pred_steps
        self.image_encoder = MobileNetV2() if mb2 is None else mb2
        self.image_emb = nn.Linear(4096, n_hid)
        self.prev_traj_emb = nn.Linear(prev_steps * 2, n_hid)

        self.fusion = nn.Linear(n_hid * 2, n_hid)
        self.pred_fc = nn.Linear(n_hid, pred_steps * 2)

    def forward(self, rasterized_image, prev_ego_centers):
        batch_size = prev_ego_centers.shape[0]
        # starting location
        rel_pos = torch.zeros(prev_ego_centers.shape).cuda()
        rel_pos[:, 1:, :] = prev_ego_centers[:, 1:, :] - prev_ego_centers[:, :-1, :]

        current_loc = prev_ego_centers[:, -1]  # [b, 2]
        current_loc = torch.unsqueeze(current_loc, 1).expand(batch_size, self.pred_steps, 2).contiguous()
        current_loc = current_loc.view(batch_size, -1)

        # embed the previous trajectories
        rel_pos = rel_pos.view(rel_pos.shape[0], -1)  # [b, prev_steps*2]
        traj_feat = self.prev_traj_emb(rel_pos)  # [b, n_hid]

        # encode the ratserized image
        if rasterized_image is not None:
            rasterized_image = rasterized_image.permute(0, 3, 1, 2)  # [b, 3, 500, 500]
            image_feat = self.image_encoder(rasterized_image)  # [b, 4096]
            image_feat = self.image_emb(image_feat)
            fused_feat = self.fusion(torch.cat([image_feat, traj_feat], dim=1))
        else:
            fused_feat = traj_feat
        rel = self.pred_fc(fused_feat)
        pos = torch.cumsum(rel.view(batch_size, -1, 2), dim=1).view(batch_size, -1) + current_loc
        return rel, pos


class AlexEncoder(nn.Module):
    """docstring for AlexEncoder"""

    def __init__(self, prev_steps=6, pred_steps=30, n_hid=256, alex=None):
        super(AlexEncoder, self).__init__()
        self.pred_steps = pred_steps
        self.image_encoder = AlexNet() if alex is None else alex
        self.image_emb = nn.Linear(4096, n_hid)
        self.prev_traj_emb = nn.Linear(prev_steps * 2, n_hid)

        self.fusion = nn.Linear(n_hid * 2, n_hid)
        self.pred_fc = nn.Linear(n_hid, pred_steps * 2)

    def forward(self, rasterized_image, prev_ego_centers):
        batch_size = prev_ego_centers.shape[0]
        # starting location
        rel_pos = torch.zeros(prev_ego_centers.shape).cuda()
        rel_pos[:, 1:, :] = prev_ego_centers[:, 1:, :] - prev_ego_centers[:, :-1, :]

        current_loc = prev_ego_centers[:, -1]  # [b, 2]
        current_loc = torch.unsqueeze(current_loc, 1).expand(batch_size, self.pred_steps, 2).contiguous()
        current_loc = current_loc.view(batch_size, -1)

        # embed the previous trajectories
        rel_pos = rel_pos.view(rel_pos.shape[0], -1)  # [b, prev_steps*2]
        traj_feat = self.prev_traj_emb(rel_pos)  # [b, n_hid]

        # encode the ratserized image
        if rasterized_image is not None:
            rasterized_image = rasterized_image.permute(0, 3, 1, 2)  # [b, 3, 500, 500]
            image_feat = self.image_encoder(rasterized_image)  # [b, 4096]
            image_feat = self.image_emb(image_feat)
            fused_feat = self.fusion(torch.cat([image_feat, traj_feat], dim=1))
        else:
            fused_feat = traj_feat
        rel = self.pred_fc(fused_feat)
        pos = torch.cumsum(rel.view(batch_size, -1, 2), dim=1).view(batch_size, -1) + current_loc
        return rel, pos


class NMPEncoder(nn.Module):
    """docstring for NMPEncoder"""

    def __init__(self, use_nmp=False, N_actors=32, N_moving_actors=32, prev_steps=6, pred_steps=30, n_hid=256,
                 do_prob=0.5, cnn='alex'):
        super(NMPEncoder, self).__init__()
        self.use_nmp = use_nmp
        self.N_actors = N_actors
        self.N_moving_actors = N_moving_actors
        self.prev_steps = prev_steps
        self.pred_steps = pred_steps

        # ------ predict the actors trajctories ------- #
        # share raster image feature extractor
        if cnn == 'alex':
            self.image_encoder = AlexNet()
            self.V_encoder = AlexEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)
            self.H_encoder = AlexEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)
            self.M_encoder = AlexEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)
        else:
            self.image_encoder = MobileNetV2()
            self.V_encoder = CNNEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)
            self.H_encoder = CNNEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)
            self.M_encoder = CNNEncoder(prev_steps, pred_steps, n_hid, self.image_encoder)

        if self.use_nmp:
            # ------ NMP -------- #
            if cnn == 'alex':
                self.image_emb = nn.Linear(4096, n_hid)
            else:
                self.image_emb = nn.Linear(1280, n_hid)

            # --- only propagate history trajectory --- #
            self.prev_traj_emb = nn.Linear(prev_steps * 2, n_hid)
            # --- propagate his + baseline prediction --- #
            # self.prev_traj_emb = nn.Linear((prev_steps+pred_steps)*2, n_hid)

            # ----- nmp ----- #
            self.mlp_n2e = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            self.mlp_e2n = MLP(n_hid * 2, n_hid, n_hid, do_prob)

            self.fusion = nn.Linear(n_hid * 2, n_hid)
            # self.fusion = nn.Linear(n_hid, n_hid)
            self.pred_fc = nn.Linear(n_hid, pred_steps * 2)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    # def edge2node(self, x, rel_rec, rel_send):
    #     # NOTE: Assumes that we have the same graph across all samples.
    #     incoming = torch.matmul(rel_rec.t(), x)
    #     return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        new_rec_rec = rel_rec.permute(0, 2, 1)
        weight_rec = torch.sum(new_rec_rec, -1).float()
        weight_rec = weight_rec + (weight_rec == 0).float()
        weight_rec = torch.unsqueeze(weight_rec, -1).expand(weight_rec.size(0), weight_rec.size(1), x.size(-1))
        incoming = torch.matmul(new_rec_rec, x)
        incoming = incoming / weight_rec

        new_rec_send = rel_send.permute(0, 2, 1)
        weight_send = torch.sum(new_rec_send, -1).float()
        weight_send = weight_send + (weight_send == 0).float()
        weight_send = torch.unsqueeze(weight_send, -1).expand(weight_send.size(0), weight_send.size(1), x.size(-1))
        outgoing = torch.matmul(new_rec_send, x)
        outgoing = outgoing / weight_send

        nodes = torch.cat([incoming, outgoing], -1)
        return nodes

    def box_index_2_ego_index(self, box_index, box_translation, box_rotation, ego_translation, ego_rotation, steps):
        '''
        Function: translate the box index in current box coordinates to the ego coordinates
        box_index: [b, N_moving_actors, pred_steps, 2]
        box_translation: [b, N_moving_actors, 2]
        box_rotation: [b, N_moving_actors, 2, 2]
        ego_translation: [b, 2]
        ego_rotation: [b, 2, 2]
        '''
        batch_size = ego_translation.shape[0]
        box_translation = box_translation.view(batch_size, self.N_moving_actors, 2, 1)  # [b, N_moving_actors, 2, 1]
        box_translation = torch.unsqueeze(box_translation, 2).expand(batch_size, self.N_moving_actors, steps, 2,
                                                                     1)  # [b, N_moving_actors, pred_steps, 2, 1]
        box_rotation = box_rotation.view(batch_size, self.N_moving_actors, 1, 2, 2)  # [b, N_moving_actors, 1, 2, 2]

        # index to coordinates
        box_coord = box_index.view(batch_size, self.N_moving_actors, steps, 2)
        box_coord = box_coord.view(-1, 2)  # [b*N_moving_actors*pred_steps, 2]
        box_coord = (249.0 - box_coord) / 5.0
        box_coord = box_coord.view(batch_size, self.N_moving_actors, steps, 2,
                                   1)  # [b, N_moving_actors, pred_setps, 2, 1]

        # box coord to global coord
        box_coord = torch.matmul(box_rotation, box_coord)  # [b, N_moving_actors, pred_steps, 2, 1]
        box_coord += box_translation

        # global coord to ego coord
        box_coord += ego_translation
        box_coord = torch.matmul(ego_rotation, box_coord)

        # coordinates to index
        ego_box_index = 249.0 - box_coord[:, :, :, :2, 0] * 5.0  # [b, N_moving_actors, pred_steps, 2]
        ego_box_index = ego_box_index.view(batch_size, self.N_moving_actors, steps * 2)
        return ego_box_index

    def forward(self, rasterized_image, prev_actor_centers, ego_translation, ego_rotation, rel_rec, rel_send, v_input,
                h_input, m_input):
        '''
        rasterized_image : [b, 500, 500, 3]
        prev_actor_centers: [b, N_actors, prev_steps, 2]
        rel_rec: [b, N_edges, N_actors]
        rel_send: [b, N_edges, N_actors]
        v/h/m_rasterized_image: [b, N_moving_actors, 500, 500, 3]
        v/h/m_prev_ego_centers: [b, N_moving_actors, prev_steps, 2]
        v/h/m_translation: [b, N_moving_actors, 3]
        v/h/m_rotation: [b, N_moving_actors, 3, 3]
        v/h/m_moving_index: [b, N_actors, N_moving_actors]
        '''
        batch_size = prev_actor_centers.shape[0]

        v_rasterized_image, v_prev_ego_centers, v_translation, v_rotation, v_moving_index = v_input
        h_rasterized_image, h_prev_ego_centers, h_translation, h_rotation, h_moving_index = h_input
        m_rasterized_image, m_prev_ego_centers, m_translation, m_rotation, m_moving_index = m_input

        # ------ pred single actors traj ---- #
        v_pred_traj_rel, v_pred_traj = self.V_encoder(None, v_prev_ego_centers.view(-1, self.prev_steps,
                                                                                                  2))  # [b, N_moving_actors, pred_steps*2]
        h_pred_traj_rel, h_pred_traj = self.H_encoder(None,
                                                      h_prev_ego_centers.view(-1, self.prev_steps, 2))
        m_pred_traj_rel, m_pred_traj = self.M_encoder(None,
                                                      m_prev_ego_centers.view(-1, self.prev_steps, 2))

        # ------ shift and rotate the predications to av-centered coord ------- #
        ego_translation = ego_translation.view(batch_size, 2, 1)
        ego_translation = torch.unsqueeze(ego_translation, 1).expand(batch_size, self.N_moving_actors, 2, 1)
        ego_translation = torch.unsqueeze(ego_translation, 2).expand(batch_size, self.N_moving_actors, self.pred_steps,
                                                                     2, 1)
        ego_rotation = ego_rotation.view(batch_size, 1, 2, 2)
        ego_rotation = torch.unsqueeze(ego_rotation, 1).expand(batch_size, self.N_moving_actors, 1, 2,
                                                               2)  # [b, N_moving_actors, 1, 2, 2]

        v_ego_traj = self.box_index_2_ego_index(v_pred_traj, v_translation, v_rotation, ego_translation, ego_rotation,
                                                self.pred_steps)  # [b, N_moving_actors, pred_steps*2]
        h_ego_traj = self.box_index_2_ego_index(h_pred_traj, h_translation, h_rotation, ego_translation, ego_rotation,
                                                self.pred_steps)
        m_ego_traj = self.box_index_2_ego_index(m_pred_traj, m_translation, m_rotation, ego_translation, ego_rotation,
                                                self.pred_steps)

        # ------- re-organize the predicted moving actors ------------ #
        '''
        Pick out the actors used in the message passing module
        '''
        # [b, N_actors, pred_steps*2]
        count_v_ego_traj = torch.matmul(v_moving_index, v_ego_traj)  # [b, N_actors, pred_steps*2]
        count_h_ego_traj = torch.matmul(h_moving_index, h_ego_traj)  # [b, N_actors, pred_steps*2]
        count_m_ego_traj = torch.matmul(m_moving_index, m_ego_traj)  # [b, N_actors, pred_steps*2]

        count_ego_traj = count_v_ego_traj + count_h_ego_traj + count_m_ego_traj

        current_loc = prev_actor_centers.view(batch_size, self.N_actors, self.prev_steps, 2)[:, :,
                      -1]  # [b, N_actors, 2]
        # current_loc = torch.unsqueeze(current_loc, -2).expand(batch_size, self.N_actors, self.pred_steps,2).contiguous()
        # current_loc = current_loc.view(batch_size, self.N_actors, -1)

        if self.use_nmp:
            # ------- predict displacement based on NMP ------- #
            # encode the ratserized image
            rasterized_image = rasterized_image.permute(0, 3, 1, 2)  # [b, 3, 500, 500]
            image_feat = self.image_encoder(rasterized_image)  # [b, 1280]
            image_feat = self.image_emb(image_feat)  # [b, n_hid]
            image_feat = torch.unsqueeze(image_feat, 1).expand(image_feat.shape[0], self.N_actors,
                                                               image_feat.shape[1])  # [n, N_actors, n_hid]

            # embed the previous trajectories
            # prev_actor_centers = prev_actor_centers.view(batch_size, self.N_actors, -1)    # [b, N_actors, prev_steps*2]
            # only propagate the history
            prev_actor_centers_rel = torch.zeros(prev_actor_centers.shape).cuda()
            prev_actor_centers_rel[:, :, 1:, ] = prev_actor_centers[:, :, 1:, :] - prev_actor_centers[:, :, :-1, :]
            prev_actor_centers_rel = prev_actor_centers_rel.view(batch_size, self.N_actors, -1)

            traj_feat = self.prev_traj_emb(prev_actor_centers_rel)  # [b, N_actors, n_hid]
            # propagate the his + pred
            # pred_actor_centers = torch.cat([prev_actor_centers, count_ego_traj], dim=-1) # [b, N_actors, (prev_steps+prev_steps)*2]
            # traj_feat = self.prev_traj_emb(pred_actor_centers)

            # Neural Message Passing
            edge_feat = self.node2edge(traj_feat, rel_rec, rel_send)  # [b, N_edges, n_hid*2]
            edge_feat = self.mlp_n2e(edge_feat)  # [b, N_edges, n_hid]
            node_feat = self.edge2node(edge_feat, rel_rec, rel_send)  # [b, N_actors, n_hid*2]
            node_feat = self.mlp_e2n(node_feat)  # [b, N_actors, n_hid]
            fused_feat = self.fusion(torch.cat([image_feat, node_feat], dim=-1))

            # fused_feat = self.fusion(torch.cat([image_feat, traj_feat], dim=-1))

            c_rel = self.pred_fc(fused_feat)
            final_pos = F.relu(c_rel + count_ego_traj)
            final_rel = final_pos - torch.cat([current_loc, final_pos[:, :, :-2]], dim=-1)
        else:
            final_pos = count_ego_traj
            final_rel = final_pos - torch.cat([current_loc, final_pos[:, :, :-2]], dim=-1)
        return final_rel, v_pred_traj_rel, h_pred_traj_rel, m_pred_traj_rel, final_pos, v_pred_traj, h_pred_traj, m_pred_traj

    def coord_trans_check_forward(self, rasterized_image, prev_actor_centers, ego_translation, ego_rotation, rel_rec,
                                  rel_send, v_input, h_input, m_input):
        '''
        Function: check the coordinate transformation implement

        rasterized_image : [b, 500, 500, 3]
        prev_actor_centers: [b, N_actors, prev_steps, 2]
        rel_rec: [b, N_edges, N_actors]
        rel_send: [b, N_edges, N_actors]
        v/h/m_rasterized_image: [b, N_moving_actors, 500, 500, 3]
        v/h/m_prev_ego_centers: [b, N_moving_actors, prev_steps, 2]
        v/h/m_translation: [b, N_moving_actors, 3]
        v/h/m_rotation: [b, N_moving_actors, 3, 3]
        v/h/m_moving_index: [b, N_actors, N_moving_actors]
        '''
        batch_size = prev_actor_centers.shape[0]

        v_rasterized_image, v_prev_ego_centers, v_translation, v_rotation, v_moving_index = v_input
        h_rasterized_image, h_prev_ego_centers, h_translation, h_rotation, h_moving_index = h_input
        m_rasterized_image, m_prev_ego_centers, m_translation, m_rotation, m_moving_index = m_input

        # ------ shift and rotate the predications to av-centered coord ------- #
        ego_translation = ego_translation.view(batch_size, 2, 1)
        ego_translation = torch.unsqueeze(ego_translation, 1).expand(batch_size, self.N_moving_actors, 2, 1)
        ego_translation = torch.unsqueeze(ego_translation, 2).expand(batch_size, self.N_moving_actors, self.prev_steps,
                                                                     2, 1)
        ego_rotation = ego_rotation.view(batch_size, 1, 2, 2)
        ego_rotation = torch.unsqueeze(ego_rotation, 1).expand(batch_size, self.N_moving_actors, 1, 2,
                                                               2)  # [b, N_moving_actors, 1, 2, 2]

        v_ego_traj = self.box_index_2_ego_index(v_prev_ego_centers, v_translation, v_rotation, ego_translation,
                                                ego_rotation,
                                                steps=self.prev_steps)  # [b, N_moving_actors, prev_steps*2]
        h_ego_traj = self.box_index_2_ego_index(h_prev_ego_centers, h_translation, h_rotation, ego_translation,
                                                ego_rotation, steps=self.prev_steps)
        m_ego_traj = self.box_index_2_ego_index(m_prev_ego_centers, m_translation, m_rotation, ego_translation,
                                                ego_rotation, steps=self.prev_steps)

        # ------- re-organize the predicted moving actors ------------ #
        '''
        Pick out the actors used in the message passing module
        '''
        # [b, N_actors, pred_steps*2]
        count_v_ego_traj = torch.matmul(v_moving_index, v_ego_traj)  # [b, N_actors, pred_steps*2]
        count_h_ego_traj = torch.matmul(h_moving_index, h_ego_traj)  # [b, N_actors, pred_steps*2]
        count_m_ego_traj = torch.matmul(m_moving_index, m_ego_traj)  # [b, N_actors, pred_steps*2]

        count_ego_traj = count_v_ego_traj + count_h_ego_traj + count_m_ego_traj

        output = count_ego_traj

        return output, v_ego_traj, h_ego_traj, m_ego_traj