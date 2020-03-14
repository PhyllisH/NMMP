import sys
import os
import numpy as np
import json
import ipdb
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torch.utils.data.distributed
import time
import cv2
import numpy as np
from skimage import draw
from tqdm import tqdm
import copy
from utils import dis, encode_onehot, load_json
from pyquaternion import Quaternion

class SdvNuscenesDataset(Dataset):
    """docstring for SdvNuscenesDataset"""
    def __init__(self, dataset='ego', mode='train', N_actors=32, subsample=1, dis_thre=50.0, N_moving_actors=None):
        self.root_path = '/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/v0.1'
        self.mode = mode
        self.N_actors = N_actors
        self.N_edges = 200
        self.actors = False
        self.dataset = dataset
        self.dis_thre = dis_thre
        samples, train_index, val_index, test_index = self.load_samples(dataset, subsample)
        self.train = samples[train_index]
        self.val = samples[val_index]
        self.test = samples[test_index]
        if self.mode == 'train':
            self.samples = self.train
        elif self.mode == 'val':
            self.samples = self.val
        else:
            self.samples = self.test
        # min_edge, max_edge = self.get_edge_max()
        return

    def load_samples(self, dataset, subsample=1):
        if dataset == 'ori-ego':
            final_json_path = os.path.join(self.root_path, 'Sdv_Nuscenes_Samples.json')
        elif dataset == 'ego':
            final_json_path = os.path.join(self.root_path, 'fulldataset', 'ego', 'Ego_Samples.json')
        elif dataset == 'human':
            final_json_path = os.path.join(self.root_path, 'fulldataset', 'human', 'Human_Samples.json')
        elif dataset == 'vehicle':
            final_json_path = os.path.join(self.root_path, 'fulldataset', 'vehicle', 'Vehicle_Samples.json')
        elif dataset == 'movable':
            final_json_path = os.path.join(self.root_path, 'fulldataset', 'movable_object', 'Movable_object_Samples.json')
        else:
            print('Unseen dataset: {}'.format(dataset))
        if os.path.exists(final_json_path):
            with open(final_json_path, 'r') as f:
                samples = json.load(f)
        samples = samples[::subsample]
        N_samples = len(samples)
        N_train = int(N_samples * 0.6)
        N_val = int(N_samples * 0.2)
        N_test = N_samples - N_train - N_val
        # index_path = os.path.join(self.root_path, 'fulldataset', final_json_path.split('/')[-2], '{}_index'.format(subsample))
        # if os.path.exists(index_path):
        #     with open(index_path, 'r') as f:
        #         index = json.load(f)
        # else:
        idx = list(range(len(samples)))
        # np.random.shuffle(idx)
        index = {}
        index['train'] = idx[:N_train]
        index['val'] = idx[N_train:(N_train+N_val)]
        index['test'] = idx[(N_train+N_val):]
        # with open(index_path, 'w') as f:
        #     json.dump(index, f)
        train_index = index['train']
        val_index = index['val']
        test_index = index['test']
        samples = np.array(samples)
        return samples, train_index, val_index, test_index

    def get_item(self, sample):
        rasterized_image = np.load(sample['rasterized_image'])    # [500, 500, 3]
        rasterized_image = rasterized_image/255.0

        ego_centers = np.load(sample['ego_centers'])    # [36, 5, 2]
        # current_ego_centers = ego_centers[sample['ego_id'], 0, :]
        # current_ego_centers = np.reshape(np.array(current_ego_centers, dtype=np.float32), [1, -1])    # [1, 2]
        prev_ego_centers = ego_centers[:(sample['ego_id']+1), 0, :]
        prev_ego_centers = np.array(prev_ego_centers, dtype=np.float32)         # [6, 2]
        target_ego_centers = ego_centers[(sample['ego_id']+1):, 0, :]
        target_ego_centers = np.array(target_ego_centers, dtype=np.float32)         # [30, 2]

        return rasterized_image, prev_ego_centers, target_ego_centers

    def get_actors(self, sample):
        boxes_centers = np.load(sample['boxes_index'])[:,:,0] # [timesteps, N_actors, 2]
        boxes_centers = np.transpose(boxes_centers, [1, 0, 2])    # [N_actors, timesteps, 2]
        # moving_actors = sample['moving_actors_index']
        # boxes_index = boxes_centers[moving_actors]
        boxes_index = boxes_centers
        actor_centers = np.zeros([self.N_actors, boxes_index.shape[1], 2]) - 1
        if boxes_index.shape[0] > self.N_actors:
            actor_centers = boxes_index[:self.N_actors]
        else:
            actor_centers[:boxes_index.shape[0]] = boxes_index
        return actor_centers

    def get_moving_mask(self, sample):
        moving_actors = [-1] + sample['moving_actors_index']        
        moving_mask = np.zeros([1, self.N_actors+1])
        for i in moving_actors:
            if i < self.N_actors:
                moving_mask[0,int(i+1)] = 1
        return moving_mask

    def get_graph(self, actor_centers, moving_actors):
        '''
        actor_centers:   [N_actors+1, 2]
        output: rel_rec  [N_edge, N_actors+1]
                rel_send [N_edge, N_actors+1]
        '''
        bin_dis_mat = dis(actor_centers, moving_actors, self.dis_thre)
        rel_rec = np.zeros([self.N_edges, (self.N_actors+1)])
        rel_send = np.zeros([self.N_edges, (self.N_actors+1)])
        exist_rel_rec = np.array(encode_onehot(np.where(bin_dis_mat)[1], (self.N_actors+1)), dtype=np.float32)
        exist_rel_send = np.array(encode_onehot(np.where(bin_dis_mat)[0], (self.N_actors+1)), dtype=np.float32)
        assert exist_rel_rec.shape[1] == (self.N_actors+1)
        assert exist_rel_send.shape[1] == (self.N_actors+1)

        if exist_rel_rec.shape[0] > self.N_edges:
            rel_rec = exist_rel_rec[:self.N_edges]
            rel_send = exist_rel_send[:self.N_edges]
        else:
            rel_rec[:exist_rel_rec.shape[0]] = exist_rel_rec
            rel_rec[:exist_rel_send.shape[0]] = exist_rel_send            
        return bin_dis_mat, rel_rec, rel_send

    def get_edge_max(self):
        '''
        Count the max edge amount
        '''
        min_edge = 1000
        max_edge = 0
        pbar = tqdm(total=len(self.samples))
        for i in range(len(self.samples)):
            sample = self.samples[i]
            moving_actors = sample['moving_actors_index']
            # ego_centers = np.load(sample['ego_centers'])[sample['ego_id'],0,:]
            # boxes_centers = np.load(sample['boxes_index'])[sample['ego_id'],:,0,:]
            # actor_centers = np.concatenate([np.expand_dims(ego_centers,0), boxes_centers], 0)
            # bin_dis_mat = dis(actor_centers, moving_actors, self.dis_thre)
            rasterized_image, prev_ego_centers, target_ego_centers = self.get_item(sample)
            actor_centers = self.get_actors(sample)
            prev_ego_centers = np.expand_dims(prev_ego_centers, 0)
            prev_actor_centers = np.concatenate([prev_ego_centers, actor_centers[:,:(sample['ego_id']+1)]], 0) # [N_actors+1, prev_steps, 2]
            bin_dis_mat, rel_rec, rel_send = self.get_graph(prev_actor_centers[:,-1,:], sample['moving_actors_index'])
            np.save(os.path.join(self.root_path, 'fulldataset', 'ego', 'distance_mat', '{}_{}.npy'.format(sample['scene_id'], sample['sweep_id'])), bin_dis_mat)
            count_edge = bin_dis_mat.sum()
            if count_edge > max_edge:
                max_edge = count_edge
            if count_edge < min_edge:
                min_edge = count_edge
            pbar.update(1)
        pbar.close()
        return min_edge, max_edge


    def load_actors(self):
        self.actors = True
        return

    def __getitem__(self, index):
        sample = self.samples[index]
        rasterized_image, prev_ego_centers, target_ego_centers = self.get_item(sample)
        rasterized_image = torch.FloatTensor(rasterized_image)

        if self.actors:
            actor_centers = self.get_actors(sample)
            moving_mask = self.get_moving_mask(sample)
            prev_ego_centers = np.expand_dims(prev_ego_centers, 0)
            prev_actor_centers = np.concatenate([prev_ego_centers, actor_centers[:,:(sample['ego_id']+1)]], 0) # [N_actors+1, prev_steps, 2]
            bin_dis_mat, rel_rec, rel_send = self.get_graph(prev_actor_centers[:,-1,:], sample['moving_actors_index'])
            target_ego_centers = np.expand_dims(target_ego_centers, 0)
            target_actor_centers = np.concatenate([target_ego_centers, actor_centers[:,(sample['ego_id']+1):]], 0) # [N_actors+1, pred_steps, 2]
            
            prev_actor_centers = torch.FloatTensor(prev_actor_centers)
            target_actor_centers = torch.FloatTensor(target_actor_centers)

            rel_rec = torch.FloatTensor(rel_rec)
            rel_send = torch.FloatTensor(rel_send)

            moving_mask = torch.FloatTensor(moving_mask)

            return rasterized_image, prev_actor_centers, target_actor_centers, rel_rec, rel_send, moving_mask

        prev_ego_centers = torch.FloatTensor(prev_ego_centers)
        target_ego_centers = torch.FloatTensor(target_ego_centers)
        return rasterized_image, prev_ego_centers, target_ego_centers

    def __len__(self):
        return len(self.samples)


class NMPNuscenesDataset(Dataset):
    """docstring for NuscenesDataset"""
    def __init__(self, dataset='full', mode='train', N_actors=32, subsample=1, dis_thre=50.0, N_moving_actors=32):
        self.root_path = '/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/v0.1'
        self.mode = mode
        self.N_actors = N_actors
        self.N_moving_actors = N_moving_actors
        self.N_edges = 200
        self.actors = False
        self.dataset = dataset
        self.dis_thre = dis_thre
        samples, train_index, val_index, test_index = self.load_samples(subsample)
        print('train: {}'.format(len(train_index)))
        print('val: {}'.format(len(val_index)))
        print('test: {}'.format(len(test_index)))
        self.train = samples[train_index]
        self.val = samples[val_index]
        self.test = samples[test_index]
        if self.mode == 'train':
            self.samples = self.train
        elif self.mode == 'val':
            self.samples = self.val
        else:
            self.samples = self.test
        # min_edge, max_edge = self.get_edge_max()
        return

    def load_samples(self, subsample=1):
        final_json_path = os.path.join(self.root_path, 'fulldataset', 'nmp', 'NMP_Samples.json')
        samples = load_json(final_json_path)
        
        samples = samples[::subsample]
        N_samples = len(samples)
        N_train = int(N_samples * 0.6)
        N_val = int(N_samples * 0.2)
        N_test = N_samples - N_train - N_val
        index_path = os.path.join(self.root_path, 'fulldataset', final_json_path.split('/')[-2], '{}_index'.format(subsample))
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            idx = list(range(len(samples)))
            # np.random.shuffle(idx)
            index = {}
            index['train'] = idx[:N_train]
            index['val'] = idx[N_train:(N_train+N_val)]
            index['test'] = idx[(N_train+N_val):]
            with open(index_path, 'w') as f:
                json.dump(index, f)
        train_index = index['train']
        val_index = index['val']
        test_index = index['test']
        samples = np.array(samples)
        return samples, train_index, val_index, test_index

    def get_item(self, sample):
        ego_translation = np.reshape(-np.array(sample['translation']), [1, -1])[:,:2]
        ego_rotation = Quaternion(sample['rotation']).inverse.rotation_matrix
        ego_rotation = np.reshape(np.array(ego_rotation), [3, 3])[:2,:2]
        rasterized_image = np.load(sample['rasterized_image'])    # [500, 500, 3]
        rasterized_image = rasterized_image/255.0

        boxes_centers = np.load(sample['boxes_index'])[:,:,0]    # [36, num_actors, 2]
        boxes_centers = np.transpose(boxes_centers, [1, 0, 2])  # [num_actors, 36, 2]

        actor_centers = np.zeros([self.N_actors, boxes_centers.shape[1], 2]) - 1
        if boxes_centers.shape[0] > self.N_actors:
            actor_centers = boxes_centers[:self.N_actors]
        else:
            actor_centers[:boxes_centers.shape[0]] = boxes_centers
        
        prev_actor_centers = actor_centers[:, :(sample['ego_id']+1), :]
        prev_actor_centers = np.array(prev_actor_centers, dtype=np.float32)         # [6, num_actors, 2]
        target_actor_centers = actor_centers[:, (sample['ego_id']+1):, :]
        target_actor_centers = np.array(target_actor_centers, dtype=np.float32)         # [30, num_actors, 2]

        return rasterized_image, prev_actor_centers, target_actor_centers, ego_translation, ego_rotation

    def get_moving_mask(self, sample):
        # ---- get from the moving_actors_index ------ #
        # moving_actors = sample['moving_actors_index']
        # for i in moving_actors:
        #     if i < self.N_actors:
        #         moving_mask[0,int(i)] = 1
        # ----- only count the exist box_type ------ #
        def get_moving_mask_by_type(sample, box_type, N_actors):
            moving_mask = np.zeros([1, self.N_actors])
            moving_actors = sample[box_type]
            for i in range(len(moving_actors)):
                moving_actor = moving_actors[i]
                if moving_actor['ref_coord_id'] < N_actors:
                    moving_mask[0, int(moving_actor['ref_coord_id'])] = 1
            return moving_mask
        v_moving_mask = get_moving_mask_by_type(sample, 'vehicle', self.N_actors)
        h_moving_mask = get_moving_mask_by_type(sample, 'human', self.N_actors)
        m_moving_mask = get_moving_mask_by_type(sample, 'movable_object', self.N_actors)
        # moving_mask = v_moving_mask + h_moving_mask + m_moving_mask
        moving_mask = v_moving_mask + h_moving_mask
        return moving_mask, v_moving_mask, h_moving_mask, m_moving_mask

    def get_graph(self, actor_centers, moving_actors):
        '''
        actor_centers:   [N_actors, 2]
        output: rel_rec  [N_edge, N_actors]
                rel_send [N_edge, N_actors]
        '''
        bin_dis_mat = dis(actor_centers, moving_actors, self.dis_thre)
        rel_rec = np.zeros([self.N_edges, self.N_actors])
        rel_send = np.zeros([self.N_edges, self.N_actors])
        exist_rel_rec = np.array(encode_onehot(np.where(bin_dis_mat)[1], self.N_actors), dtype=np.float32)
        exist_rel_send = np.array(encode_onehot(np.where(bin_dis_mat)[0], self.N_actors), dtype=np.float32)
        assert exist_rel_rec.shape[1] == self.N_actors
        assert exist_rel_send.shape[1] == self.N_actors

        if exist_rel_rec.shape[0] > self.N_edges:
            rel_rec = exist_rel_rec[:self.N_edges]
            rel_send = exist_rel_send[:self.N_edges]
        else:
            rel_rec[:exist_rel_rec.shape[0]] = exist_rel_rec
            rel_rec[:exist_rel_send.shape[0]] = exist_rel_send            
        return bin_dis_mat, rel_rec, rel_send

    def get_moving_actors(self, sample, box_type='vehicle'):
        moving_actors = sample[box_type]
        N_moving_actors = len(moving_actors)

        # rasterized_image = np.zeros([self.N_moving_actors, 500, 500, 3])
        prev_ego_centers = np.zeros([self.N_moving_actors, 6, 2]) - 1
        target_ego_centers = np.zeros([self.N_moving_actors, 30, 2]) - 1
        translation = np.zeros([self.N_moving_actors, 2])
        rotation = np.zeros([self.N_moving_actors, 2, 2])
        moving_index = np.zeros([self.N_actors, self.N_moving_actors])

        N = min([N_moving_actors, self.N_moving_actors])
        for i in range(N):
            moving_actor = moving_actors[i]
            
            # rasterized_image[i] = np.load(moving_actor['rasterized_image'])
            
            ego_centers = np.load(moving_actor['ego_centers'])[:,0,:] # [36, 2]
            prev_ego_centers[i] = ego_centers[:(moving_actor['ego_id']+1)]
            target_ego_centers[i] = ego_centers[(moving_actor['ego_id']+1):]

            translation[i] = np.array(moving_actor['translation'])[:2]
            rotation[i] = np.reshape(np.array(Quaternion(moving_actor['rotation']).rotation_matrix), [3, 3])[:2,:2]
            if moving_actor['ref_coord_id'] < self.N_actors:
                moving_index[int(moving_actor['ref_coord_id']), i] = 1

        # print('rasterized_image: {}'.format(rasterized_image.shape))
        # print('prev_ego_centers: {}'.format(prev_ego_centers.shape))
        # print('target_ego_centers: {}'.format(target_ego_centers.shape))
        # print('translation: {}'.format(translation.shape))
        # print('rotation: {}'.format(rotation.shape))
        # print('moving_index: {}'.format(moving_index.shape))

        return None, prev_ego_centers, target_ego_centers, translation, rotation, moving_index

    def tensorize(self, rasterized_image, prev_ego_centers, target_ego_centers, translation, rotation, moving_index):
        # rasterized_image = torch.FloatTensor(rasterized_image)
        prev_ego_centers = torch.FloatTensor(prev_ego_centers)
        target_ego_centers = torch.FloatTensor(target_ego_centers)
        translation = torch.FloatTensor(translation)
        rotation = torch.FloatTensor(rotation)
        moving_index = torch.FloatTensor(moving_index)
        return None, prev_ego_centers, target_ego_centers, translation, rotation, moving_index

    def vis_rasterized_image(self, index, only_moving_actors=False):
        sample = self.samples[index]
        rasterized_image = np.load(sample['rasterized_image'])

        gt_traj = np.load(sample['boxes_index'])[:,:,0]    # [36, num_actors, 2]
        gt_traj = np.transpose(gt_traj, [1, 0, 2])  # [num_actors, 36, 2]
        
        moving_mask,_,_,_ = self.get_moving_mask(sample)

        if only_moving_actors:
            gt_traj = gt_traj[moving_mask]
        gt_traj = np.array(np.reshape(gt_traj, [-1, 2]), dtype=np.int32)

        # draw gt in blue color
        blue = np.array([0, 0, 255], dtype=np.int32)
        draw.set_color(rasterized_image, [gt_traj[:,0], gt_traj[:,1]], blue)        

        bgr_rasterized_image = rasterized_image[:,:,::-1]
        cv2.imwrite(os.path.join('/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/sample_visualization', 
            '{}_{}_ego.png'.format(sample['scene_id'], sample['sweep_id'])), bgr_rasterized_image)
        return

    def vis_raster_all_actors(self, index):
        '''
        Function: translate the box index in current box coordinates to the ego coordinates
        '''
        def get_moving_actors(sample, box_type, N_actors):
            moving_actors = sample[box_type]
            N_moving_actors = len(moving_actors)

            rasterized_image = np.zeros([len(moving_actors), 500, 500, 3])
            ego_centers = np.zeros([len(moving_actors), 36, 2]) - 1
            translation = np.zeros([len(moving_actors), 3])
            rotation = np.zeros([len(moving_actors), 3, 3])
            moving_index = np.zeros([N_actors, len(moving_actors)])

            for i in range(len(moving_actors)):
                moving_actor = moving_actors[i]
                rasterized_image[i] = np.load(moving_actor['rasterized_image'])
                
                ego_centers[i] = np.load(moving_actor['ego_centers'])[:,0,:] # [36, 2]

                translation[i] = np.array(moving_actor['translation'])
                rotation[i] = np.reshape(np.array(Quaternion(moving_actor['rotation']).rotation_matrix), [3, 3])
                if moving_actor['ref_coord_id'] < N_actors:
                    moving_index[int(moving_actor['ref_coord_id']), i] = 1

                gt_traj = np.array(np.reshape(ego_centers[i], [-1, 2]), dtype=np.int32)
                # draw gt in blue color
                blue = np.array([0, 0, 255], dtype=np.int32)
                draw.set_color(rasterized_image[i], [gt_traj[:,0], gt_traj[:,1]], blue)        

                bgr_rasterized_image = rasterized_image[i,:,:,::-1]
                cv2.imwrite(os.path.join('/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/sample_visualization', 
                    '{}_{}_{}_{}.png'.format(sample['scene_id'], sample['sweep_id'], box_type, moving_actor['ref_coord_id'])), bgr_rasterized_image)

            return ego_centers, translation, rotation, moving_index

        def box_index_2_ego_index(current_box_index, box_translation, box_rotation, ego_translation, ego_rotation):
            '''
            current_box_index: [N_box, 36, 2]
            '''
            N_box = current_box_index.shape[0]
            steps = current_box_index.shape[1]
            # ------ shift and rotate the predications to av-centered coord ------- #
            ego_translation = np.reshape(ego_translation, [3, 1])
            ego_translation = np.repeat(np.expand_dims(ego_translation, 0), steps, axis=0) # [steps, 3, 1]
            ego_translation = np.repeat(np.expand_dims(ego_translation, 0), N_box, axis=0) # [N_box, steps, 3, 1]
            ego_rotation = np.reshape(ego_rotation, [1, 3, 3])
            ego_rotation = np.repeat(np.expand_dims(ego_rotation, 0), N_box, axis=0) #[N_box, 1, 3, 3]

            box_translation = np.reshape(box_translation, [N_box, 3, 1])
            box_translation = np.repeat(np.expand_dims(box_translation, 1), steps, axis=1) # [N_box, steps, 3, 1]
            box_rotation = np.reshape(box_rotation, [N_box, 1, 3, 3])

            # index to coordinates
            box_coord = np.reshape(np.concatenate([current_box_index, np.zeros([N_box, steps, 1])], -1), [N_box, steps, 3, 1])    #[N_box, steps, 3, 1]
            box_coord = (249.0 - box_coord)/5.0
            
            # box coord to global coord
            box_coord = np.matmul(box_rotation, box_coord) # [N_box, steps, 2, 1]
            box_coord += box_translation
            
            # global coord to ego coord
            box_coord += ego_translation
            box_coord = np.matmul(ego_rotation, box_coord)

            box_coord = np.reshape(box_coord, [N_box, steps*3])

            # coordinates to index
            ego_box_index = 249.0 - box_coord*5.0 # [N_box, steps*3]
            return ego_box_index

        sample = self.samples[index]
        rasterized_image = np.load(sample['rasterized_image'])

        box_index = np.load(sample['boxes_index'])[:,:,0]    # [36, num_actors, 2]
        box_index = np.transpose(box_index, [1, 0, 2])  # [num_actors, 36, 2]

        N_actors = box_index.shape[0]
        steps = box_index.shape[1]

        ego_translation = np.reshape(-np.array(sample['translation']), [1, -1])
        ego_rotation = Quaternion(sample['rotation']).inverse.rotation_matrix
        ego_rotation = np.reshape(np.array(ego_rotation), [3, 3])

        v_centers, v_translation, v_rotation, v_moving_index = get_moving_actors(sample, 'vehicle', N_actors)
        h_centers, h_translation, h_rotation, h_moving_index = get_moving_actors(sample, 'human', N_actors)
        m_centers, m_translation, m_rotation, m_moving_index = get_moving_actors(sample, 'movable_object', N_actors)

        v_ego_centers = box_index_2_ego_index(v_centers, v_translation, v_rotation, ego_translation, ego_rotation)
        h_ego_centers = box_index_2_ego_index(h_centers, h_translation, h_rotation, ego_translation, ego_rotation)
        m_ego_centers = box_index_2_ego_index(m_centers, m_translation, m_rotation, ego_translation, ego_rotation)

        v_transform_ego = np.reshape(np.matmul(v_moving_index, v_ego_centers), [N_actors, steps, 3])[:,:,:2]
        h_transform_ego = np.reshape(np.matmul(h_moving_index, h_ego_centers), [N_actors, steps, 3])[:,:,:2]
        m_transform_ego = np.reshape(np.matmul(m_moving_index, m_ego_centers), [N_actors, steps, 3])[:,:,:2]
        transform_ego = v_transform_ego + h_transform_ego + m_transform_ego # [num_actors, 36, 2]

        # draw ego centers
        ori_rasterized_image = copy.deepcopy(rasterized_image)
        gt_traj = np.array(np.reshape(box_index, [-1, 2]), dtype=np.int32)
        # draw gt in blue color
        blue = np.array([0, 0, 255], dtype=np.int32)
        draw.set_color(ori_rasterized_image, [gt_traj[:,0], gt_traj[:,1]], blue)        

        bgr_rasterized_image = ori_rasterized_image[:,:,::-1]
        cv2.imwrite(os.path.join('/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/sample_visualization', 
            '{}_{}_ori_ego.png'.format(sample['scene_id'], sample['sweep_id'])), bgr_rasterized_image)

        # draw trans ego centers
        trans_rasterized_image = copy.deepcopy(rasterized_image)
        gt_traj = np.array(np.reshape(transform_ego, [-1, 2]), dtype=np.int32)
        # draw gt in blue color
        blue = np.array([0, 0, 255], dtype=np.int32)
        draw.set_color(trans_rasterized_image, [gt_traj[:,0], gt_traj[:,1]], blue)        

        bgr_rasterized_image = trans_rasterized_image[:,:,::-1]
        cv2.imwrite(os.path.join('/DATA5_DB8/data/yhu/Nuscenes/data/nuscenes/sample_visualization', 
            '{}_{}_trans_ego.png'.format(sample['scene_id'], sample['sweep_id'])), bgr_rasterized_image)
        return
        
    def __getitem__(self, index):

        sample = self.samples[index]

        # ---- NMP data ----- #
        rasterized_image, prev_actor_centers, target_actor_centers, ego_translation, ego_rotation = self.get_item(sample)
        bin_dis_mat, rel_rec, rel_send = self.get_graph(prev_actor_centers[:,-1,:], sample['moving_actors_index'])
        moving_mask, v_moving_mask, h_moving_mask, m_moving_mask = self.get_moving_mask(sample)
        
        rasterized_image = torch.FloatTensor(rasterized_image)
        prev_actor_centers = torch.FloatTensor(prev_actor_centers)
        target_actor_centers = torch.FloatTensor(target_actor_centers)
        ego_translation = torch.FloatTensor(ego_translation)
        ego_rotation = torch.FloatTensor(ego_rotation)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        moving_mask = torch.FloatTensor(moving_mask)
        v_moving_mask = torch.FloatTensor(v_moving_mask)
        h_moving_mask = torch.FloatTensor(h_moving_mask)
        m_moving_mask = torch.FloatTensor(m_moving_mask)

        # ---- Baseline data ---- #
        _, v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index = self.get_moving_actors(sample, 'vehicle')
        _, v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index = \
            self.tensorize(None, v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index)
        _, h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index = self.get_moving_actors(sample, 'human')
        _, h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index = \
            self.tensorize(None, h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index)
        _, m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index = self.get_moving_actors(sample, 'movable_object')
        _, m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index = \
            self.tensorize(None, m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index)


        return rasterized_image, prev_actor_centers, target_actor_centers, ego_translation, ego_rotation, rel_rec, rel_send, \
                 moving_mask, v_moving_mask, h_moving_mask, m_moving_mask, \
                v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index, \
                h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index, \
                m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index

    def __len__(self):
        return len(self.samples)


def load_dataset(dataset='ego', subsample=2, batch_size=32, shuffle=True, N_actors=32, N_moving_actors=32):
    if dataset == 'full':
        dataset_name = NMPNuscenesDataset
    else:
        dataset_name = SdvNuscenesDataset
    train_data = dataset_name(dataset=dataset, mode='train', N_actors=N_actors, subsample=subsample, N_moving_actors=N_moving_actors)
    val_data = dataset_name(dataset=dataset, mode='val', N_actors=N_actors, subsample=subsample, N_moving_actors=N_moving_actors)
    test_data = dataset_name(dataset=dataset, mode='test', N_actors=N_actors, subsample=subsample, N_moving_actors=N_moving_actors)

    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader