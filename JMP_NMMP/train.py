import time
import argparse
import os
import ipdb
import pickle
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler

import time

from modules import CNNEncoder, NMPEncoder, AlexEncoder
from dataloader import load_dataset
from visualization import Visualizer, AverageMeter, ProgressMeter
from utils import displacement_error, compute_loss, rel_mat, final_displacement_error

# from parallel import DataParallelModel, DataParallelCriterion

# ===================== Args Definition ===================== #
parser = argparse.ArgumentParser()

# -------  model arguments ------------ #
parser.add_argument('--prev-steps', type=int, default=6,
                    help='Num steps before prediction')
parser.add_argument('--pred-steps', type=int, default=30,
                    help='Num steps to predict')
parser.add_argument('--n-hid', type=int, default=256,
                    help='Hidden dimension')
parser.add_argument('--do-prob', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--encoder', type=str, default='nmp',
                    help='Encoder base/nmp ')
parser.add_argument('--N-actors', type=int, default=64,
                    help='The amount of actors in the scene')
parser.add_argument('--N-moving-actors', type=int, default=64,
                    help='The amount of moving actors in each \
                    type of the scene')
parser.add_argument('--cnn', type=str, default='alex',
                    help='alex/mobile')

# ------- training arguments ---------- #
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=5,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='Shuffle the training data')
parser.add_argument('--dataset', type=str, default='full',
                    help='Full/Ego/vehicle/movable/human')
parser.add_argument('--subsample', type=int, default=5,
                    help='Subsample rate.')
parser.add_argument('--use-nmp', action='store_true', default=False,
                    help='Whether use message passing.')

# -------- log arguments -------------- #
parser.add_argument('--mode', type=str, default='whole',
                    help='whole, train, val, eval')
parser.add_argument('--restore', action='store_true', default=False,
                    help='Restore the trained model from the load-folder.')
parser.add_argument('--save-folder', type=str, default='./logs',
                    help='Where to save the trained model.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model.')
parser.add_argument('--tail', type=str, default='',
                    help='specific name')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Visualize the loss curve in visdom windows')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.visualize:
    vis = Visualizer(env='{}_{}_{}'.format(args.dataset, args.encoder, args.tail))

# ================== Model & Log Save Folder =================== #
log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    if args.restore:
        pass
    else:
        exp_counter = 0
        save_folder = os.path.join(args.save_folder,
                                   '{}_{}_{}_exp{}'.format(args.dataset, args.encoder, args.tail, exp_counter))
        while os.path.isdir(save_folder):
            exp_counter += 1
            save_folder = os.path.join(args.save_folder,
                                       '{}_{}_{}_exp{}'.format(args.dataset, args.encoder, args.tail, exp_counter))
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        model_file = os.path.join(save_folder, 'temp.pt')
        best_model_file = os.path.join(save_folder, 'encoder.pt')
        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
        print("save_folder: {}".format(save_folder))
else:
    print("Save_folder: {}".format(save_folder))

if args.load_folder:
    load_folder = os.path.join('./logs', '{}_{}_{}_{}'.format(args.dataset, args.encoder, args.tail, args.load_folder))
    meta_file = os.path.join(load_folder, 'metadata.pkl')
    model_file = os.path.join(load_folder, 'temp.pt')
    best_model_file = os.path.join(load_folder, 'encoder.pt')
    log_file = os.path.join(load_folder, 'log_new.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
    if args.restore:
        save_folder = load_folder
else:
    load_folder = save_folder
    print("Load_folder: {}".format(load_folder))

# ======================= Load Data ========================== #
train_loader, val_loader, test_loader = load_dataset(dataset=args.dataset,
                                                     subsample=args.subsample,
                                                     batch_size=args.batch_size,
                                                     shuffle=args.shuffle,
                                                     N_actors=args.N_actors,
                                                     N_moving_actors=args.N_moving_actors)

# ======================= Build Model ========================== #
if args.encoder == 'base':
    model = CNNEncoder(prev_steps=args.prev_steps, pred_steps=args.pred_steps, n_hid=args.n_hid)
elif args.encoder == 'alex':
    model = AlexEncoder(prev_steps=args.prev_steps, pred_steps=args.pred_steps, n_hid=args.n_hid)
elif args.encoder == 'nmp':
    model = NMPEncoder(use_nmp=args.use_nmp, N_actors=args.N_actors, N_moving_actors=args.N_moving_actors,
                       prev_steps=args.prev_steps, pred_steps=args.pred_steps, n_hid=args.n_hid, do_prob=args.do_prob,
                       cnn=args.cnn)

mse_loss = nn.MSELoss(reduction='none')
if args.cuda:
    model = DataParallel(model)
    # model = DataParallelModel(model)  # 并行化model
    # mse_loss = DataParallelCriterion(mse_loss)  # 并行化损失函数
    model.cuda()

# --------------- Parameters Loader ------------------#
model_params = model.state_dict()
if args.restore:
    model.load_state_dict(torch.load(model_file))

# ------------------ Optimizer --------------------- #
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005, momentum=0, centered=False)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)


# =============== iterate one epoch =====================#
def iter_one_epoch(data_loader, epoch=0, batch_size=args.batch_size, is_training=True):
    loss_all = []
    pbar = tqdm(total=len(data_loader.dataset))
    if not is_training:
        pred_centers = np.zeros([len(data_loader.dataset), args.pred_steps, 2])
        target_centers = np.zeros([len(data_loader.dataset), args.pred_steps, 2])

    for batch_idx, (rasterized_image, prev_ego_centers, target_ego_centers) in enumerate(data_loader):
        if args.cuda:
            rasterized_image, prev_ego_centers, target_ego_centers = rasterized_image.cuda(), prev_ego_centers.cuda(), target_ego_centers.cuda()

        # ------ clean the prev grad ------ #
        if is_training:
            optimizer.zero_grad()

        # ------------ forward ------------ #
        output = model(rasterized_image, prev_ego_centers)  # [b, pred_steps*2]

        # ------------ loss --------------- #
        target = target_ego_centers.view(target_ego_centers.shape[0], -1)

        loss = compute_loss(mse_loss, output, target)

        # ---------- backward --------------#
        if is_training:
            loss.backward()
            optimizer.step()

        # ========= Accuracy ======== #
        loss_all.append(loss.item())
        if args.visualize and loss.item() < 5e2:
            vis.plot_many_stack({'{}_loss'.format(data_loader.dataset.mode): loss.item()})

        # --------- save prediction ------- #
        if not is_training:
            if (batch_idx + 1) * batch_size > len(data_loader.dataset):
                pred_centers[batch_idx * batch_size:] = output.view(-1, args.pred_steps, 2).data.cpu().numpy()
                target_centers[batch_idx * batch_size:] = target_ego_centers.view(-1, args.pred_steps,
                                                                                  2).data.cpu().numpy()
            else:
                pred_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = output.view(-1, args.pred_steps,
                                                                                                2).data.cpu().numpy()
                target_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target_ego_centers.view(-1,
                                                                                                              args.pred_steps,
                                                                                                              2).data.cpu().numpy()
        if batch_idx % 10 == 0:
            print('[{}/{}] {:04d} | {}_loss: {:.04f}'.format(epoch, args.epochs, batch_idx, data_loader.dataset.mode,
                                                             loss.item()))
        pbar.update(batch_size)
    pbar.close()
    if not is_training:
        np.save(os.path.join(save_folder, '{}_pred_centers.npy'.format(data_loader.dataset.mode)), pred_centers)
        np.save(os.path.join(save_folder, '{}_target_centers.npy'.format(data_loader.dataset.mode)), target_centers)
    return np.mean(loss_all)


def nmp_iter_one_epoch(data_loader, epoch=0, batch_size=args.batch_size, is_training=True):
    count_time = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_avg = AverageMeter('loss', ':.04f')
    v_loss_avg = AverageMeter('v_loss', ':.04f')
    h_loss_avg = AverageMeter('h_loss', ':.04f')
    m_loss_avg = AverageMeter('m_loss', ':.04f')

    nmp_loss_avg = AverageMeter('nmp_loss', ':.04f')
    progress = ProgressMeter(len(data_loader.dataset), loss_avg, v_loss_avg,
                             h_loss_avg, m_loss_avg, nmp_loss_avg, prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # pbar = tqdm(total = len(data_loader.dataset))
    end = time.time()
    if not is_training:
        pred_centers = np.zeros([len(data_loader.dataset), args.N_actors, args.pred_steps, 2])
        target_centers = np.zeros([len(data_loader.dataset), args.N_actors, args.pred_steps, 2])
        moving_masks = np.zeros([len(data_loader.dataset), args.N_actors])
        v_moving_masks = np.zeros([len(data_loader.dataset), args.N_actors])
        h_moving_masks = np.zeros([len(data_loader.dataset), args.N_actors])
        m_moving_masks = np.zeros([len(data_loader.dataset), args.N_actors])

        v_pred_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])
        v_target_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])
        h_pred_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])
        h_target_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])
        m_pred_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])
        m_target_centers = np.zeros([len(data_loader.dataset), args.N_moving_actors, args.pred_steps, 2])

    for batch_idx, (
    rasterized_image, prev_actor_centers, target_actor_centers, ego_translation, ego_rotation, rel_rec, rel_send, \
    moving_mask, v_moving_mask, h_moving_mask, m_moving_mask, \
    v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index, \
    h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index, \
    m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            prev_actor_centers, target_actor_centers = prev_actor_centers.cuda(), target_actor_centers.cuda()
            ego_translation, ego_rotation = ego_translation.cuda(), ego_rotation.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            moving_mask = moving_mask.cuda()
            v_prev_ego_centers, v_target_ego_centers, v_translation, v_rotation, v_moving_index = \
                v_prev_ego_centers.cuda(), v_target_ego_centers.cuda(), v_translation.cuda(), v_rotation.cuda(), v_moving_index.cuda()
            v_input = (None, v_prev_ego_centers, v_translation, v_rotation, v_moving_index)
            h_prev_ego_centers, h_target_ego_centers, h_translation, h_rotation, h_moving_index = \
                h_prev_ego_centers.cuda(), h_target_ego_centers.cuda(), h_translation.cuda(), h_rotation.cuda(), h_moving_index.cuda()
            h_input = (None, h_prev_ego_centers, h_translation, h_rotation, h_moving_index)
            m_prev_ego_centers, m_target_ego_centers, m_translation, m_rotation, m_moving_index = \
                m_prev_ego_centers.cuda(), m_target_ego_centers.cuda(), m_translation.cuda(), m_rotation.cuda(), m_moving_index.cuda()
            m_input = (None, m_prev_ego_centers, m_translation, m_rotation, m_moving_index)
        # ------ clean the prev grad ------ #
        if is_training:
            optimizer.zero_grad()
            # ------------ forward ------------ #
            output_rel, v_pred_traj_rel, h_pred_traj_rel, m_pred_traj_rel, output, v_pred_traj, h_pred_traj, m_pred_traj = model(
                rasterized_image, prev_actor_centers, \
                ego_translation, ego_rotation, rel_rec, rel_send, v_input, h_input, m_input)  # [b, *, pred_steps*2]
        else:
            # ------------ forward ------------ #
            with torch.no_grad():
                start = time.time()
                output_rel, v_pred_traj_rel, h_pred_traj_rel, m_pred_traj_rel, output, v_pred_traj, h_pred_traj, m_pred_traj = model(
                    rasterized_image, prev_actor_centers, \
                    ego_translation, ego_rotation, rel_rec, rel_send, v_input, h_input, m_input)  # [b, *, pred_steps*2]
                count_time.append(time.time()-start)
        # ------------ base model loss -------------- #
        v_pred_traj = v_pred_traj.view(-1, v_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        v_target_ego_centers = v_target_ego_centers.view(-1, v_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        v_loss = compute_loss(mse_loss, v_pred_traj, v_target_ego_centers)
        # v_pred_traj_rel = v_pred_traj_rel.view(-1, v_pred_traj_rel.shape[-1])
        # v_target_ego_centers_rel = v_target_ego_centers - \
        #                            torch.cat([v_prev_ego_centers.view(-1, v_prev_ego_centers.shape[-2] * 2)[:, :2],
        #                                       v_target_ego_centers[:, :-2]], dim=-1)
        # v_loss = compute_loss(mse_loss, v_pred_traj_rel, v_target_ego_centers_rel)

        h_pred_traj = h_pred_traj.view(-1, h_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        h_target_ego_centers = h_target_ego_centers.view(-1, h_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        h_loss = compute_loss(mse_loss, h_pred_traj, h_target_ego_centers)
        # h_pred_traj_rel = h_pred_traj_rel.view(-1, h_pred_traj_rel.shape[-1])
        # h_target_ego_centers_rel = h_target_ego_centers - \
        #                            torch.cat([h_prev_ego_centers.view(-1, h_prev_ego_centers.shape[-2] * 2)[:, :2],
        #                                       h_target_ego_centers[:, :-2]], dim=-1)
        # h_loss = compute_loss(mse_loss, h_pred_traj_rel, h_target_ego_centers_rel)

        m_pred_traj = m_pred_traj.view(-1, m_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        m_target_ego_centers = m_target_ego_centers.view(-1, m_pred_traj.shape[-1])  # [b*N_moving_actors, pred_steps*2]
        m_loss = compute_loss(mse_loss, m_pred_traj, m_target_ego_centers)
        # m_pred_traj_rel = m_pred_traj_rel.view(-1, m_pred_traj_rel.shape[-1])
        # m_target_ego_centers_rel = m_target_ego_centers - \
        #                            torch.cat([m_prev_ego_centers.view(-1, m_prev_ego_centers.shape[-2] * 2)[:, :2],
        #                                       m_target_ego_centers[:, :-2]], dim=-1)
        # m_loss = compute_loss(mse_loss, m_pred_traj_rel, m_target_ego_centers_rel)

        # ------------ nmp loss --------------- #
        output = output.view(-1, output.shape[-1])  # [b*N_actors, pred_steps*2]
        target = target_actor_centers.view(-1, output.shape[-1])  # [b*N_actors, pred_steps*2]
        moving_mask = moving_mask.view(-1, 1)  # [b*N_actors, 1]
        nmp_loss = compute_loss(mse_loss, output, target, moving_mask)
        # output_rel = output_rel.view(-1, output_rel.shape[-1])
        # target_rel = target - torch.cat(
        #     [prev_actor_centers.view(-1, prev_actor_centers.shape[-2] * 2)[:, :2], target[:, :-2]], dim=-1)
        # nmp_loss = compute_loss(mse_loss, output_rel, target_rel)

        # ---------- Final Loss -------- #
        if args.use_nmp:
            # loss = v_loss + h_loss + m_loss + nmp_loss
            loss = v_loss + h_loss + nmp_loss
        else:
            # loss = v_loss + h_loss + m_loss
            loss = v_loss + h_loss
        # ---------- backward --------------#
        if is_training:
            loss.backward()
            optimizer.step()

        # ========= Accuracy ======== #
        v_loss_avg.update(v_loss.item(), v_pred_traj.shape[0])
        h_loss_avg.update(h_loss.item(), h_pred_traj.shape[0])
        m_loss_avg.update(m_loss.item(), m_pred_traj.shape[0])
        nmp_loss_avg.update(nmp_loss.item(), output.shape[0])
        if args.use_nmp:
            loss_avg.update(loss.item(), output.shape[0])
        else:
            loss_avg.update(loss.item(), v_pred_traj.shape[0])

        # --------- save prediction ------- #
        if not is_training:
            if (batch_idx + 1) * batch_size > len(data_loader.dataset):
                pred_centers[batch_idx * batch_size:] = output.view(-1, args.N_actors, args.pred_steps,
                                                                    2).data.cpu().numpy()
                target_centers[batch_idx * batch_size:] = target_actor_centers.view(-1, args.N_actors, args.pred_steps,
                                                                                    2).data.cpu().numpy()
                moving_masks[batch_idx * batch_size:] = moving_mask.view(-1, args.N_actors).data.cpu().numpy()

                v_moving_masks[batch_idx * batch_size:] = v_moving_mask.view(-1, args.N_actors).data.cpu().numpy()
                h_moving_masks[batch_idx * batch_size:] = h_moving_mask.view(-1, args.N_actors).data.cpu().numpy()
                m_moving_masks[batch_idx * batch_size:] = m_moving_mask.view(-1, args.N_actors).data.cpu().numpy()

                v_pred_centers[batch_idx * batch_size:] = v_pred_traj.view(-1, args.N_moving_actors, args.pred_steps,
                                                                           2).data.cpu().numpy()
                v_target_centers[batch_idx * batch_size:] = v_target_ego_centers.view(-1, args.N_moving_actors,
                                                                                      args.pred_steps,
                                                                                      2).data.cpu().numpy()
                h_pred_centers[batch_idx * batch_size:] = h_pred_traj.view(-1, args.N_moving_actors, args.pred_steps,
                                                                           2).data.cpu().numpy()
                h_target_centers[batch_idx * batch_size:] = h_target_ego_centers.view(-1, args.N_moving_actors,
                                                                                      args.pred_steps,
                                                                                      2).data.cpu().numpy()
                m_pred_centers[batch_idx * batch_size:] = m_pred_traj.view(-1, args.N_moving_actors, args.pred_steps,
                                                                           2).data.cpu().numpy()
                m_target_centers[batch_idx * batch_size:] = m_target_ego_centers.view(-1, args.N_moving_actors,
                                                                                      args.pred_steps,
                                                                                      2).data.cpu().numpy()

            else:
                pred_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = output.view(-1, args.N_actors,
                                                                                                args.pred_steps,
                                                                                                2).data.cpu().numpy()
                target_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target_actor_centers.view(-1,
                                                                                                                args.N_actors,
                                                                                                                args.pred_steps,
                                                                                                                2).data.cpu().numpy()
                moving_masks[batch_idx * batch_size:(batch_idx + 1) * batch_size] = moving_mask.view(-1,
                                                                                                     args.N_actors).data.cpu().numpy()

                v_moving_masks[batch_idx * batch_size:(batch_idx + 1) * batch_size] = v_moving_mask.view(-1,
                                                                                                         args.N_actors).data.cpu().numpy()
                h_moving_masks[batch_idx * batch_size:(batch_idx + 1) * batch_size] = h_moving_mask.view(-1,
                                                                                                         args.N_actors).data.cpu().numpy()
                m_moving_masks[batch_idx * batch_size:(batch_idx + 1) * batch_size] = m_moving_mask.view(-1,
                                                                                                         args.N_actors).data.cpu().numpy()

                v_pred_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = v_pred_traj.view(-1,
                                                                                                       args.N_moving_actors,
                                                                                                       args.pred_steps,
                                                                                                       2).data.cpu().numpy()
                v_target_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = v_target_ego_centers.view(-1,
                                                                                                                  args.N_moving_actors,
                                                                                                                  args.pred_steps,
                                                                                                                  2).data.cpu().numpy()
                h_pred_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = h_pred_traj.view(-1,
                                                                                                       args.N_moving_actors,
                                                                                                       args.pred_steps,
                                                                                                       2).data.cpu().numpy()
                h_target_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = h_target_ego_centers.view(-1,
                                                                                                                  args.N_moving_actors,
                                                                                                                  args.pred_steps,
                                                                                                                  2).data.cpu().numpy()
                m_pred_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = m_pred_traj.view(-1,
                                                                                                       args.N_moving_actors,
                                                                                                       args.pred_steps,
                                                                                                       2).data.cpu().numpy()
                m_target_centers[batch_idx * batch_size:(batch_idx + 1) * batch_size] = m_target_ego_centers.view(-1,
                                                                                                                  args.N_moving_actors,
                                                                                                                  args.pred_steps,
                                                                                                                  2).data.cpu().numpy()

        # ------- visualization ------- #
        if args.visualize:
            vis.plot_many_stack({'{}_v_loss'.format(data_loader.dataset.mode): v_loss_avg.avg,
                                 '{}_h_loss'.format(data_loader.dataset.mode): h_loss_avg.avg,
                                 '{}_m_loss'.format(data_loader.dataset.mode): m_loss_avg.avg,
                                 '{}_nmp_loss'.format(data_loader.dataset.mode): nmp_loss_avg.avg})
            vis.plot_many_stack({'{}_loss'.format(data_loader.dataset.mode): loss_avg.avg})

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            progress.print(batch_idx)
            # print('[{}/{}] {:04d} | {}_loss: {:.04f}'.format(epoch, args.epochs, batch_idx, data_loader.dataset.mode, loss.item()))
    #     pbar.update(batch_size)
    # pbar.close()

    if not is_training:
        np.save(os.path.join(save_folder, '{}_pred_centers.npy'.format(data_loader.dataset.mode)), pred_centers)
        np.save(os.path.join(save_folder, '{}_target_centers.npy'.format(data_loader.dataset.mode)), target_centers)
        np.save(os.path.join(save_folder, '{}_moving_masks.npy'.format(data_loader.dataset.mode)), moving_masks)
        np.save(os.path.join(save_folder, '{}_v_moving_masks.npy'.format(data_loader.dataset.mode)), v_moving_masks)
        np.save(os.path.join(save_folder, '{}_h_moving_masks.npy'.format(data_loader.dataset.mode)), h_moving_masks)
        np.save(os.path.join(save_folder, '{}_m_moving_masks.npy'.format(data_loader.dataset.mode)), m_moving_masks)
        np.save(os.path.join(save_folder, '{}_v_pred_centers.npy'.format(data_loader.dataset.mode)), v_pred_centers)
        np.save(os.path.join(save_folder, '{}_v_target_centers.npy'.format(data_loader.dataset.mode)), v_target_centers)
        np.save(os.path.join(save_folder, '{}_h_pred_centers.npy'.format(data_loader.dataset.mode)), h_pred_centers)
        np.save(os.path.join(save_folder, '{}_h_target_centers.npy'.format(data_loader.dataset.mode)), h_target_centers)
        np.save(os.path.join(save_folder, '{}_m_pred_centers.npy'.format(data_loader.dataset.mode)), m_pred_centers)
        np.save(os.path.join(save_folder, '{}_m_target_centers.npy'.format(data_loader.dataset.mode)), m_target_centers)

    print('Time cost: {:.08f}s'.format(sum(count_time)/631.0))
    return loss_avg.avg


# ======================== Training Operation ======================= #
def train(epoch, min_val_loss):
    t = time.time()
    loss_train = 0.0
    loss_val = 0.0
    v_displacement_error = 0.0
    h_displacement_error = 0.0
    m_displacement_error = 0.0

    if args.encoder == 'nmp':
        iter_fn = nmp_iter_one_epoch
    else:
        iter_fn = iter_one_epoch

    model.train()
    scheduler.step()

    loss_train = iter_fn(train_loader, epoch, is_training=True)
    model.eval()
    loss_val = iter_fn(val_loader, epoch, is_training=False)
    model.eval()
    loss_test = iter_fn(test_loader, epoch, is_training=False)

    if args.encoder == 'nmp':
        v_pred_centers = np.load(os.path.join(save_folder, 'test_v_pred_centers.npy'))
        v_target_centers = np.load(os.path.join(save_folder, 'test_v_target_centers.npy'))
        h_pred_centers = np.load(os.path.join(save_folder, 'test_h_pred_centers.npy'))
        h_target_centers = np.load(os.path.join(save_folder, 'test_h_target_centers.npy'))
        m_pred_centers = np.load(os.path.join(save_folder, 'test_m_pred_centers.npy'))
        m_target_centers = np.load(os.path.join(save_folder, 'test_m_target_centers.npy'))

        v_displacement_error = displacement_error(v_pred_centers, v_target_centers)
        h_displacement_error = displacement_error(h_pred_centers, h_target_centers)
        m_displacement_error = displacement_error(m_pred_centers, m_target_centers)

        v_f_displacement_error = final_displacement_error(v_pred_centers, v_target_centers)
        h_f_displacement_error = final_displacement_error(h_pred_centers, h_target_centers)
        m_f_displacement_error = final_displacement_error(m_pred_centers, m_target_centers)

        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(v_displacement_error, v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(h_displacement_error, h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(m_displacement_error, m_f_displacement_error))
        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(v_displacement_error, v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(h_displacement_error, h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(m_displacement_error, m_f_displacement_error), file=log)

        test_pred_centers = np.load(os.path.join(save_folder, 'test_pred_centers.npy'))
        test_target_centers = np.load(os.path.join(save_folder, 'test_target_centers.npy'))
        test_moving_masks = np.load(os.path.join(save_folder, 'test_moving_masks.npy'))
        test_v_moving_masks = np.load(os.path.join(save_folder, 'test_v_moving_masks.npy'))
        test_h_moving_masks = np.load(os.path.join(save_folder, 'test_h_moving_masks.npy'))
        test_m_moving_masks = np.load(os.path.join(save_folder, 'test_m_moving_masks.npy'))
        nmp_v_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_v_moving_masks)
        nmp_h_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_h_moving_masks)
        nmp_m_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_m_moving_masks)

        nmp_v_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_v_moving_masks)
        nmp_h_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_h_moving_masks)
        nmp_m_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_m_moving_masks)

        print('Epoch: {:04d}'.format(epoch),
              'Loss_Train: {:.04f}'.format(loss_train),
              'Loss_Val: {:.04f}'.format(loss_val),
              'Loss_Test: {:.04f}'.format(loss_test),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(nmp_v_displacement_error, nmp_v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(nmp_h_displacement_error, nmp_h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(nmp_m_displacement_error, nmp_m_f_displacement_error),
              'time: {:.04f}'.format(time.time() - t))
        print('Epoch: {:04d}'.format(epoch),
              'Loss_Train: {:.04f}'.format(loss_train),
              'Loss_Val: {:.04f}'.format(loss_val),
              'Loss_Test: {:.04f}'.format(loss_test),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(nmp_v_displacement_error, nmp_v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(nmp_h_displacement_error, nmp_h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(nmp_m_displacement_error, nmp_m_f_displacement_error),
              'time: {:.04f}'.format(time.time() - t), file=log)
    else:
        test_pred_centers = np.load(os.path.join(save_folder, 'test_pred_centers.npy'))
        test_target_centers = np.load(os.path.join(save_folder, 'test_target_centers.npy'))
        test_moving_masks = np.load(os.path.join(save_folder, 'test_moving_masks.npy'))
        avg_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_moving_masks)

        print('Epoch: {:04d}'.format(epoch),
              'Loss_Eval: {:.04f}'.format(loss_test),
              'Displace_Error: {:.04f}'.format(avg_displacement_error),
              'time: {:.04f}'.format(time.time() - t))
        print('Epoch: {:04d}'.format(epoch),
              'Loss_Eval: {:.04f}'.format(loss_test),
              'Displace_Error: {:.04f}'.format(avg_displacement_error),
              'time: {:.04f}'.format(time.time() - t), file=log)
    log.flush()
    torch.save(model.state_dict(), model_file)

    if args.save_folder and loss_val < min_val_loss:
        torch.save(model.state_dict(), best_model_file)
    return loss_val


def eval(epoch, data_loader=test_loader):
    t = time.time()
    loss_eval = 0.0
    model.eval()
    if args.mode == 'eval':
        model.load_state_dict(torch.load(best_model_file))
    else:
        model.load_state_dict(torch.load(model_file))

    if args.encoder == 'nmp':
        loss_eval = nmp_iter_one_epoch(data_loader, epoch=epoch, is_training=False)

        v_pred_centers = np.load(os.path.join(save_folder, 'test_v_pred_centers.npy'))
        v_target_centers = np.load(os.path.join(save_folder, 'test_v_target_centers.npy'))
        h_pred_centers = np.load(os.path.join(save_folder, 'test_h_pred_centers.npy'))
        h_target_centers = np.load(os.path.join(save_folder, 'test_h_target_centers.npy'))
        m_pred_centers = np.load(os.path.join(save_folder, 'test_m_pred_centers.npy'))
        m_target_centers = np.load(os.path.join(save_folder, 'test_m_target_centers.npy'))

        v_displacement_error = displacement_error(v_pred_centers, v_target_centers)
        h_displacement_error = displacement_error(h_pred_centers, h_target_centers)
        m_displacement_error = displacement_error(m_pred_centers, m_target_centers)

        v_f_displacement_error = final_displacement_error(v_pred_centers, v_target_centers)
        h_f_displacement_error = final_displacement_error(h_pred_centers, h_target_centers)
        m_f_displacement_error = final_displacement_error(m_pred_centers, m_target_centers)

        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(v_displacement_error, v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(h_displacement_error, h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(m_displacement_error, m_f_displacement_error))
        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(v_displacement_error, v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(h_displacement_error, h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(m_displacement_error, m_f_displacement_error), file=log)

        test_pred_centers = np.load(os.path.join(save_folder, 'test_pred_centers.npy'))
        test_target_centers = np.load(os.path.join(save_folder, 'test_target_centers.npy'))
        test_moving_masks = np.load(os.path.join(save_folder, 'test_moving_masks.npy'))
        test_v_moving_masks = np.load(os.path.join(save_folder, 'test_v_moving_masks.npy'))
        test_h_moving_masks = np.load(os.path.join(save_folder, 'test_h_moving_masks.npy'))
        test_m_moving_masks = np.load(os.path.join(save_folder, 'test_m_moving_masks.npy'))
        nmp_v_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_v_moving_masks)
        nmp_h_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_h_moving_masks)
        nmp_m_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_m_moving_masks)

        nmp_v_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_v_moving_masks)
        nmp_h_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_h_moving_masks)
        nmp_m_f_displacement_error = final_displacement_error(test_pred_centers, test_target_centers, test_m_moving_masks)

        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(nmp_v_displacement_error, nmp_v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(nmp_h_displacement_error, nmp_h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(nmp_m_displacement_error, nmp_m_f_displacement_error),
              'time: {:.04f}'.format(time.time() - t))
        print('Epoch: {:04d}'.format(epoch),
              'V_Dis_Error: {:.04f}/{:.04f}'.format(nmp_v_displacement_error, nmp_v_f_displacement_error),
              'H_Dis_Error: {:.04f}/{:.04f}'.format(nmp_h_displacement_error, nmp_h_f_displacement_error),
              'M_Dis_Error: {:.04f}/{:.04f}'.format(nmp_m_displacement_error, nmp_m_f_displacement_error),
              'time: {:.04f}'.format(time.time() - t), file=log)
    else:
        loss_eval = iter_one_epoch(data_loader, epoch=epoch, is_training=False)
        test_pred_centers = np.load(os.path.join(save_folder, 'test_pred_centers.npy'))
        test_target_centers = np.load(os.path.join(save_folder, 'test_target_centers.npy'))
        test_moving_masks = np.load(os.path.join(save_folder, 'test_moving_masks.npy'))
        avg_displacement_error = displacement_error(test_pred_centers, test_target_centers, test_moving_masks)
        # avg_displacement_error = displacement_error(test_pred_centers, test_target_centers)

        print('Epoch: {:04d}'.format(epoch),
              'Loss_Eval: {:.04f}'.format(loss_eval),
              'Displace_Error: {:.04f}'.format(avg_displacement_error),
              'time: {:.04f}'.format(time.time() - t))
        print('Epoch: {:04d}'.format(epoch),
              'Loss_Eval: {:.04f}'.format(loss_eval),
              'Displace_Error: {:.04f}'.format(avg_displacement_error),
              'time: {:.04f}'.format(time.time() - t), file=log)
    return


if __name__ == '__main__':
    t_total = time.time()
    if args.mode == 'whole' or args.mode == 'train':
        min_val_loss = 200
        best_epoch = 0
        pbar = tqdm(total=args.epochs)

        for epoch in range(args.epochs):
            print('================= Epoch {} ==================='.format(epoch))
            val_loss = train(epoch, min_val_loss)
            if epoch == 0:
                min_val_loss = val_loss
                best_epoch = epoch
                # print('======= Eval ========')
                # eval(epoch, test_loader)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                # print('======= Eval ========')
                # eval(epoch, test_loader)
            pbar.update(1)
        pbar.close()

        print('========== Optimization Finshied! ==========')
        print('Besh Epoch :{:04d}'.format(best_epoch))
        print('======= Eval ========')
        args.mode = 'eval'
        eval(best_epoch, test_loader)
        if args.save_folder:
            print('Besh Epoch :{:04d}'.format(best_epoch), file=log)
            log.flush()
        if log is not None:
            print(save_folder)
            log.close()
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    elif args.mode == 'eval':
        print('======= Eval ========')
        eval(0, test_loader)
        if log is not None:
            print(load_folder)
            log.close()
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
