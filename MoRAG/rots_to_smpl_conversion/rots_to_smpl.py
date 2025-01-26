import os
import sys
import numpy as np
import json
import pickle
import geometry as geometry
import torch
from common.skeleton import Skeleton
from common.quaternion import *

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from plot_utils import (
    recover_from_ric,
    plot_3d_motion,
    t2m_kinematic_chain 
)

from scipy.ndimage import gaussian_filter


os.environ['PYOPENGL_PLATFORM'] = 'egl'
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
faces = c2c(male_bm.f)

trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
ex_fps = 20

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# (joints_num, 3)

l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions, feet_thre, tgt_offsets):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)


    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data, global_positions, positions, l_velocity

def rotmat_to_rot6d(x):
    return x[...,:2]

def rots_to_h3d(rots_axisangle,trans):
    # bdata = np.load(src_path, allow_pickle=True)
    fps = 30
#     try:
#         fps = bdata['mocap_framerate']
#         frame_number = bdata['trans'].shape[0]
#     except:
# #         print(list(bdata.keys()))
#         return fps
    
    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    bm = male_bm
    down_sample = int(fps / ex_fps)
    bdata_poses = rots_axisangle[::down_sample,...]
    bdata_trans = trans[::down_sample,...]

    body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),
            'betas': None,
        }
    
    with torch.no_grad():
        body = bm(**body_parms)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    pose_seq_np_n[:,:,0]*=-1    
    # pose_seq_np_n = pose_seq_np_n.reshape([trans.shape[0],-1])
    # return pose_seq_np_n
    # np.save(save_path, pose_seq_np_n)
    # return fps

    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pose_seq_np_n[0]))
    data, ground_positions, positions, l_velocity = process_file(pose_seq_np_n[:,:22,:], 0.002,tgt_offsets)
    return data

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def plot_t2m(data, result_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=2.5) #(T,22,3) => 3d coordinates of motion sequence
    # print("joint.shape")
    # print(joint.shape)
    # np.save("3d_joints.npy",joint)
    mot = joint.transpose((1,2,0))
    mot = np.expand_dims(mot, axis=0)
    lengths = np.array([mot.shape[-1]])
    d = {}
    d['motion']=mot
    d['lengths']=lengths
    d['num_samples']=1
    d['num_repetitions']=1
    d['text']=[text]
    np.save("rots_to_smpl_conversion/motion-diffusion-model/results.npy",d)        #Saving this dict d to generate smpl parameters using mdm visualization code.
    # sys.exit()
    print("Generating smpl params")
    os.chdir("rots_to_smpl_conversion/motion-diffusion-model")
    os.system("python -m visualize.render_mesh --input_path sample00_rep00.mp4") #Install numpy==1.23 to work this line in the same script


def convert_rots_to_h3d(rots,trans):
    rots_axisangle = geometry.matrix_to_axis_angle(rots).reshape([trans.shape[0],-1]) #(T,156) => (T,52*3)
    h3d_format = rots_to_h3d(rots_axisangle,trans) #(T,263)
    return h3d_format

if __name__ == "__main__":
    text = "text"
    trans = pickle.load(open("morag_retrieved/trans.pkl","rb"))[0] #(T,3)
    rots = pickle.load(open("morag_retrieved/rots.pkl","rb"))[0] #(T,52,3,3)
    rots_axisangle = geometry.matrix_to_axis_angle(rots).reshape([trans.shape[0],-1]) #(T,156) => (T,52*3)
    h3d_format = rots_to_h3d(rots_axisangle,trans) #(T,263)
    # print("h3d_format.shape")
    # print(h3d_format.shape)
    # np.save("h3d_format.npy",h3d_format)
    plot_t2m(h3d_format, f"outputs/{text}.gif", text)        
