from PIL import Image as pilimg
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model 

import glob
import os
import tqdm
import imageio
import argparse
import numpy as np
import pickle
import toolz
import smplx
import torch
import pyrender
import trimesh
import cv2
from star_model import STAR
import plyfile

def parse_arguments():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--scale", type=float, default=0.2, help='Downscaling factor for the rendered output.')
    parser.add_argument("--betas", type=int, default=10, help='NUmber of shape coefficients.')
    parser.add_argument("--smplx_root", type=str, required=True, 
        help='The root path of SMPL where the SMPLX and VPoser models are located.'
    )
    return parser.parse_args()

__KEY_MAP__ = {
    'translation': 'camera_translation',
    'body.betas': 'betas',
    'body.rotation': 'global_orient',
    'body.left_hand': 'left_hand_pose',
    'body.right_hand': 'right_hand_pose',
    'body.jaw': 'jaw_pose',
    'body.pose': '',
    'body.expression': 'expression',
    'body.pose': 'body_pose',
}# dict_keys(['camera_rotation', 'camera_translation', 'betas', 'global_orient', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'body_pose'])

def _create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

if __name__ == '__main__':
    args = parse_arguments()
    model_params = dict(model_path=os.path.join(args.smplx_root, 'models_smplx_v1_1\models'),
        create_global_orient=True,
        create_body_pose=False,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=torch.float32,
        gender='neutral',
        float_dtype='float32',
        model_type='smplx',
        num_pca_comps=12,
        use_pca=False,
        num_betas=args.betas,
    )
    model = smplx.create(**model_params)    
    star = STAR(
        model_file=os.path.join(args.smplx_root, 'star_1_1\\neutral\\model.npz'), 
        num_betas=args.betas
    )

    pose_embedding = torch.zeros([1, 32], dtype=torch.float32, requires_grad=False).numpy()
    vposer_ckpt = os.path.join(args.smplx_root, 'V02_05\V02_05')
    vposer, _ = load_model(vposer_ckpt, model_code=VPoser,
        remove_words_in_model_weights='vp_model.', disable_grad=True
    )
    vposer.eval()

    light_nodes = _create_raymond_lights()
    
    body_pose = vposer.decode(torch.from_numpy(pose_embedding)).get('pose_body').reshape(1, -1)
    est_params = {}
    est_params['betas'] = model.betas
    est_params['body_pose'] = body_pose
    est_params['camera_rotation'] = torch.eye(3)[np.newaxis, ...]                
    est_params['leye_pose'] = torch.zeros(1, 3)
    est_params['reye_pose'] = torch.zeros(1, 3)
    est_params['camera_translation'] = torch.zeros(1, 3)

    model_output = model(**est_params)
    
    pose = torch.cat([
        model_output.global_orient, model_output.body_pose, 
        model_output.left_hand_pose[:, :3], model_output.right_hand_pose[:, :3],
    ], dim=1)
    star_model = star(pose=pose, betas=model_output.betas, 
        # trans=est_params['camera_translation']
        trans=torch.zeros_like(est_params['camera_translation'])
    )

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    smplx_joints = model_output.joints.squeeze()
    vertex_data = np.empty(len(smplx_joints), dtype=dtype)
    vertex_data["x"] = smplx_joints[:, 0].detach().cpu().numpy()
    vertex_data["y"] = smplx_joints[:, 1].detach().cpu().numpy()
    vertex_data["z"] = smplx_joints[:, 2].detach().cpu().numpy()
    plydata = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
    ], text=True)
    plydata.write('smplx_joints.ply')
    
    smplx_joints = model_output.vertices.squeeze()
    vertex_data = np.empty(len(smplx_joints), dtype=dtype)
    vertex_data["x"] = smplx_joints[:, 0].detach().cpu().numpy()
    vertex_data["y"] = smplx_joints[:, 1].detach().cpu().numpy()
    vertex_data["z"] = smplx_joints[:, 2].detach().cpu().numpy()
    plydata = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
    ], text=True)
    plydata.write('smplx_mesh.ply')

    indices = torch.from_numpy(np.array([ # from star to openpose
        15, 12, 17, 19, 21, 16, 
        18, 20, 0, 2, 5, 8,
        1, 4, 7,  23, 23, 23,
        23, 10, 23, 23, 11, 23
    ], dtype=np.int32))
    # ignore: [0, 23, 22, 20, 19, 17, 16, 15, 14, ]
    star_joints = star.v_template.squeeze()
    vertex_data = np.empty(len(star_joints), dtype=dtype)
    vertex_data["x"] = star_joints[:, 0].detach().cpu().numpy()
    vertex_data["y"] = star_joints[:, 1].detach().cpu().numpy()
    vertex_data["z"] = star_joints[:, 2].detach().cpu().numpy()
    plydata = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
    ], text=True)
    plydata.write('star_mesh_template.ply')

    star_joints = star_model['vertices'].squeeze()
    vertex_data = np.empty(len(star_joints), dtype=dtype)
    vertex_data["x"] = star_joints[:, 0].detach().cpu().numpy()
    vertex_data["y"] = star_joints[:, 1].detach().cpu().numpy()
    vertex_data["z"] = star_joints[:, 2].detach().cpu().numpy()
    plydata = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
    ], text=True)
    plydata.write('star_mesh_posed.ply')

    star_joints = star_model['joints'].squeeze()
    star_joints = torch.index_select(star_joints, dim=0, index=indices)
    vertex_data = np.empty(len(star_joints), dtype=dtype)
    vertex_data["x"] = star_joints[:, 0].detach().cpu().numpy()
    vertex_data["y"] = star_joints[:, 1].detach().cpu().numpy()
    vertex_data["z"] = star_joints[:, 2].detach().cpu().numpy()
    plydata = plyfile.PlyData([
        plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
    ], text=True)
    plydata.write('star_joints.ply')

    vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    
    org_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    out_mesh = trimesh.Trimesh(
        star_model['vertices'].squeeze().detach().cpu().numpy(),
        star_model['faces'].squeeze().detach().cpu().numpy(),
        process=False
    )