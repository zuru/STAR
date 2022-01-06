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
    parser.add_argument('--pkl_glob', type=str, required=True,        
        help='The pkl folder that contains the files that will be read.')
    parser.add_argument('--img_glob', type=str, required=True,
        help='The folder that contains the images that will be read.')
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
    'translation_t': 'camera_translation',
    'betas': 'betas',
    'betas_t': 'betas',
    'global_orient': 'global_orient',
    'global_orient_t': 'global_orient',
    'lhand_t': 'left_hand_pose',
    'rhand_t': 'right_hand_pose',
    'jaw_t': 'jaw_pose',
    'expression_t': 'expression',
    'decoded.pose': 'body_pose',
    'star_left_hand': 'left_hand_pose',
    'star_right_hand': 'right_hand_pose',
    'decoded.pose': 'body_pose',
    'star.joints': 'joints',
    'joints2d': 'joints2d',
    'keypoints': 'keypoints2d',
    'pose_t': 'pose_t',
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

    pose_embedding = torch.zeros([1, 32], dtype=torch.float32, requires_grad=False)
    vposer_ckpt = os.path.join(args.smplx_root, 'V02_05\V02_05')
    vposer, _ = load_model(vposer_ckpt, model_code=VPoser,
        remove_words_in_model_weights='vp_model.', disable_grad=True
    )
    vposer.eval()

    light_nodes = _create_raymond_lights()
    pkl_filenames = glob.glob(args.pkl_glob)
    img_filenames = glob.glob(args.img_glob)
    for i, (pkl_filename, img_filename) in tqdm.tqdm(
        enumerate(zip(pkl_filenames, img_filenames)), 
        desc='Results', total=len(pkl_filenames)
    ):
        iters = []
        with open(pkl_filename, 'rb') as f:
            f.seek(0, 2)
            eof = f.tell()
            f.seek(0, 0)
            while f.tell() != eof:
                data = pickle.load(f)
                iters.append(toolz.merge(*data.values()))
        
        img = cv2.imread(img_filename)
        H, W, C = img.shape
        if args.scale != 1.0:
            ratio = H / W
            W = int(W * args.scale)
            H = int(W * ratio)
            img = np.array(pilimg.fromarray(img).resize((W, H)))
        gif_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.mp4'
        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        for node in light_nodes:
            scene.add_node(node)
        with imageio.get_writer(gif_filename, mode='I', fps=25) as writer:
            # iters = toolz.drop(1700, iters)
            for iter in tqdm.tqdm(iters, desc=f"({os.path.basename(img_filename)}) - {i}: iterations"):
                iter = toolz.keymap(lambda k: k.replace('params.', ''), iter)
                pose_embedding = iter.get('embedding') or iter.get('pose_t')
                iteration = iter.pop('iteration')
                stage = iter.pop('stage')
                body_pose = vposer.decode(torch.from_numpy(pose_embedding)).get('pose_body').reshape(1, -1)
                est_params = {}
                for key, val in iter.items(): # dict_keys(['camera_rotation', 'camera_translation', 'betas', 'global_orient', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'body_pose'])
                    est_params[key] = torch.from_numpy(val)
                est_params = toolz.keymap(lambda k: __KEY_MAP__[k], est_params)
                est_params['body_pose'] = body_pose
                est_params['camera_rotation'] = torch.eye(3)[np.newaxis, ...]                
                est_params['leye_pose'] = torch.zeros(1, 3)
                est_params['reye_pose'] = torch.zeros(1, 3)

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
                # smplx_joints = model.v_template.squeeze()
                vertex_data = np.empty(len(smplx_joints), dtype=dtype)
                vertex_data["x"] = smplx_joints[:, 0].numpy()
                vertex_data["y"] = smplx_joints[:, 1].numpy()
                vertex_data["z"] = smplx_joints[:, 2].numpy()
                plydata = plyfile.PlyData([
                    plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
                ], text=True)
                plydata.write('smplx.ply')
                # plydata.write('smplx_mesh.ply')

                star_joints = star_model['joints'].squeeze()
                indices = torch.from_numpy(np.array([ # from star to openpose
                    15, 12, 17, 19, 21, 16, 
                    18, 20, 0, 2, 5, 8,
                    1, 4, 7,  23, 23, 23,
                    23, 10, 23, 23, 11, 23
                ], dtype=np.int32))
                # ignore: [0, 23, 22, 20, 19, 17, 16, 15, 14, ]
                # star_joints = star.v_template.squeeze()
                star_joints = star_model['vertices'].squeeze()
                # star_joints = torch.index_select(star_joints, dim=0, index=indices)
                vertex_data = np.empty(len(star_joints), dtype=dtype)
                vertex_data["x"] = star_joints[:, 0]
                vertex_data["y"] = star_joints[:, 1]
                vertex_data["z"] = star_joints[:, 2]
                plydata = plyfile.PlyData([
                    plyfile.PlyElement.describe(vertex_data, 'vertex'),                    
                ], text=True)
                # plydata.write('star.ply')
                plydata.write('star_mesh.ply')

                vertices = model_output.vertices.detach().cpu().numpy().squeeze()
                
                org_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
                out_mesh = trimesh.Trimesh(
                    star_model['vertices'].squeeze().detach().cpu().numpy(),
                    star_model['faces'].squeeze().detach().cpu().numpy(),
                    process=False
                )
                # out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
                material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                    alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0)
                )
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                out_mesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
                mesh_node = scene.add(mesh, 'mesh')

                org_material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                    alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.0, 1.0)
                )
                org_mesh.apply_transform(rot)
                org_mesh = pyrender.Mesh.from_trimesh(org_mesh, material=org_material)
                org_mesh_node = scene.add(org_mesh, 'org_mesh')

                camera_transl = est_params['camera_translation'].detach().cpu().numpy().squeeze()
                camera_transl[0] *= -1.0
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = camera_transl
                focal_length = 5000 * args.scale
                camera_center = torch.tensor([W, H], dtype=torch.float32) * 0.5
                camera_center = camera_center.detach().cpu().numpy().squeeze()
                camera = pyrender.camera.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                    cx=camera_center[0], cy=camera_center[1]
                )
                camera_node = scene.add(camera, pose=camera_pose)

                # color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
                # color = color.astype(np.float32) / 255.0

                # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
                # input_img = img.copy()
                # point = (int(W * 0.04), int(H * 0.04))
                # text = f"{stage}: {iteration}/{len(iters)}"
                # font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                # font_scale = 1.25
                # thickness = 2
                # size = cv2.getTextSize(text, font, font_scale, thickness)
                # while size[0][0] > 0.9 * W:
                #     font_scale *= 0.9
                #     thickness *= 0.9
                #     size = cv2.getTextSize(text, font, font_scale, int(thickness if thickness >= 1 else 1))
                # cv2.putText(input_img, text, point, font, font_scale, (30, 70, 240), int(thickness if thickness >= 1 else 1))
                # input_img = input_img / 255.0
                # output_img = (color[:, :, :-1] * valid_mask +
                #             (1 - valid_mask) * input_img)

                # writer.append_data((output_img[:,:,::-1] * 255).astype(np.uint8))
                pyrender.Viewer(scene, use_raymond_lighting=True)
                scene.remove_node(mesh_node)
                scene.remove_node(camera_node)