import torch
import numpy as np
import os 
try:
    import cPickle as pickle
except ImportError:
    import pickle

#NOTE: code from https://github.com/ahmedosman/STAR

__all__ = ['STAR']

class STAR(torch.nn.Module):
    def __init__(self,
        model_file: str='',
        num_betas:  int=10,
    ):
        super(STAR, self).__init__()
        #TODO: assert gender choices or unify with ckpt
        #TODO: assert file exists
        star_model = np.load(model_file, allow_pickle=True)
        J_regressor = star_model['J_regressor']
        self.num_betas = num_betas
        #NOTE: check if np arrays are doubles
        self.register_buffer('J_regressor', torch.Tensor(
            J_regressor
        ).float())
        self.register_buffer('weights', torch.Tensor(
            star_model['weights']).float()
        )
        self.register_buffer('posedirs', torch.Tensor(
            star_model['posedirs'].reshape((-1, 93))
        ).float())
        self.register_buffer('v_template', torch.Tensor(
            star_model['v_template']
        ).float())
        self.register_buffer('shapedirs', torch.from_numpy(
            np.array(star_model['shapedirs'][:,:,:num_betas])
        ).float())
        self.register_buffer('faces', torch.from_numpy(
            star_model['f'].astype(np.int64))
        )
        self.f = star_model['f'] #NOTE: is this needed?
        self.register_buffer('kintree_table', torch.from_numpy(
            star_model['kintree_table'].astype(np.int64))
        )
        id_to_col = {
            self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])
        }
        self.register_buffer('parent', torch.Tensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]
        ).long())

    def _quat_feat(self, theta: torch.Tensor) -> torch.Tensor:
        '''
            Computes a normalized quaternion ([0,0,0,0]  when the body is in rest pose)
            given joint angles
        :param theta: A tensor of joints axis angles, batch size x number of joints x 3
        :return:
        '''
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_sin * normalized, v_cos - 1.0], dim=1)
        return quat

    def _quat2mat(self, quat: torch.Tensor) -> torch.Tensor:
        '''
            Converts a quaternion to a rotation matrix
        :param quat:
        :return:
        '''
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
        B = quat.size(0)
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rotMat = torch.stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ], dim=1).view(B, 3, 3)
        return rotMat

    def _rodrigues(self, theta: torch.Tensor) -> torch.Tensor:
        '''
            Computes the rodrigues representation given joint angles

        :param theta: batch_size x number of joints x 3
        :return: batch_size x number of joints x 3 x 4
        '''
        l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
        return self._quat2mat(quat)

    def _with_zeros(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor

        :param input: A tensor of dimensions batch size x 3 x 4
        :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
        '''
        b = input.shape[0]
        row_append = torch.Tensor(([0.0, 0.0, 0.0, 1.0])).to(input)
        row_append.requires_grad = False
        padded_tensor = torch.cat([
            input, row_append.view(1, 1, 4).repeat(b, 1, 1)
        ], dim=1)
        return padded_tensor

    def forward(self, 
        pose:       torch.Tensor,
        betas:      torch.Tensor,
        trans:      torch.Tensor,
    ):
        '''
            STAR forward pass given pose, betas (shape) and trans
            return the model vertices and transformed joints
        :param pose: pose  parameters - A batch size x 72 tensor (3 numbers for each joint)
        :param beta: beta  parameters - A batch size x number of betas
        :param beta: trans parameters - A batch size x 3
        :return:
                 v         : batch size x 6890 x 3
                             The STAR model vertices
                 v.v_vposed: batch size x 6890 x 3 model
                             STAR vertices in T-pose after adding the shape
                             blend shapes and pose blend shapes
                 v.v_shaped: batch size x 6890 x 3
                             STAR vertices in T-pose after adding the shape
                             blend shapes and pose blend shapes
                 v.J_transformed:batch size x 24 x 3
                                Posed model joints.
                 v.f: A numpy array of the model face.
        '''        
        b = pose.shape[0]
        v_template = self.v_template[np.newaxis, :]
        shapedirs = self.shapedirs.view(-1, self.num_betas)[np.newaxis, :].expand(b, -1, -1)
        beta = betas[:, :, np.newaxis]
        v_shaped = torch.matmul(
            shapedirs, beta
        ).view(-1, 6890, 3) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        pose_quat = self._quat_feat(pose.view(-1, 3)).view(b, -1)
        pose_feat = torch.cat((pose_quat[:, 4:], beta[:, 1]), 1)

        R = self._rodrigues(pose.view(-1, 3)).view(b, 24, 3, 3)
        R = R.view(b, 24, 3, 3)#NOTE: get joint count

        posedirs = self.posedirs[np.newaxis, :].expand(b, -1, -1)
        v_posed = v_shaped + torch.matmul(
            posedirs, pose_feat[:, :, np.newaxis]
        ).view(-1, 6890, 3)#NOTE: get face count
        
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]

        G_ = torch.cat([R, J_[:, :, :, np.newaxis]], dim=-1)
        pad_row = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(
            self.faces.device
        ).view(1, 1, 1, 4).expand(b, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]        
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat(
            [J, torch.zeros_like(pad_row[..., 0])], dim=2
        ).view(b, 24, 4, 1)
        zeros = torch.zeros(b, 24, 4, 3).to(rest.device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(
            self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)
        ).view(6890, b, 4, 4).transpose(0, 1)
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1
        )
        v = torch.matmul(T, rest_shape_h[:, :, :, np.newaxis])[:, :, :3, 0]
        v = v + trans[:, np.newaxis, :]
            
        root_transform = self._with_zeros(
            torch.cat((R[:, 0], J[:, 0][:, :, np.newaxis]), 2)
        )
        results =  [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = self._with_zeros(
                torch.cat((
                    R[:, i + 1], J[:, i + 1][:, :, np.newaxis] - J[:, self.parent[i]][:, :, np.newaxis]), dim=2
                ))
            curr_res = torch.matmul(results[self.parent[i]], transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        J_transformed = posed_joints + trans[:, np.newaxis, :]
        
        res = { }
        res['vertices'] = v
        res['faces'] = self.faces #NOTE: or self.faces and ignore the np.array self.f
        res['v_posed'] = v_posed
        res['v_shaped'] = v_shaped
        res['joints'] = J_transformed
        return res


if __name__ == '__main__':
    MODEL_FILENAME = r'E:\VCL\Data\SMPL\star_1_1\neutral\model.npz'
    NUM_BETAS = 10
    star = STAR(MODEL_FILENAME, num_betas=NUM_BETAS)
    trans = torch.Tensor([[0.0, 0.0, 2.0]])
    pose = torch.randn(1, 72)
    betas = torch.zeros(1, NUM_BETAS)
    res = star(pose, betas, trans)
    for k, v in res.items():
        print(f"{k}: {v.shape}")
    