import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import struct
from models.rotation import Quaternion
from models.camera import Camera


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose




class Image:
    def __init__(self, name_, camera_id_, q_, tvec_):
        self.name = name_
        self.camera_id = camera_id_
        self.q = q_
        self.tvec = tvec_

        self.points2D = np.empty((0, 2), dtype=np.float64)
        self.point3D_ids = np.empty((0,), dtype=np.uint64)

    #---------------------------------------------------------------------------

    def R(self):
        return self.q.ToR()

    #---------------------------------------------------------------------------

    def C(self):
        return -self.R().T.dot(self.tvec)

    #---------------------------------------------------------------------------

    @property
    def t(self):
        return self.tvec


def get_pose_norm(c2w, target_radius):


    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for i in range(c2w.shape[0]):
        cam_centers.append(c2w[i, :3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    scale = target_radius / radius
    cam_center_all = c2w[:, :3, 3]
    cam_center_all = (cam_center_all + translate) * scale
    c2w = c2w.astype(np.float64)
    c2w[:, :3, 3] = cam_center_all

    return c2w



class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.use_self_data = conf.get_bool('use_self_data', default=False)
        if not self.use_self_data:
            self.data_dir = conf.get_string('data_dir')
            self.render_cameras_name = conf.get_string('render_cameras_name')
            self.object_cameras_name = conf.get_string('object_cameras_name')

            self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
            self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

            # world_mat is a projection matrix from world to image
            self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.scale_mats_np = []

            # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

            self.intrinsics_all = []
            self.pose_all = []

            for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
            self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]

            print('Load data: End')
        else:
            img_down = int(conf.get_string('img_down'))
            self.img_down = img_down
            self.data_dir = conf.get_string('self_data_dir')
            self.render_cameras_name = conf.get_string('render_cameras_name')
            self.object_cameras_name = conf.get_string('object_cameras_name')

            self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
            self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

            self.image_lists = []
            self.camera_list = []
            self._load_images_bin(os.path.join(conf.self_data_dir, 'sparse/0/images.bin'))
            self._load_cameras_bin(os.path.join(conf.self_data_dir, 'sparse/0/cameras.bin'))
            cam = self.camera_list[0]
            # Extract focal lengths and principal point parameters.
            fx, fy, cx, cy = cam.fx / img_down, cam.fy / img_down, cam.cx / img_down, cam.cy / img_down
            intri = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1.],
                ])
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = intri
            
            self.images_np = np.stack([cv.imread(self.data_dir + f'/images_{img_down}/' + image_.name) for image_ in self.image_lists]) / 256.0
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
            for image_i in self.image_lists:
                rot = image_i.R()
                trans = image_i.tvec.reshape(3, 1)
                w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
                w2c_mats.append(w2c)
            w2c_mats = np.stack(w2c_mats, axis=0)
            c2w_mats = np.linalg.inv(w2c_mats)
            c2w_mats_norm = get_pose_norm(c2w=c2w_mats, target_radius=1.)
            # poses = c2w_mats[:, :3, :4]
            poses = c2w_mats_norm[:, :3, :4]
            # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
            poses = poses @ np.diag([1, -1, -1, 1])

            # self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images.new/*.jpg')))
            # self.cams_list = sorted(glob(os.path.join(self.data_dir, 'cams/*.txt')))
            self.n_images = len(self.image_lists)

            # self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
            # self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0


            self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
            self.masks = torch.from_numpy(np.ones_like(self.images_np)).cpu()  # [n_images, H, W, 3]
            self.intrinsics_all = torch.from_numpy(intrinsics).repeat(self.n_images, 1, 1).to(self.device)  # [n_images, 4, 4]
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]
            self.pose_all = torch.from_numpy(poses).to(self.device)  # [n_images, 4, 4]
            self.H, self.W = self.images.shape[1], self.images.shape[2]
            self.image_pixels = self.H * self.W

            object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
            object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
            # Object scale mat: region of interest to **extract mesh**
            # object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            # object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            # object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3]
            self.object_bbox_max = object_bbox_max[:3]

            print('Load data: End')

    def _load_images_bin(self, input_file):
        images = {}
        with open(input_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            for image_index in range(num_reg_images):
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                # qvec = np.array(binary_image_properties[1:5])
                q = Quaternion(np.array(binary_image_properties[1:5]))
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":  # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                           format_char_sequence="ddq" * num_points2D)

                image = Image(q_=q, tvec_=tvec,
                    camera_id_=camera_id, name_=image_name,
                    )
                self.image_lists.append(image)


    def _load_cameras_bin(self, input_file):

        with open(input_file, 'rb') as f:
            num_cameras = struct.unpack('L', f.read(8))[0]

            for _ in range(num_cameras):
                camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))
                num_params = Camera.GetNumParams(camera_type)
                params = struct.unpack('d' * num_params, f.read(8 * num_params))
                self.camera_list.append(Camera(camera_type, w, h, params))
                # self.last_camera_id = max(self.last_camera_id, camera_id)



    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3].float(), p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3].float(), rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].float().expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)].float()    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)].float()      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3].float(), p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3].float(), rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape).float() # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        if not self.use_self_data:
            img = cv.imread(self.images_lis[idx])
            return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
        else:
            img = cv.imread(self.data_dir + f'/images_{self.img_down}/' + self.image_lists[idx].name)
            return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)