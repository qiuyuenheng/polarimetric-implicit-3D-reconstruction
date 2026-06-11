import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 num_views,
                 cam_file=None
                 ):

        self.instance_dir = data_dir

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        image_dir = '{0}/I-sum'.format(self.instance_dir)
        image_paths_all = sorted(utils.glob_imgs(image_dir))

        # image_00_dir = '{0}/I-00'.format(self.instance_dir)
        # image_00_paths_all = sorted(utils.glob_imgs(image_00_dir))
        # image_45_dir = '{0}/I-45'.format(self.instance_dir)
        # image_45_paths_all = sorted(utils.glob_imgs(image_45_dir))
        # image_90_dir = '{0}/I-90'.format(self.instance_dir)
        # image_90_paths_all = sorted(utils.glob_imgs(image_90_dir))
        # image_135_dir = '{0}/I-135'.format(self.instance_dir)
        # image_135_paths_all = sorted(utils.glob_imgs(image_135_dir))

        mask_dir = '{0}/masks'.format(self.instance_dir)
        mask_paths_all = sorted(utils.glob_imgs(mask_dir))
        # normal_dir ='{0}/normals-png'.format(self.instance_dir)
        # normal_paths_all = sorted(utils.glob_imgs(normal_dir))
        aolp_dir = '{0}/params/AoLP'.format(self.instance_dir)
        aolp_paths_all = sorted(utils.glob_imgs(aolp_dir))
        dolp_dir = '{0}/params/DoLP'.format(self.instance_dir)
        dolp_paths_all = sorted(utils.glob_imgs(dolp_dir))

        # num_views sampler
        if(num_views>0):
            interval = int(len(image_paths_all) / num_views)
        else:
            interval = 1
            num_views = len(image_paths_all)

        image_paths = []
        image_00_paths, image_45_paths, image_90_paths, image_135_paths = [],[],[],[]
        mask_paths = []
        normal_paths = []
        aolp_paths = []
        dolp_paths = []
        for i in range(0,num_views):
            if(interval * i > len(image_paths_all)):
                image_paths.append(image_paths_all[-1])
                mask_paths.append(mask_paths_all[-1])
                # normal_paths.append(normal_paths_all[-1])
                aolp_paths.append(aolp_paths_all[-1])
                dolp_paths.append(dolp_paths_all[-1])
                # image_00_paths.append(image_00_paths_all[-1])
                # image_45_paths.append(image_45_paths_all[-1])
                # image_90_paths.append(image_90_paths_all[-1])
                # image_135_paths.append(image_135_paths_all[-1])
            else:
                image_paths.append(image_paths_all[interval * i])
                mask_paths.append(mask_paths_all[interval * i])
                # normal_paths.append(normal_paths_all[interval * i])
                aolp_paths.append(aolp_paths_all[interval * i])
                dolp_paths.append(dolp_paths_all[interval * i])
                # image_00_paths.append(image_00_paths_all[interval * i])
                # image_45_paths.append(image_45_paths_all[interval * i])
                # image_90_paths.append(image_90_paths_all[interval * i])
                # image_135_paths.append(image_135_paths_all[interval * i])


        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras_new.npz'.format(self.instance_dir)
        # self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths_all))]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(image_paths_all))]
        # scale_mats,world_mats = [],[]
        # for idx in range(len(image_paths_all)+1):
        #     # if idx == 5:continue
        #     scale_mats.append(camera_dict['scale_mat_%d' % idx])
        #     world_mats.append(camera_dict['world_mat_%d' % idx])

        intrinsics_all_all = []
        pose_all_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            #intrinsics[0, 2] /= 2
            #intrinsics[1, 2] /= 2
            #intrinsics[0, 0] /= 2
            #intrinsics[1, 1] /= 2
            intrinsics_all_all.append(torch.from_numpy(intrinsics).float())
            pose_all_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = []
        self.pose_all = []
        for i in range(0,num_views):
            if(interval * i > len(image_paths_all)):
                self.intrinsics_all.append(intrinsics_all_all[-1])
                self.pose_all.append(pose_all_all[-1])

            else:
                self.intrinsics_all.append(intrinsics_all_all[interval * i])
                self.pose_all.append(pose_all_all[interval * i])


        self.rgb_images = []
        self.origin_rgb_images = []
        for path in image_paths:
            # rgb = rend_util.load_gray_as_rgb(path)
            rgb = rend_util.load_rgb(path, downscale=1)
            self.origin_rgb_images.append(torch.from_numpy(rgb).float())
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        data_category = 'SNeRSP'
        if data_category == 'Pandora':
            self.s0, self.s1, self.s2 = [], [], []
            downscale = 4
            for i in range(1,self.n_images + 1):
                image_name = "0"+str(i) if i<10 else str(i)
                s0 = 0.5 * rend_util.load_exr(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0.hdr', downscale)
                s0p1 = 0.5 * rend_util.load_exr(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0p1.hdr', downscale)
                s0p2 = 0.5 * rend_util.load_exr(f'{self.instance_dir}/images_stokes/{int(image_name):02d}_s0p2.hdr', downscale)
                s1 = s0p1 - s0
                s2 = s0p2 - s0
                self.s0.append(s0.reshape(3, -1).transpose(1, 0))
                self.s1.append(s1.reshape(3, -1).transpose(1, 0))
                self.s2.append(s2.reshape(3, -1).transpose(1, 0))
        elif data_category == 'RNeRSP':
            self.s0, self.s1, self.s2 = [], [], []
            self.s0_ori, self.s1_ori, self.s2_ori = [], [], []
            for i in range(1, self.n_images + 1):
                s0,s1,s2 = rend_util.load_stokes_npz(f'{self.instance_dir}/images_stokes/{int(i - 1):04d}.npz', 1)
                s0, s1, s2 = s0.transpose(2,0,1),s1.transpose(2,0,1),s2.transpose(2,0,1)
                self.s0_ori.append(s0.astype(np.float32))
                self.s1_ori.append(s1.astype(np.float32))
                self.s2_ori.append(s2.astype(np.float32))
                self.s0.append(s0.reshape(3,-1).astype(np.float32).transpose(1,0))
                self.s1.append(s1.reshape(3,-1).astype(np.float32).transpose(1,0))
                self.s2.append(s2.reshape(3,-1).astype(np.float32).transpose(1,0))
        elif data_category == 'SNeRSP':
            self.s0, self.s1, self.s2 = [], [], []
            self.s0_ori, self.s1_ori, self.s2_ori = [], [], []
            for i in range(self.n_images):
                s0 = rend_util.load_rgb_npy(f'{self.instance_dir}/s0/{i:04d}.npy', 1) * 0.5
                s1 = rend_util.load_rgb_npy(f'{self.instance_dir}/s1/{i:04d}.npy', 1) * 0.5
                s2 = rend_util.load_rgb_npy(f'{self.instance_dir}/s2/{i:04d}.npy', 1) * 0.5
                # s0, s1, s2 = s0.transpose(2,0,1),s1.transpose(2,0,1),s2.transpose(2,0,1)
                self.s0_ori.append(s0.astype(np.float32).transpose(2,0,1))
                self.s1_ori.append(s1.astype(np.float32).transpose(2,0,1))
                self.s2_ori.append(s2.astype(np.float32).transpose(2,0,1))
                self.s0.append(s0.reshape(-1,3).astype(np.float32))
                self.s1.append(s1.reshape(-1,3).astype(np.float32))
                self.s2.append(s2.reshape(-1,3).astype(np.float32))
        elif data_category == 'NeISF':
            self.s0, self.s1, self.s2 = [], [], []
            self.s0_ori, self.s1_ori, self.s2_ori = [], [], []
            for i in range(1,self.n_images + 1):
                image_name = "00"+str(i) if i<10 else "0" + str(i)
                image_name = "100" if image_name == "0100" else image_name
                s0 = rend_util.load_exr(f'{self.instance_dir}/images_s0/img_{image_name}.exr', 1)
                s1 = rend_util.load_exr(f'{self.instance_dir}/images_s1/img_{image_name}.exr', 1)
                s2 = rend_util.load_exr(f'{self.instance_dir}/images_s2/img_{image_name}.exr', 1)
                self.s0_ori.append(s0.astype(np.float32))
                self.s1_ori.append(s1.astype(np.float32))
                self.s2_ori.append(s2.astype(np.float32))
                self.s0.append(s0.reshape(-1,3))
                self.s1.append(s1.reshape(-1,3))
                self.s2.append(s2.reshape(-1,3))

                self.aolp_images = []
                self.origin_aolp = []
                aolp = 0.5 * torch.atan2(torch.from_numpy(s2), torch.from_numpy(s1) + 1e-6)
                aolp = torch.remainder(aolp, torch.pi)
                self.origin_aolp.append(aolp.float())
                aolp = aolp.reshape(-1,3)  # (0,1)
                # aolp = aolp * np.pi # (0,pi)
                self.aolp_images.append(aolp.float())

                self.dolp_images = []
                self.origin_dolp = []
                dolp = torch.sqrt(torch.from_numpy(s1 ** 2) + torch.from_numpy(s2 ** 2)) / (torch.from_numpy(s0) + 1e-6)
                dolp[s0 < 1e-6] = 0.0
                self.origin_dolp.append(dolp.float())
                dolp = dolp.reshape(3, -1).transpose(1, 0)  # (0,1)
                # aolp = aolp * np.pi # (0,pi)
                self.dolp_images.append(dolp.float())


        elif data_category == 'PIR':
            #### 加载四个偏振度 ####
            self.rgb_images_00 = []
            for path in image_00_paths:
                # rgb = rend_util.load_gray_as_rgb(path)
                rgb = rend_util.load_polar_rgb(path)
                # rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(-1)
                self.rgb_images_00.append(torch.from_numpy(rgb).float())
            self.rgb_images_45 = []
            for path in image_45_paths:
                # rgb = rend_util.load_gray_as_rgb(path)
                rgb = rend_util.load_polar_rgb(path)
                # rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(-1)
                self.rgb_images_45.append(torch.from_numpy(rgb).float())
            self.rgb_images_90 = []
            for path in image_90_paths:
                # rgb = rend_util.load_gray_as_rgb(path)
                rgb = rend_util.load_polar_rgb(path)
                # rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(-1)
                self.rgb_images_90.append(torch.from_numpy(rgb).float())
            self.rgb_images_135 = []
            for path in image_135_paths:
                # rgb = rend_util.load_gray_as_rgb(path)
                rgb = rend_util.load_polar_rgb(path)
                # rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(-1)
                self.rgb_images_135.append(torch.from_numpy(rgb).float())
            #### 完成加载四个偏振度 ####

            #### 计算三个通道的stokes ####
            img_num = len(self.rgb_images)
            print(self.rgb_images_00, self.rgb_images_45, self.rgb_images_90, self.rgb_images_135)
            self.s0 = [((self.rgb_images_00[i] / 2 + 0.5) +
                        (self.rgb_images_45[i] / 2 + 0.5) +
                        (self.rgb_images_90[i] / 2 + 0.5) +
                        (self.rgb_images_135[i] / 2 + 0.5)) / 2.0 for i in range(img_num)]
            self.s1 = [(self.rgb_images_00[i] / 2 + 0.5) - (self.rgb_images_90[i] / 2 + 0.5) for i in range(img_num)]
            self.s2 = [(self.rgb_images_45[i] / 2 + 0.5) - (self.rgb_images_135[i] / 2 + 0.5) for i in range(img_num)]


        self.object_masks = []
        self.origin_object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path, downscale=1)
            self.origin_object_masks.append(torch.from_numpy(object_mask).bool())
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

        self.aolp_images = []
        self.origin_aolp = []
        for path_idx in range(len(aolp_paths)):
            aolp = 0.5 * torch.atan2(torch.from_numpy(self.s2_ori[path_idx]), torch.from_numpy(self.s1_ori[path_idx]) + 1e-6)
            aolp = torch.remainder(aolp, torch.pi)
            # import cv2
            # im = cv2.cvtColor(aolp.numpy(), cv2.COLOR_BGR2RGB)
            # cv2.imshow("k", im[:,:,1]/np.pi)
            # cv2.waitKey(0)
            self.origin_aolp.append(aolp.float())
            aolp = aolp.reshape(3,-1).transpose(1,0) # (0,1)
            # aolp = aolp * np.pi # (0,pi)
            self.aolp_images.append(aolp.float())

        self.dolp_images = []
        self.origin_dolp = []
        for path_idx in range(len(dolp_paths)):
            # dolp = rend_util.load_aolp_as_rgb(path)   # 3通道
            dolp = torch.sqrt(torch.from_numpy(self.s1_ori[path_idx] ** 2)+torch.from_numpy(self.s2_ori[path_idx] ** 2)) / (torch.from_numpy(self.s0_ori[path_idx]) + 1e-6)
            # dolp[self.s0_ori[path_idx] < 1e-6] = 0.0
            # import cv2
            # im = cv2.cvtColor(aolp.numpy(), cv2.COLOR_BGR2RGB)
            # cv2.imshow("k", im[:,:,1]/np.pi)
            # cv2.waitKey(0)
            self.origin_dolp.append(dolp.float())
            dolp = dolp.reshape(3,-1).transpose(1,0)  # (0,1)
            # aolp = aolp * np.pi # (0,pi)
            self.dolp_images.append(dolp.float())


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "idx": idx,
            "all_intrinsics": self.intrinsics_all,
            "all_pose": self.pose_all,

            # 's0': self.s0[idx],
            # 's1': self.s1[idx],
            # 's2': self.s2[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "origin_rgb": self.origin_rgb_images[idx],
            "origin_object_mask":self.origin_object_masks[idx],
            # "normal": self.normal_images[idx], # (-1,1)
            "aolp":self.aolp_images[idx],     # (-1,1)\
            "origin_aolp": self.origin_aolp[idx],
            "dolp":self.dolp_images[idx],
            "origin_dolp":self.origin_dolp[idx],

            "all_aolp": self.origin_aolp,
            "all_dolp": self.origin_dolp,
            "all_rgb" : self.origin_rgb_images,

            's0': self.s0[idx],
            's1': self.s1[idx],
            's2': self.s2[idx],
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["s0"] = self.s0[idx][self.sampling_idx]
            ground_truth["s1"] = self.s1[idx][self.sampling_idx]
            ground_truth["s2"] = self.s2[idx][self.sampling_idx]

            # ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["aolp"] = self.aolp_images[idx][self.sampling_idx, :]
            ground_truth["dolp"] = self.dolp_images[idx][self.sampling_idx, :]

            # ground_truth["physics"] = [self.physics[idx][0][self.sampling_idx, :],self.physics[idx][1][self.sampling_idx, :],
            #                            self.physics[idx][2][self.sampling_idx, :],self.physics[idx][3][self.sampling_idx, :],
            #                            self.physics[idx][4][self.sampling_idx, :],self.physics[idx][5][self.sampling_idx, :]]
            # ground_truth["var_confident"] = self.var_confident[idx][self.sampling_idx]

            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
