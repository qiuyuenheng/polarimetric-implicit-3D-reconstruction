import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np

import utils.general as utils
import utils.plots as plt

def get_max_dis(points, ori_camera_idx):
    import math
    """
    points:a list[shape:(1,3)]
    """
    # 初始化最长距离为0
    max_distance = 0

    # 遍历所有点对,计算最长距离
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # 计算两点之间的距离
            distance = math.sqrt((points[j][0][0] - points[i][0][0]) ** 2 +
                                 (points[j][0][1] - points[i][0][1]) ** 2 +
                                 (points[j][0][2] - points[i][0][2]) ** 2)
            # 更新最长距离
            if distance > max_distance:
                max_distance = distance

    radius = max_distance / 4.0   # 通过调整比例，减小融合所使用的camera数量
    # 计算可能可以用于融合的相机有哪些。
    mask = torch.zeros((len(points)), dtype=bool)
    ori_camera = points[ori_camera_idx]
    for i in range(len(points)):
        if i != ori_camera_idx:
            distance = math.sqrt((ori_camera[0][0] - points[i][0][0]) ** 2 +
                                 (ori_camera[0][1] - points[i][0][1]) ** 2 +
                                 (ori_camera[0][2] - points[i][0][2]) ** 2)
            if distance < radius:
                mask[i] = True

    return mask.cuda()


def visible(surface_points, camera_locations, valid_mask, sdf, sample_num=10):
    """
    surface_points：
    camera_locations：相机坐标（in world coordinate）
    valid_mask： 用于颜色对比所使用的所有可能的camera
    sdf：
    sample_num: 采样点到相机距离中，采样多少个sdf值

    return:  一个包含list的list，每个元素也是一个list，表示这个相机对于所有表面点的可见度。
    """
    camera_locations = torch.cat(camera_locations).cuda()
    delta = 0.2 / sample_num
    surface_points_num = len(surface_points)
    res = []

    for camera_location in camera_locations[valid_mask]:
        # direction = surface_points - camera_location.cuda().expand_as(surface_points)
        direction = camera_location.cuda().expand_as(surface_points) - surface_points
        # surface_points_to_camera_dist = torch.sqrt((direction ** 2).sum(dim=1))
        # delta = surface_points_to_camera_dist / sample_num / 2.0
        sample_points = torch.cat([(surface_points + i*delta*direction).unsqueeze(0) for i in range(1, sample_num+1)]).permute(1,0,2)
        sdf_value = sdf(sample_points.reshape(-1,3)).reshape(surface_points_num, sample_num)
        internal_mask = sdf_value <= 0.0
        res.append(internal_mask.sum(dim=1)==0)

    return res


class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']
        # self.render_color = kwargs['render_color']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(
            self.train_cameras,
            **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           )
        print('Creating model ...')
        self.stokes_render_weight_milestones = self.conf.get_list('train.stokes_render_weight_milestones', default=[])
        self.use_color = self.conf.get_int('train.use_color')
        self.use_refract_ray = self.conf.get_int('train.use_refract_ray')

        # self.sdf_network = utils.get_class(self.conf.get_string('train.sdf_network'))(**self.conf.get_config('model')['implicit_network'])
        # self.color_network = utils.get_class(self.conf.get_string('train.color_network'),name='color_network')(256,**self.conf.get_config('model')['RenderingNetwork'])
        # self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(self.stokes_render_weight_milestones,self.use_color,
                                                                                conf=self.conf.get_config('model'))

        if torch.cuda.is_available():
            self.model.cuda()
        print('Creating settings ...')
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(
            **self.conf.get_config('loss'),use_refract_ray=self.use_refract_ray)

        self.lr = self.conf.get_float('train.learning_rate')

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # 原版
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # self.optimizer = torch.optim.Adam(params_to_train, lr=self.lr)

        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones,
                                                              gamma=self.sched_factor)

        print('settings for camera optimization...')
        # settings for camera optimization
        self.num_images = len(self.train_dataset)
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(),
                                                        self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            # old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            old_checkpnts_dir = os.path.join(self.expdir, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            # self.color_network.load_state_dict(saved_model_state["color_state_dict"])
            # self.nerf.load_state_dict(saved_model_state["nerf_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            # 加载学习率
            self.optimizer.param_groups[0]['lr'] = self.lr
            for num, min_milestone in enumerate(self.sched_milestones):
                if self.start_epoch > min_milestone:
                    self.optimizer.param_groups[0]['lr'] = self.lr * (0.5 ** (num + 1))

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir,
                                 str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.fused_aolp = self.conf.get_int('train.fused_aolp')

        self.aolp_render_weight_milestones = self.conf.get_list('train.aolp_render_weight_milestones', default=[])
        self.stokes_render_weight_milestones = self.conf.get_list('train.stokes_render_weight_milestones', default=[])

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def get_transparent_area(self, model_input, model_outputs, fuse_aolp=True):
        # 获取本相机的颜色信息
        gt_color = model_outputs['gt_color'][model_outputs['first_net_mask'] & model_outputs['object_mask']]
        fuse_aolp = False if gt_color.shape[0] == 0 else fuse_aolp
        # 获取本相机的偏振信息
        gt_aolp = model_outputs['gt_aolp'][model_outputs['first_net_mask'] & model_outputs['object_mask']]
        gt_dolp = model_outputs['gt_dolp'][model_outputs['first_net_mask'] & model_outputs['object_mask']]

        camera_location = model_input["pose"][:, :3, 3]   # 这应该是世界坐标？
        current_camera_idx = model_input["idx"]

        all_camera_locations = [pose[:, :3, 3] for pose in model_input["all_pose"]]
        all_camera_locations_np = [pose[:, :3, 3].numpy() for pose in model_input["all_pose"]]
        # all_c2w = [pose[:, :3, :3] for pose in model_input["all_pose"]]
        # all_w2c = [np.linalg.inv(pose[:, :3, :3]) for pose in model_input["all_pose"]]
        all_w2c4 = [np.linalg.inv(pose) for pose in model_input["all_pose"]]
        # all_aolp = [aolp[0] for aolp in model_input["all_aolp"]]
        # all_dolp = [dolp[0] for dolp in model_input["all_dolp"]]

        which_camera_to_use = get_max_dis(all_camera_locations_np, current_camera_idx)
        surface_points = model_outputs["differentiable_surface_points"]  # 获取表面点
        sdf = model_outputs["sdf_network"]

        visible_mask = visible(surface_points, all_camera_locations, which_camera_to_use, sdf)
        useful_camera_id_list = torch.masked_select(torch.arange(0, len(which_camera_to_use)).cuda(),
                                                    which_camera_to_use)

        differents = []
        fused_aolp = gt_aolp[:,0].clone()
        max_dolp = gt_dolp[:,0].clone()

        available_cam_num = which_camera_to_use.sum()  # 可以用的相机的个数
        for i in range(available_cam_num):   # 对每个相机进行遍历
            camera_id = useful_camera_id_list[i]
            camera_visible_mask = visible_mask[i]   # 获取相机可见性。
            k = model_input["all_intrinsics"][camera_id][0].cuda()
            w2c = all_w2c4[camera_id][0]
            surface_points_world = torch.cat((surface_points, torch.ones_like(surface_points)[:,:1].cuda()), dim=1)
            surface_points_cam = torch.matmul(torch.from_numpy(w2c).cuda(),
                                              surface_points_world.permute(1, 0)).permute(1, 0)  # (vaild_ray_num,4)

            surface_points_pixel_mul_Zc = (torch.matmul(k, surface_points_cam.T).T)[:,:3]
            surface_points_pixel = surface_points_pixel_mul_Zc[:,:2]/surface_points_pixel_mul_Zc[:,2:3]
            pixel_x, pixel_y = surface_points_pixel[:,0], surface_points_pixel[:,1]   # 获取到像素坐标系

            ##  以上操作默认：像素坐标系的（0，0）在图片的左下角  ##
            picture_size_mask = (pixel_x >= 0) & (pixel_x <= (self.img_res[1] - 1)) & (pixel_y >= 0) & (
                        pixel_y <= (self.img_res[0] - 1))  # 要求计算得到的投影像素坐标系值要在图片大小范围内。
            ux, uy = pixel_x, pixel_y  # 获取符合的像素点坐标
            ux, uy = torch.round(ux).to(torch.int), torch.round(uy).to(torch.int)
            ux[~picture_size_mask] = 0
            uy[~picture_size_mask] = 0

            this_camera_color = gt_color.clone()
            this_camera_color[camera_visible_mask & picture_size_mask] = \
                (model_input['all_rgb'][camera_id][0].cuda()[:,uy,ux].T)[camera_visible_mask & picture_size_mask]  #(n,3)
            # visible_camera_colors.append(this_camera_color)

            ## 计算投影到其它相机的颜色与目标相机的颜色直接的马氏距离  ##
            distance = torch.abs(gt_color - this_camera_color)
            different = torch.sum(distance, dim=1)  # size: (n,)
            differents.append(different)

            if fuse_aolp:
                this_camera_aolp = gt_aolp.clone()
                this_camera_dolp = gt_dolp.clone()
                this_camera_aolp[camera_visible_mask & picture_size_mask] = \
                    (model_input['all_aolp'][camera_id][0].cuda()[:, uy, ux].T)[camera_visible_mask & picture_size_mask]  # (n,)
                this_camera_dolp[camera_visible_mask & picture_size_mask] = \
                    (model_input['all_dolp'][camera_id][0].cuda()[:, uy, ux].T)[camera_visible_mask & picture_size_mask]  # (n,)
                this_camera_aolp, this_camera_dolp = this_camera_aolp[:,0], this_camera_dolp[:, 0]
                change_mask = max_dolp < this_camera_dolp
                fused_aolp[change_mask] = this_camera_aolp[change_mask]
                max_dolp[change_mask] = this_camera_dolp[change_mask]

        ## 计算每个相机对应像素点的方差 ##
        var_sum = torch.zeros_like(gt_color[:,0]).cuda()
        for different in differents:
            var_sum = var_sum + different**2
        var = torch.sqrt(var_sum) / available_cam_num

        transparent_confidence = torch.zeros_like(model_input['rgb'][0,:,0])
        def sigmoid(x):
            coefficient = 20
            return 1. / (1. + torch.exp(coefficient*(0.5-x)))
        # var = sigmoid(var)
        var[var > 0.6] = 1.0
        transparent_confidence[model_outputs['first_net_mask'] & model_outputs['object_mask']] = var

        # transparent_mask = var > threshold
        if fuse_aolp:
            best_aolp = model_outputs['gt_aolp'].clone()
            best_aolp[model_outputs['first_net_mask'] & model_outputs['object_mask']] = fused_aolp.unsqueeze(1).repeat(1,3)
        else:
            best_aolp = model_outputs['gt_aolp']

        return transparent_confidence, best_aolp

    def get_transparent_area_three_channel(self, model_input, model_outputs, fuse_aolp=True):
        # 获取本相机的颜色信息
        gt_color = model_outputs['gt_color'][model_outputs['first_net_mask'] & model_outputs['object_mask']]
        fuse_aolp = False if gt_color.shape[0] == 0 else fuse_aolp
        # 获取本相机的偏振信息
        gt_aolp = model_outputs['gt_aolp'][model_outputs['first_net_mask'] & model_outputs['object_mask']]
        gt_dolp = model_outputs['gt_dolp'][model_outputs['first_net_mask'] & model_outputs['object_mask']]

        camera_location = model_input["pose"][:, :3, 3]   # 这应该是世界坐标？
        current_camera_idx = model_input["idx"]

        all_camera_locations = [pose[:, :3, 3] for pose in model_input["all_pose"]]
        all_camera_locations_np = [pose[:, :3, 3].numpy() for pose in model_input["all_pose"]]
        # all_c2w = [pose[:, :3, :3] for pose in model_input["all_pose"]]
        # all_w2c = [np.linalg.inv(pose[:, :3, :3]) for pose in model_input["all_pose"]]
        all_w2c4 = [np.linalg.inv(pose) for pose in model_input["all_pose"]]
        # all_aolp = [aolp[0] for aolp in model_input["all_aolp"]]
        # all_dolp = [dolp[0] for dolp in model_input["all_dolp"]]

        which_camera_to_use = get_max_dis(all_camera_locations_np, current_camera_idx)
        surface_points = model_outputs["differentiable_surface_points"]  # 获取表面点
        sdf = model_outputs["sdf_network"]

        visible_mask = visible(surface_points, all_camera_locations, which_camera_to_use, sdf)
        useful_camera_id_list = torch.masked_select(torch.arange(0, len(which_camera_to_use)).cuda(),
                                                    which_camera_to_use)

        differents = []
        fused_aolp = gt_aolp.clone()
        max_dolp = gt_dolp.clone()

        available_cam_num = which_camera_to_use.sum()  # 可以用的相机的个数
        for i in range(available_cam_num):   # 对每个相机进行遍历
            camera_id = useful_camera_id_list[i]
            camera_visible_mask = visible_mask[i]   # 获取相机可见性。
            k = model_input["all_intrinsics"][camera_id][0].cuda()
            w2c = all_w2c4[camera_id][0]
            surface_points_world = torch.cat((surface_points, torch.ones_like(surface_points)[:,:1].cuda()), dim=1)
            surface_points_cam = torch.matmul(torch.from_numpy(w2c).cuda(),
                                              surface_points_world.permute(1, 0)).permute(1, 0)  # (vaild_ray_num,4)

            surface_points_pixel_mul_Zc = (torch.matmul(k, surface_points_cam.T).T)[:,:3]
            surface_points_pixel = surface_points_pixel_mul_Zc[:,:2]/surface_points_pixel_mul_Zc[:,2:3]
            pixel_x, pixel_y = surface_points_pixel[:,0], surface_points_pixel[:,1]   # 获取到像素坐标系

            ##  以上操作默认：像素坐标系的（0，0）在图片的左下角  ##
            picture_size_mask = (pixel_x >= 0) & (pixel_x <= (self.img_res[1] - 1)) & (pixel_y >= 0) & (
                        pixel_y <= (self.img_res[0] - 1))  # 要求计算得到的投影像素坐标系值要在图片大小范围内。
            ux, uy = pixel_x, pixel_y  # 获取符合的像素点坐标
            ux, uy = torch.round(ux).to(torch.int), torch.round(uy).to(torch.int)
            ux[~picture_size_mask] = 0
            uy[~picture_size_mask] = 0

            this_camera_color = gt_color.clone()
            this_camera_color[camera_visible_mask & picture_size_mask] = \
                (model_input['all_rgb'][camera_id][0].cuda()[:,uy,ux].T)[camera_visible_mask & picture_size_mask]  #(n,3)
            # visible_camera_colors.append(this_camera_color)

            ## 计算投影到其它相机的颜色与目标相机的颜色直接的马氏距离  ##
            distance = torch.abs(gt_color - this_camera_color)
            different = torch.sum(distance, dim=1)  # size: (n,)
            differents.append(different)

            if fuse_aolp:
                this_camera_aolp = gt_aolp.clone()
                this_camera_dolp = gt_dolp.clone()
                this_camera_aolp[camera_visible_mask & picture_size_mask] = \
                    (model_input['all_aolp'][camera_id][0].cuda()[:, uy, ux].T)[camera_visible_mask & picture_size_mask]  # (n,)
                this_camera_dolp[camera_visible_mask & picture_size_mask] = \
                    (model_input['all_dolp'][camera_id][0].cuda()[:, uy, ux].T)[camera_visible_mask & picture_size_mask]  # (n,)
                this_camera_aolp, this_camera_dolp = this_camera_aolp[:,0], this_camera_dolp[:, 0]
                change_mask = max_dolp < this_camera_dolp
                fused_aolp[change_mask] = this_camera_aolp[change_mask]
                max_dolp[change_mask] = this_camera_dolp[change_mask]

        ## 计算每个相机对应像素点的方差 ##
        var_sum = torch.zeros_like(gt_color[:,0]).cuda()
        for different in differents:
            var_sum = var_sum + different**2
        var = torch.sqrt(var_sum) / available_cam_num

        transparent_confidence = torch.zeros_like(model_input['rgb'][0,:,0])
        transparent_confidence[model_outputs['first_net_mask'] & model_outputs['object_mask']] = var

        # transparent_mask = var > threshold
        if fuse_aolp:
            best_aolp = model_outputs['gt_aolp'].clone()
            best_aolp[model_outputs['first_net_mask'] & model_outputs['object_mask']] = fused_aolp.unsqueeze(1).repeat(1,3)
        else:
            best_aolp = model_outputs['gt_aolp']

        return transparent_confidence, best_aolp


    def run(self):
        print("training...")
        # self.model.eval()
        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.scheduler.last_epoch = epoch
            if (epoch >= self.aolp_render_weight_milestones[0]):  # 当epoch次数达到了开始使用aolp进行优化的时候
                self.loss.aolp_render_weight = self.loss.aolp_render_weight_init
            if epoch >= self.stokes_render_weight_milestones[0]:   # 此时开启stokes进行训练
                self.loss.stokes_render_weight = self.loss.stokes_render_weight_init
            if epoch >= self.stokes_render_weight_milestones[0] or self.use_color == False:   # 此时关闭颜色网络
            # if epoch >= 0:  # 此时关闭颜色网络
                self.loss.color_render_weight = 0.0
                self.model.rendering_network.eval()
                for name, param in self.model.rendering_network.named_parameters():
                    param.requires_grad = False
            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch % 50 == 0:
                self.save_checkpoints(epoch)

            if epoch % self.plot_freq == 0:
                self.model.eval()
                # if self.render_color is True:
                #     self.color_network.eval()
                if self.train_cameras:
                    self.pose_vecs.eval()

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                # idx = np.random.randint(0, len(self.plot_dataloader.dataset.aolp_images))
                for i, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                    if indices == 13: break

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()
                model_input["rgb"] = ground_truth['rgb'].cuda()
                model_input["origin_rgb"] = ground_truth["origin_rgb"].cuda()  # 1, 3, 1028, 1232
                model_input["origin_object_mask"] = ground_truth["origin_object_mask"].cuda()  # 1, 1028, 1232
                model_input["origin_dolp"] = ground_truth["origin_dolp"].cuda()
                model_input["origin_aolp"] = ground_truth["origin_aolp"].cuda()
                model_input["aolp"] = ground_truth["aolp"].cuda()
                model_input["dolp"] = ground_truth["dolp"].cuda()
                model_input["gt_s0"] = ground_truth["s0"].cuda()
                model_input["gt_s1"] = ground_truth["s1"].cuda()
                model_input["gt_s2"] = ground_truth["s2"].cuda()
                model_input["all_rgb"] = ground_truth["all_rgb"]
                model_input["all_aolp"] = ground_truth["all_aolp"]
                model_input["all_dolp"] = ground_truth["all_dolp"]
                model_input["img_res"] = self.img_res
                model_input["batch_size"] = self.num_pixels

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()

                split = utils.split_input(model_input, self.total_pixels)
                res = []
                for s in split:
                    # out = self.model(s)
                    if self.use_refract_ray:
                        out = self.model.refract_ray_forward(s, epoch)
                    else:
                        out = self.model.straight_ray_forward(s, epoch)
                        # out = self.model(s)
                    transparent_confidence, fused_aolp = self.get_transparent_area(s, out, fuse_aolp=self.fused_aolp)
                    res.append({
                        'points': out['points'].detach(),
                        'rgb_values': out['rgb_values'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach(),
                        'transparent_confidence': transparent_confidence.detach(),
                        'fused_aolp': fused_aolp.detach(),
                        'normals': out['normals'].detach(),
                        's0_spec': out['stokes_r'][:, 0].detach(),
                        's0_diff': out['stokes_d'][:, 0].detach(),
                    })
                    del out

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                plt.plot(self.model,
                         indices,
                         model_outputs,
                         model_input['pose'],
                         ground_truth['rgb'],
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )
                del res, model_outputs
                self.model.train()

                if epoch >= self.stokes_render_weight_milestones[0] or self.use_color == False:  # 此时关闭颜色网络
                    # if epoch >= 0:  # 此时关闭颜色网络
                    self.loss.color_render_weight = 0.0
                    self.model.rendering_network.eval()
                    for name, param in self.model.rendering_network.named_parameters():
                        param.requires_grad = False

                if self.train_cameras:
                    self.pose_vecs.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            color_loss, stokes_loss, aolp_render_loss = 0, 0, 0

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                # if indices == 10:
                #     print(*[p for name, p in self.model.named_parameters()], sep='\n')
                # for group in self.optimizer.param_groups:
                #     for param in group["params"]:
                #         k=1

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()
                model_input["origin_rgb"] = ground_truth["origin_rgb"].cuda()
                model_input["origin_object_mask"] = ground_truth["origin_object_mask"].cuda()
                model_input["origin_dolp"] = ground_truth["origin_dolp"].cuda()
                model_input["origin_aolp"] = ground_truth["origin_aolp"].cuda()
                model_input["aolp"] = ground_truth["aolp"].cuda()
                model_input["dolp"] = ground_truth["dolp"].cuda()
                model_input["gt_s0"] = ground_truth["s0"].cuda()
                model_input["gt_s1"] = ground_truth["s1"].cuda()
                model_input["gt_s2"] = ground_truth["s2"].cuda()
                model_input["rgb"] = ground_truth["rgb"].cuda()
                model_input["all_rgb"] = ground_truth["all_rgb"]
                model_input["all_aolp"] = ground_truth["all_aolp"]
                model_input["all_dolp"] = ground_truth["all_dolp"]
                model_input["img_res"] = self.img_res
                model_input["batch_size"] = self.num_pixels

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()

                # torch.autograd.set_detect_anomaly(True)
                # self.model.cuda()
                if self.use_refract_ray:
                    model_outputs = self.model.refract_ray_forward(model_input, epoch)
                else:
                    model_outputs = self.model.straight_ray_forward(model_input, epoch)
                    # model_outputs = self.model(model_input)

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print("requires_grad: True ", name)
                #     else:
                #         print("requires_grad: False ", name)

                transparent_confidence, fused_aolp = self.get_transparent_area(model_input, model_outputs, fuse_aolp=self.fused_aolp)

                model_outputs['transparent_confidence'] = transparent_confidence
                model_outputs['fused_aolp'] = fused_aolp

                # with torch.autograd.detect_anomaly():
                loss_output = self.loss(model_input, model_outputs, ground_truth, epoch,
                                        self.train_dataset.sampling_idx)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                # if torch.isnan(loss).any():
                #     continue
                # clip_gradient(optimizer=self.optimizer, grad_clip=3)

                loss.backward()

                # clip_gradient(optimizer=self.optimizer,grad_clip=3)
                self.optimizer.step()

                # for group in self.optimizer.param_groups:
                #     for param in group["params"]:
                #         k=1

                if self.train_cameras:
                    self.optimizer_cam.step()

                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, eikonal_loss = {5}, mask_loss = {6}, aolp_render_loss = {7}, alpha = {8}, lr = {9}'
                    .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                            loss_output['eikonal_loss'].item(),
                            loss_output['mask_loss'].item(),
                            loss_output['aolp_render_loss'].item(),
                            self.loss.alpha,
                            self.scheduler.get_lr()[0])
                )
                # print('aolp_loss = {0}'.format(loss_output['aolp_loss'].item()))
                print("stokes_loss = {0}".format(loss_output['stokes_loss'].item()))
                print("color_loss = {0}".format(loss_output['color_loss'].item()))

                color_loss = color_loss + loss_output['color_loss'].item()
                stokes_loss = stokes_loss + loss_output['stokes_loss'].item()
                aolp_render_loss = aolp_render_loss + loss_output['aolp_render_loss'].item()

            # if os.path.exists('log_spamle_kun.txt'):
            #     os.remove('log_spamle_kun.txt')

            if epoch % 5 == 0:
                self.log_loss(epoch, color_loss/self.num_images, stokes_loss/self.num_images, aolp_render_loss/self.num_images)
            self.scheduler.step()

    def log_loss(self,epoch, color_loss, stokes_loss, aolp_render_loss):
        import os
        filename = os.path.join(self.expdir, self.timestamp, 'log_loss.txt')
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write("epcoh:       color_loss       stokes_loss         aolp_render_loss\n")
                file.write("{0} :        {1}              {2}                {3}\n"
                           .format(epoch, format(color_loss, '.4f'),
                                   format(stokes_loss, '.4f'),format(aolp_render_loss, '.6f')))
        else:
            # 如果文件已存在，打开文件并追加"true"
            with open(filename, 'a') as file:
                file.write("{0} :        {1}              {2}                {3}\n"
                           .format(epoch, format(color_loss, '.4f'),
                                   format(stokes_loss, '.4f'),format(aolp_render_loss, '.6f')))

