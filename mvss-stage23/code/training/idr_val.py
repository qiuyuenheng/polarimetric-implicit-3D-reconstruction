import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np

import utils.general as utils
import utils.plots as plt

class IDRTestRunner():
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


    def run(self):
        print("test...")
        self.model.eval()

        self.train_dataset.change_sampling_idx(-1)

        indices, model_input, ground_truth = next(iter(self.plot_dataloader))
        # idx = np.random.randint(0, len(self.plot_dataloader.dataset.aolp_images))
        # for i, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
        #     if i == idx: break

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

        sdf = self.model['implicit_network']

        if self.train_cameras:
            pose_input = self.pose_vecs(indices.cuda())
            model_input['pose'] = pose_input
        else:
            model_input['pose'] = model_input['pose'].cuda()


        # plt.plot(self.model,
        #          indices,
        #          model_outputs,
        #          model_input['pose'],
        #          ground_truth['rgb'],
        #          self.plots_dir,
        #          self.start_epoch,
        #          self.img_res,
        #          **self.plot_conf
        #          )

