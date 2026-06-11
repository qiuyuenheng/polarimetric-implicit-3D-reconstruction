import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import *
from model.sample_network import SampleNetwork

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x


class SIRENLayer(nn.Module):
    """ SIREN layer [Sitzmann et al. 2020].

    For the implementation, refer to:
        https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L10.
    The original paper is: https://arxiv.org/pdf/2006.09661.pdf.

    Attributes:
        in_num (int): the number of input dimension.
        omega_o (float): a hyperparameter for initialization. For more details, see Sec. 3.2 of the original paper.
    """

    def __init__(self,
                 in_num: int,
                 out_num: int,
                 use_bias: bool = True,
                 is_first_layer: bool = False,
                 omega_o: float = 30.):
        super(SIRENLayer, self).__init__()

        self.in_num = in_num
        self.omega_o = omega_o

        self.linear = nn.Linear(in_num, out_num, bias=use_bias)

        # initialize weights of the linear layer.
        # See Sec 3.2 of the original paper.
        if is_first_layer:
            nn.init.uniform_(self.linear.weight, -1 / self.in_num * self.omega_o, 1 / self.in_num * self.omega_o)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(3 / self.in_num), np.sqrt(3 / self.in_num))
        nn.init.zeros_(self.linear.bias)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(inx))

class IncidentNet(nn.Module):
    """ this is an MLP for estimating a position-dependent stokes vector.
    Implementation is modified from: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L114

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        last_activation_func : activation function for the output. if empty, no activation.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 multires: int,
                 skips: list,
                 last_activation_func,
                 weight_norm: bool = False):
        super(IncidentNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.last_activation_func = last_activation_func
        embed_fn, input_ch = get_embedder(multires)
        self.embed_fn = embed_fn

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = SIRENLayer(in_ch, width, is_first_layer=True)

            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, np.log(1.5))  # ?

            elif i in skips:  # this layer contains a skip connection.
                layer = SIRENLayer(3 + in_ch + width, width)  # pos (3,) + view_embedded (emb_out,) + width

            else:
                layer = SIRENLayer(width, width)

            if weight_norm:
                if isinstance(layer, SIRENLayer):
                    layer.linear = nn.utils.weight_norm(layer.linear)
                else:
                    raise ValueError("all the weights are initialized to be zero. "
                                     "weight normalization could produce`nan`value.")

            setattr(self, "layer_{}".format(i), layer)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        pos_embeded = self.embed_fn(inx[:,:3])
        wi_embeded = self.embed_fn(inx[:, 3:])
        x = torch.cat((pos_embeded, wi_embeded),dim=1)
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x


class IDRNetwork(nn.Module):
    def __init__(self, stokes_render_weight_milestones,use_color,conf):
    # def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.refract_ray_tracer = Refract_RayTracing(**conf.get_config('refract_ray_tracer'))
        self.straight_ray_tracer = Straight_RayTracing(**conf.get_config('straight_ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('straight_ray_tracer.object_bounding_sphere')
        self.mueller = Mueller()
        self.stokes_render_weight_milestones = stokes_render_weight_milestones
        self.use_color = use_color
        # self.gb = nn.Linear(100,100)
        #
        self.incident_s0_net = IncidentNet(
            depth=4,
            width=256,
            in_ch=27 + 27,
            out_ch=2,  # 3-channel s0
            multires=4,
            last_activation_func=nn.Softplus(),  # the value of the incident light can be greater than 1.0
            skips=[4]  # no skip
        )
        self.incident_diff_net = IncidentNet(
            depth=4,
            width=256,
            in_ch=27 + 27,
            out_ch=2,  # 3-channel s1
            multires=4,
            last_activation_func=nn.Tanh(),  # s1/s0 and s2/s0 are between [-1.0, 1.0]
            skips=[4]
        )
        self.incident_spec_net = IncidentNet(
            depth=4,
            width=256,
            in_ch=27 + 27,
            out_ch=2,  # 3-channel s1, 3-channel s2
            multires=4,
            last_activation_func=nn.Tanh(),  # s1/s0 and s2/s0 are between [-1.0, 1.0]
            skips=[4]
        )

    def refract_ray_forward(self, input, epoch):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        # rgb = input["origin_rgb"]
        # origin_object_mask = input["origin_object_mask"]
        # origin_dolp = input['origin_dolp']
        # origin_aolp = input['origin_aolp']
        object_mask = input["object_mask"].reshape(-1)
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics) #(1,5120,3)  (1,3)
        # ray_dirs, cam_loc, gt_color, gt_mask, gt_aolp, gt_dolp = self.gen_random_rays_at(input['img_res'], pose, intrinsics, rgb,
        #                                                                origin_object_mask, origin_aolp, origin_dolp,
        #                                                                input['batch_size'],xy_list=xy_list)
        # object_mask = gt_mask
        (gt_color, gt_aolp, gt_dolp, gt_s0, gt_s1, gt_s2) = (input["rgb"][0], input["aolp"][0], input["dolp"][0],
                                                             input["gt_s0"][0], input["gt_s1"][0], input["gt_s2"][0])

        batch_size, num_pixels, _ = ray_dirs.shape


        self.implicit_network.eval()
        for name, param in self.implicit_network.named_parameters():
            param.requires_grad = False

        first_pts, first_net_mask, first_dists,\
        second_pts, second_net_mask, second_dists,\
        first_refract_dirs, first_reflect_dirs, first_attenuate, first_refract_total_reflect_mask,\
        second_refract_dirs, second_reflect_dirs, second_attenuate, second_refract_total_reflect_mask,stokes,dict,valid_mask,\
        bottom_start_pts,bottom_pts,bottom_mask, near_far = self.refract_ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                    normal_fn = lambda x: self.implicit_network.gradient(x)[:,0,:],
                                                                    cam_loc=cam_loc,
                                                                    object_mask=object_mask,
                                                                    ray_directions=ray_dirs
                                                                    )
        if self.stokes_render_weight_milestones[0] > epoch or self.stokes_render_weight_milestones[1] < epoch:
            self.implicit_network.train()
            for name, param in self.implicit_network.named_parameters():
                param.requires_grad = True

        # show_trace(first_pts,second_pts,cam_loc)
        # show_direction(second_pts,second_refract_dirs,cam_loc)
        # show_direction2(cam_loc, ray_dirs[0][first_net_mask])
        # 可知，first_pts是光线第一次碰到物体的采样点。second_pts是光线第二次碰到物体的采样点。
        # first_pts = first_pts.split(512)

        # first_net_mask = first_net_mask & object_mask

        n_rays = first_net_mask.shape[0]
        n_valid_rays = torch.sum(first_net_mask)


        dolp_render = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        aolp_render = torch.zeros((num_pixels,1)).reshape(-1).cuda()
        s0 = torch.zeros((num_pixels,1)).cuda()
        s1 = torch.zeros((num_pixels,1)).cuda()
        s2 = torch.zeros((num_pixels,1)).cuda()

        extra_stokes_size = torch.zeros((num_pixels,1)).reshape(-1).cuda()

        if n_valid_rays > 0:
            L_t2 = torch.zeros(n_valid_rays,1).cuda().float()
            L_r1 = torch.zeros(n_valid_rays,1).cuda().float()
            F1 = first_attenuate
            F2 = second_attenuate
            L_t2[~second_refract_total_reflect_mask] = self.refract_ray_tracer.sampleEnvLight(second_refract_dirs[~second_refract_total_reflect_mask])

            L_t1 = L_t2 * (1 - F2)

            L_t0 = L_t1 * (1 - F1)
            L_t0[second_refract_total_reflect_mask] = 0.5


            L_r1[~first_refract_total_reflect_mask] = self.refract_ray_tracer.sampleEnvLight(first_reflect_dirs[~first_refract_total_reflect_mask])
            L_r0 = L_r1 * F1

            L_r0_output = torch.zeros(n_rays, 1).cuda()
            L_t0_output = torch.zeros(n_rays, 1).cuda()

            L_r0_output[first_net_mask] = L_r0
            L_t0_output[first_net_mask] = L_t0

        else:
            L_r0_output = torch.zeros_like(first_net_mask).cuda().float()
            L_t0_output = torch.zeros_like(first_net_mask).cuda().float()

        self.implicit_network.train()
        points = first_pts # (n_rays,3)
        surface_points = first_pts[first_net_mask]
        sdf_output = self.implicit_network(points)[:,0]
        network_object_mask = first_net_mask
        grad_theta = self.implicit_network.gradient(surface_points)[:,0,:]
        surface_sdf_values = self.implicit_network(surface_points)[:,0:1]
        normals = self.implicit_network.gradient(points)[:,0,:]
        # surface_normal = self.implicit_network.gradient(first_pts[first_net_mask & object_mask])[:,0,:]   # 作用于几何loss。要保证同时是网络预测表面点和真实表面点，此时做的loss才有意义。
        points_refractive = second_pts
        mask_refractive = second_net_mask
        normals_refractive = self.implicit_network.gradient(points_refractive)[:,0,:]
        out_mask = ~second_refract_total_reflect_mask
        out_points = second_pts + 1 * second_refract_dirs

        attenuate = second_attenuate
        if(torch.sum(first_net_mask)>0):
            reflected_points = first_pts[first_net_mask] + 1 * first_reflect_dirs
        else:
            reflected_points = first_pts + 1 * first_reflect_dirs

        reflected_light = self.refract_ray_tracer.sampleEnvLight(first_reflect_dirs)
        out_light = self.refract_ray_tracer.sampleEnvLight(second_refract_dirs)

        ## 用颜色  ##
        # differentiable_surface_points = surface_points
        rgb_values = torch.ones_like(points).float().cuda()

        useful_mask = first_net_mask & object_mask
        fresnel_invalid_mask = None

        view = -ray_dirs[0,useful_mask]
        stokes_r, stokes_d = torch.zeros((num_pixels,4)).cuda(), torch.zeros((num_pixels,4)).cuda()

        if epoch <= self.stokes_render_weight_milestones[0] and self.use_color == True:   # 只有第一阶段用颜色
            self.rendering_network.train()
            for name, param in self.rendering_network.named_parameters():
                param.requires_grad = True
            rgb_values[useful_mask] = self.get_rbg_value(surface_points, view)
        else:
            self.rendering_network.eval()
            for name, param in self.rendering_network.named_parameters():
                param.requires_grad = False

        if epoch >= self.stokes_render_weight_milestones[0]:  # 在第二第三阶段启用stoke render
            self.incident_s0_net.train()
            self.incident_diff_net.train()
            self.incident_spec_net.train()
            for name, param in self.incident_s0_net.named_parameters():
                param.requires_grad = True
            for name, param in self.incident_diff_net.named_parameters():
                param.requires_grad = True
            for name, param in self.incident_spec_net.named_parameters():
                param.requires_grad = True
            # polar rendering
            incident_dirs = ray_dirs.squeeze(0)[useful_mask, :]

            reflected_dirs = self.refract_ray_tracer.reflection(incident_dirs, grad_theta)  # 交互点向光源
            normals_demo = normals[useful_mask]
            [s0_valid, s1_valid, s2_valid], stokes_rr, stokes_dd = self.get_render_stokes_one_channel(
                first_pts[useful_mask], -reflected_dirs,-view, normals_demo, pose, eta=1.50)

            s0[useful_mask] = s0_valid
            s1[useful_mask] = s1_valid
            s2[useful_mask] = s2_valid
            stokes_r[useful_mask] = stokes_rr[:, :, 0]
            stokes_d[useful_mask] = stokes_dd[:, :, 0]
        else:
            self.incident_s0_net.eval()
            self.incident_diff_net.eval()
            self.incident_spec_net.eval()
            for name, param in self.incident_s0_net.named_parameters():
                param.requires_grad = False
            for name, param in self.incident_diff_net.named_parameters():
                param.requires_grad = False
            for name, param in self.incident_spec_net.named_parameters():
                param.requires_grad = False

        output = {
            'points': points,  # (num_pixels,3)
            'first_net_mask': first_net_mask,
            'sdf_network': lambda x: self.implicit_network(x)[:, 0],
            'differentiable_surface_points': surface_points,  # 仅有效点 (n_valid_rays,3)
            'sdf_output': sdf_output,  # (num_pixels,3)
            'network_object_mask': network_object_mask,  # (num_pixels,) bool
            'object_mask': object_mask,  # (num_pixels,) bool
            'grad_theta': grad_theta,
            # 'surface_normal': surface_normal,
            'normals': normals,  # (num_pixels,3)
            'points_refractive': points_refractive,  # (n_valid_rays,3)
            'sdf_refractive': self.implicit_network(points_refractive)[:, 0],
            'mask_refractive': mask_refractive,  # (n_valid_rays,) bool
            'normals_refractive': normals_refractive,  # (n_valid_rays,3)
            'out_mask': out_mask,  # (n_valid_rays,)
            'out_points': out_points,  # (n_valid_rays,3)
            'out_attenuate': attenuate,  # (n_valid_days,)
            'reflected_points': reflected_points,  # (n_valid_rays,3)
            'reflected_light': reflected_light,  # (n_valid_rays,)
            'reflection_intensity': L_r0_output,  # (n_rays,) (0,1)
            'transmittance_intensity': L_t0_output,  # (n_rays,)(0,1)
            'out_light': out_light,  # (n_valid_rays,)
            'first_attenuate': first_attenuate,  # (n_valid_rays,)  == F1
            'dolp_render': dolp_render,  # (num_pixels,)
            'aolp_render': aolp_render,  # (num_pixels,)
            's0_render': s0,  # (num_pixels,)
            's1_render': s1,  # (num_pixels,)
            's2_render': s2,  # (num_pixels,)
            'rayTracing_dict': dict,
            'valid_mask': valid_mask,  # (num_pixels,)
            'bottom_normals': self.implicit_network.gradient(bottom_pts)[:, 0, :],
            'bottom_mask': bottom_mask,
            'bottom_pts': bottom_pts,
            'bottom_start_pts': bottom_start_pts,
            'extra_stokes_size': extra_stokes_size,
            'ray_dirs': ray_dirs,
            'cam_location': cam_loc,
            'near_far': near_far,
            'second_refract_dirs': second_refract_dirs,
            'first_reflect_dirs': first_reflect_dirs,
            'gt_color': gt_color,
            'gt_dolp': gt_dolp,
            'gt_aolp': gt_aolp,
            'rgb_values': rgb_values,
            'stokes_r': stokes_r,   # 用于偏振优化
            'stokes_d': stokes_d,
        }
        return output

    def straight_ray_forward(self, input ,epoch):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        (gt_color, gt_aolp, gt_dolp, gt_s0, gt_s1, gt_s2) = (input["rgb"][0], input["aolp"][0], input["dolp"][0],
                                                            input["gt_s0"][0], input["gt_s1"][0], input["gt_s2"][0])

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        for name, param in self.implicit_network.named_parameters():
            param.requires_grad = False
        with torch.no_grad():
            points, network_object_mask, dists = self.straight_ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        if self.stokes_render_weight_milestones[0] > epoch or self.stokes_render_weight_milestones[1] < epoch:
            self.implicit_network.train()
            for name, param in self.implicit_network.named_parameters():
                param.requires_grad = True

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)
        surface_normal = None

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
            # surface_normal = self.implicit_network.gradient(surface_points)[:, 0, :]


        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        # normals = self.implicit_network.gradient(points)[:, 0, :]
        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        s0, s1, s2 = None, None, None

        if differentiable_surface_points.shape[0] > 0:
            # if epoch <= self.stokes_render_weight_milestones[0] and self.use_color:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

            if self.training:
                if epoch <= self.stokes_render_weight_milestones[0]:   # 还处于第一阶段
                    self.incident_s0_net.eval()
                    self.incident_diff_net.eval()
                    self.incident_spec_net.eval()
                    for name, param in self.incident_s0_net.named_parameters():
                        param.requires_grad = False
                    for name, param in self.incident_diff_net.named_parameters():
                        param.requires_grad = False
                    for name, param in self.incident_spec_net.named_parameters():
                        param.requires_grad = False
                else:   # 第二or第三阶段
                    self.incident_s0_net.train()
                    self.incident_diff_net.train()
                    self.incident_spec_net.train()
                    for name, param in self.incident_s0_net.named_parameters():
                        param.requires_grad = True
                    for name, param in self.incident_diff_net.named_parameters():
                        param.requires_grad = True
                    for name, param in self.incident_spec_net.named_parameters():
                        param.requires_grad = True
                    reflected_dirs = ray_dirs.squeeze(0)[surface_mask, :]  # 相机到交互点
                    # normals_demo = self.implicit_network.gradient(surface_points)[:, 0, :]
                    incident_dirs = self.refract_ray_tracer.reflection(reflected_dirs, surface_normal)  # 交互点向光源
                    [s0, s1, s2] = self.get_render_stokes(differentiable_surface_points, -incident_dirs, reflected_dirs,
                                                          surface_normal, pose)

        output = {
            'points': points,
            'differentiable_surface_points': differentiable_surface_points,
            'sdf_network': lambda x: self.implicit_network(x)[:, 0],
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,  # 真实mask
            'grad_theta': grad_theta,
            # 'surface_normal': surface_normal,  # 表面处有效的normal，用来做几何loss
            # 'normals': normals,  # 全部采样点的法向量，用来可视化。
            'gt_color': gt_color,
            'gt_dolp': gt_dolp,
            'gt_aolp': gt_aolp,
            's0_render': s0,
            's1_render': s1,
            's2_render': s2,
            's0_gt': gt_s0,
            's1_gt': gt_s1,
            's2_gt': gt_s2,
        }

        return output

    def reflection_polarized_demo(self, pose, incident_dirs, reflection_dirs, normals, reflection_light):
        normals = self.mueller.normalize(normals)
        # pose: (bs,4,4)
        if pose.shape[1] == 7:  # In case of quaternion vector representation
            R = rend_util.quat_to_rot(pose[:, :4]).squeeze(dim=0)

        else:  # In case of pose matrix representation
            R = pose[0, 0:3, 0:3]

        n_rays = incident_dirs.shape[0]
        eta = (torch.ones(n_rays, 1) * 1.52).cuda().reshape(-1)

        stokes_init = torch.zeros((n_rays, 4, 1)).cuda().float()
        stokes_init[:, 0, 0] = reflection_light.reshape(-1)  # 初始化一个无偏光的stokes

        w_o_hat = reflection_dirs
        w_i_hat = -incident_dirs

        s_axis_in = self.mueller.normalize(torch.cross(normals, -w_o_hat))
        s_axis_out = self.mueller.normalize(torch.cross(normals, w_i_hat))

        cos_theta_i = self.mueller.dot(normals, w_o_hat)

        weight = self.mueller.specular_reflection(cos_theta_i, eta)  # (n_rays,4,4)

        weight = self.mueller.rotate_mueller_basis(weight,
                                                   -w_o_hat, s_axis_in, self.mueller.stokes_basis(-w_o_hat),
                                                   w_i_hat, s_axis_out, self.mueller.stokes_basis(w_i_hat)
                                                   )
        stokes = torch.bmm(weight, stokes_init)

        return stokes

    def polarized_render(self, pose, incident_dirs, reflection_dirs, normals,
                         stokes_r_init, stokes_d_init, eta=1.52):

        normals = self.mueller.normalize(normals)

        n_rays = incident_dirs.shape[0]
        eta = (torch.ones(n_rays, 1) * eta).cuda().reshape(-1)  # 物体折射率/空气折射率

        w_o_hat = reflection_dirs
        w_i_hat = -incident_dirs

        s_axis_in = self.mueller.normalize(torch.cross(normals, -w_o_hat))
        s_axis_out = self.mueller.normalize(torch.cross(normals, w_i_hat))

        cos_theta_i = self.mueller.dot(normals, w_o_hat)

        # 首先计算反射stokes
        weight_r = self.mueller.specular_reflection(cos_theta_i, eta)  # (n_rays,4,4)
        if weight_r == None:
            return None
        weight_r = self.mueller.rotate_mueller_basis(weight_r,
                                                     #  in_forward, in_basis_current, in_basis_target
                                                     -w_o_hat, s_axis_in, self.mueller.stokes_basis(-w_o_hat),
                                                     w_i_hat, s_axis_out, self.mueller.stokes_basis(w_i_hat)
                                                     )
        stokes_r = torch.bmm(weight_r, stokes_r_init)  # 得到R0*Mr*Ri*S0

        # 其次试试计算透射/漫反射stokes
        weight_d = self.mueller.specular_transmission(cos_theta_i, eta)  # (n_rays,4,4)
        if weight_d == None:
            return None
        weight_d = self.mueller.rotate_mueller_basis(weight_d,
                                                     -w_o_hat, s_axis_in, self.mueller.stokes_basis(-w_o_hat),
                                                     w_i_hat, s_axis_out, self.mueller.stokes_basis(w_i_hat)
                                                     )
        stokes_d = torch.bmm(weight_d, stokes_d_init)  # 得到R0*Mr*Ri*S0

        return stokes_r, stokes_d
        # return stokes_r

    def stokes_to_camera(self, pose, stokes, incident_dirs):
        if pose.shape[1] == 7:  # In case of quaternion vector representation
            R = rend_util.quat_to_rot(pose[:, :4]).squeeze(dim=0)

        else:  # In case of pose matrix representation
            R = pose[0, 0:3, 0:3]
        n_rays, _, _ = stokes.shape

        vertical_cam = torch.tensor([0.0, -1.0, 0.0]).cuda().reshape(-1, 1)  # (3,1)
        vertical = torch.mm(R, vertical_cam).reshape(-1).unsqueeze(0)
        vertical = vertical.repeat(n_rays, 1)  # (n_rays,3)
        current_basis = self.mueller.stokes_basis(-incident_dirs)
        target_basis = torch.cross(incident_dirs, vertical)
        M_cam = self.mueller.rotate_stokes_basis(-incident_dirs, current_basis, target_basis)
        stokes_cam = torch.bmm(M_cam, stokes)
        return stokes_cam

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]

        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)
        return rgb_vals

    def get_render_stokes_one_channel(self, surface_points, view_dirs, cam_dirs, normal, pose, eta=1.52):
        """
        view_dirs:  光源指向交互点
        cam_dirs:  相机指向交互点
        """
        # ------------MLP-------------
        # inputs with incident directions
        inp_w_dir = torch.cat([surface_points, view_dirs], dim=1)  # (bs, 3+3)

        incident_s0 = self.incident_s0_net(inp_w_dir)  # (bs, 2)
        incident_s0_spec = incident_s0[:, :1]
        incident_s0_diff = incident_s0[:, 1:]

        # estimate incident specular s1 and s2
        incident_spec = self.incident_spec_net(inp_w_dir)  # (bs, 6)
        incident_s1_spec = incident_spec[:, :1]  # (bs, 3)
        incident_s2_spec = incident_spec[:, 1:]  # (bs, 3)

        s3 = torch.zeros((len(incident_spec)), 1).cuda()
        incident_spec_stokes = torch.cat([incident_s0_spec[:, 0][:, None],
                                               incident_s1_spec[:, 0][:, None],
                                               incident_s2_spec[:, 0][:, None], s3], dim=1)

        # estimate incident diffuse s1
        incident_diff = self.incident_diff_net(inp_w_dir)  # (bs, 3)
        incident_s1_diff = incident_diff[:, :1]  # (bs, 3)
        incident_s2_diff = incident_diff[:, 1:]  # (bs, 3)
        incident_diff_stokes = torch.cat([incident_s0_diff[:, 0][:, None],
                                               incident_s1_diff[:, 0][:, None],
                                               incident_s2_diff[:, 0][:, None], s3], dim=1)

        stokes_r, stokes_d = self.polarized_render(pose, cam_dirs, -view_dirs,
                                              normal,
                                              incident_spec_stokes[:, :, None],
                                              incident_diff_stokes[:, :, None],
                                              eta=eta)  # (n_valid_rays,)
        stokes_render = stokes_r + stokes_d

        if stokes_render == None:
            # self.log_invalid_event()
            return [None, None, None]
        # 把计算得到的“stokes_render”映射到相机坐标系
        stokes_cam = self.stokes_to_camera(pose, stokes_render, cam_dirs)[:, :3, 0]  # (n_valid_rays,3)
        stokes_r = self.stokes_to_camera(pose, stokes_r, cam_dirs)  # (n_valid_rays,3)
        stokes_d = self.stokes_to_camera(pose, stokes_d, cam_dirs)  # (n_valid_rays,3)

        return [stokes_cam[:,0:1], stokes_cam[:,1:2], stokes_cam[:,2:3]], stokes_r, stokes_d


    def get_render_stokes(self, surface_points, view_dirs, cam_dirs, normal, pose, eta=1.52):
        """
        view_dirs:  光源指向交互点
        cam_dirs:  相机指向交互点
        """
        # ------------MLP-------------
        # inputs with incident directions
        inp_w_dir = torch.cat([surface_points, view_dirs], dim=1)  # (bs, 3+3)

        incident_s0 = self.incident_s0_net(inp_w_dir)  # (bs, 3)

        # estimate incident specular s1 and s2
        incident_spec = self.incident_spec_net(inp_w_dir)  # (bs, 6)
        incident_s1_spec = incident_spec[:, :3] * incident_s0  # (bs, 3)
        incident_s2_spec = incident_spec[:, 3:] * incident_s0  # (bs, 3)

        s3 = torch.zeros((len(incident_spec)), 1).cuda()
        incident_spec_stokes_rgb = [torch.cat([incident_s0[:, i][:, None],
                                               incident_s1_spec[:, i][:, None],
                                               incident_s2_spec[:, i][:, None], s3], dim=1) for i in range(3)]

        # estimate incident diffuse s1
        incident_diff = self.incident_diff_net(inp_w_dir)  # (bs, 3)
        incident_s1_diff = incident_diff[:, :3] * incident_s0  # (bs, 3)
        incident_s2_diff = incident_diff[:, 3:] * incident_s0  # (bs, 3)
        incident_diff_stokes_rgb = [torch.cat([incident_s0[:, i][:, None],
                                               incident_s1_diff[:, i][:, None],
                                               incident_s2_diff[:, i][:, None], s3], dim=1) for i in range(3)]

        stokes_cam = []
        for incident_spec_stokes, incident_diff_stokes in zip(incident_spec_stokes_rgb, incident_diff_stokes_rgb):
            stokes_render = self.polarized_render(pose, cam_dirs, -view_dirs,
                                                  normal,
                                                  incident_spec_stokes[:, :, None],
                                                  incident_diff_stokes[:, :, None],
                                                  eta=eta)  # (n_valid_rays,)
            if stokes_render == None:
                # self.log_invalid_event()
                return [None, None, None]
            # 把计算得到的“stokes_render”映射到相机坐标系
            stokes_cam.append(self.stokes_to_camera(pose, stokes_render, cam_dirs)[:, :3, 0])  # (n_valid_rays,3)

        r1, g1, b1 = stokes_cam[0].split(1, dim=1)  # dim=1 表示沿着列（即通道）分割
        r2, g2, b2 = stokes_cam[1].split(1, dim=1)
        r3, g3, b3 = stokes_cam[2].split(1, dim=1)

        # 组合 R、G、B 通道
        S0 = torch.cat((r1, r2, r3), dim=1)  # S0  # (n_valid_rays,3)
        S1 = torch.cat((g1, g2, g3), dim=1)  # S1  # (n_valid_rays,3)
        S2 = torch.cat((b1, b2, b3), dim=1)  # S2  # (n_valid_rays,3)

        return [S0, S1, S2]

    def log_invalid_event(self):
        import os
        filename = './log1.txt'
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write("catch a potential nan error\n")
        else:
            # 如果文件已存在，打开文件并追加"true"
            with open(filename, 'a') as file:
                file.write("catch a potential nan error\n")

