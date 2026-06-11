import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
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
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

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


    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)
        normal = None

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

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            normal = self.implicit_network.gradient(points).squeeze(1)

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'normal': normal,
            'differentiable_surface_points': points[network_object_mask&object_mask],
            'sdf_network': lambda x: self.implicit_network(x)[:, 0],
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals
