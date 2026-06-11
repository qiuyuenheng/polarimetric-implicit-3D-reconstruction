import torch
import torch.nn as nn
from utils import rend_util
from polarTracing.mueller import Mueller

class Straight_RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_secant_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):
        # import time
        # t0 = time.time()

        batch_size, num_pixels, _ = ray_directions.shape

        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        network_object_mask = (acc_start_dis < acc_end_dis)
        # t1 = time.time()
        # delta = t1 - t0
        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]
        # t2 = time.time()
        # delta2 = t2 - t1

        # if network_object_mask.sum() == 20480:
        #     k=1
        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
              .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        print('----------------------------------------------------------------')

        if not self.training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)

        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        out_mask = ~object_mask & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect
        # t3 = time.time()
        # delta3 = t3 - t2
        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist
        # t4 = time.time()
        # delta4 = t4 - t3
        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)


            import time
            t0 = time.time()
            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])
            t1 = time.time()
            delta = t1-t0

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''
        # import time
        # t1 = time.time()
        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        # t2 = time.time()
        # delta2 = t2 - t1
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))
        # t3 = time.time()
        # delta3 = t3 - t2
        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]
        # t4 = time.time()
        # delta4 = t4 - t3
        return min_mask_points, min_mask_dist


class Refract_RayTracing(nn.Module):
    def __init__(
            self,
            n_secant_steps,
            n_steps,
            eta1,
            eta2):
        super().__init__()
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps
        self.eta1 = eta1
        self.eta2 = eta2
        self.mueller = Mueller()



    def forward(self,sdf,normal_fn,cam_loc,object_mask,ray_directions):
        # cam_loc: (bs,3)
        # object_mask: (n_rays,)
        # ray_directions: (bs,n_rays,3)

        batch_size, num_pixels, _ = ray_directions.shape
        assert batch_size == 1, 'error: batch size bigger than 1'
        self.cam_loc = cam_loc
        cam_loc = cam_loc.repeat(num_pixels,1) #(n_rays,3)

        ray_directions = ray_directions.reshape(num_pixels,3)

        stokes = None
        dict = None
        valid_mask = None
        near_far = None

        # 有mask情况下的处理：（通常使用此种方法）
        if(torch.sum(object_mask) > 0):
            with (torch.no_grad()):
                first_pts,first_pts_neg,first_net_mask,first_net_mask_neg,first_dists,first_dists_neg,first_sphere_points,near_far = self.get_intersection_points(sdf = sdf,ray_directions = ray_directions,start_points = cam_loc,mask_intersect = object_mask,isInternal = False,isOutSphere=True)

            if(torch.sum(first_net_mask) > 0):
                normals_first = normal_fn(first_pts[first_net_mask]) # (n_valid_rays,3)
                with torch.no_grad():
                    first_refract_dirs, first_reflect_dirs, first_attenuate, first_refract_total_reflect_mask = self.interaction(ray_directions[first_net_mask],normals_first,self.eta1,self.eta2)  # 与n_valid_rays保持相同

                    second_pts,_,second_net_mask,_, second_dists,_,second_sphere_points,_ = self.get_intersection_points(sdf = sdf,
                                                                                                                 ray_directions = first_refract_dirs,
                                                                                                                 start_points = first_pts[first_net_mask],
                                                                                                                 mask_intersect = ~first_refract_total_reflect_mask,
                                                                                                                 isInternal = True
                                                                                                                 )

                second_refract_dirs = torch.zeros_like(first_refract_dirs).cuda()
                second_reflect_dirs = torch.zeros_like(first_refract_dirs).cuda()
                second_attenuate = torch.zeros_like(first_attenuate).cuda()
                second_refract_total_reflect_mask = torch.zeros_like(first_refract_total_reflect_mask).cuda()


                if(torch.sum(second_net_mask) > 0):

                    normals_second = normal_fn(second_pts) # (n_valid_rays,3)
                    with torch.no_grad():
                        second_refract_dirs[second_net_mask,:],second_reflect_dirs[second_net_mask,:],second_attenuate[second_net_mask], second_refract_total_reflect_mask[second_net_mask] = self.interaction(
                            first_refract_dirs[second_net_mask,:],
                            -normals_second[second_net_mask,:],self.eta2,self.eta1)

            else:
                second_pts = torch.zeros_like(first_pts).cuda()
                second_net_mask = torch.zeros_like(first_net_mask).cuda()
                second_dists = torch.zeros_like(first_net_mask).cuda()

                first_refract_dirs = torch.zeros_like(first_pts).cuda()
                first_reflect_dirs = torch.zeros_like(first_pts).cuda()
                first_attenuate = torch.zeros_like(first_net_mask).cuda()
                first_refract_total_reflect_mask = torch.zeros_like(first_net_mask).cuda()

                second_refract_dirs = torch.zeros_like(first_pts).cuda()
                second_reflect_dirs = torch.zeros_like(first_pts).cuda()
                second_attenuate = torch.zeros_like(first_net_mask).cuda()
                second_refract_total_reflect_mask = torch.zeros_like(first_net_mask).cuda()
                near_far = torch.zeros((first_pts.shape[0],2)).cuda()

        # 无mask情况下的处理：
        else:
            # object_mask为0的时候处理
            first_pts = torch.zeros_like(ray_directions).cuda() # (n_rays)
            first_net_mask = torch.zeros_like(object_mask).cuda()
            first_dists = torch.zeros_like(object_mask).cuda()

            second_pts = torch.zeros_like(first_pts).cuda()
            second_net_mask = torch.zeros_like(first_net_mask).cuda()
            second_dists = torch.zeros_like(first_net_mask).cuda()

            first_refract_dirs = torch.zeros_like(first_pts).cuda()
            first_reflect_dirs = torch.zeros_like(first_pts).cuda()
            first_attenuate = torch.zeros_like(first_net_mask).cuda()
            first_refract_total_reflect_mask = torch.zeros_like(first_net_mask).cuda()

            second_refract_dirs = torch.zeros_like(first_pts).cuda()
            second_reflect_dirs = torch.zeros_like(first_pts).cuda()
            second_attenuate = torch.zeros_like(first_net_mask).cuda()
            second_refract_total_reflect_mask = torch.zeros_like(first_net_mask).cuda()
            near_far = torch.zeros((first_pts.shape[0],2)).cuda()

        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}.'
              .format(first_net_mask.sum(), len(first_net_mask)))
        print('----------------------------------------------------------------')

        # with torch.no_grad():
        #     bottom_start_pts,bottom_pts,bottom_mask = self.get_bottom_points(sdf,num_pixels)
        bottom_pts = torch.zeros_like(first_pts).cuda()
        bottom_mask = torch.zeros_like(first_pts).cuda()
        bottom_start_pts = torch.zeros_like(first_pts).cuda()
        return first_pts, \
               first_net_mask, \
               first_dists, \
                second_pts, \
                second_net_mask, \
                second_dists,\
                first_refract_dirs,\
                first_reflect_dirs,\
                first_attenuate,\
                first_refract_total_reflect_mask,\
                second_refract_dirs,\
                second_reflect_dirs,\
                second_attenuate,\
                second_refract_total_reflect_mask,\
                stokes,\
                dict, \
                valid_mask, \
                bottom_start_pts,\
                bottom_pts,\
                bottom_mask, near_far








    def interaction(self,v_in,normals,eta1,eta2):
        refract_dirs, attenuate, total_reflect_mask = self.refraction(v_in,normals,eta1,eta2)
        reflect_dirs = self.reflection(v_in,normals)
        return refract_dirs,reflect_dirs,attenuate,total_reflect_mask








    def refraction(self,v_in,normals,eta1,eta2):
        refractive_index = eta1 / eta2
        v_in = v_in/torch.linalg.norm(v_in,ord=2,dim=1).reshape(-1,1)  # 归一化
        normals = normals/torch.linalg.norm(normals,ord=2,dim=1).reshape(-1,1)  # 归一化

        dt = -torch.sum(v_in * normals, dim=1).reshape(-1,1) # 入射方向和法向量的cos
        discriminant = 1.0 - refractive_index * refractive_index * (1 - dt ** 2)  # (num_rays,)# c2**2
        mask_intersect = discriminant > 0  # (num_rays,)  full internal reflection rays
        dt = dt.reshape(-1,1)
        mask_intersect = mask_intersect.reshape(-1)
        refracted_directions = torch.zeros_like(v_in).cuda().float()
        refracted_directions[mask_intersect,:] = (refractive_index * (v_in[mask_intersect,:] + normals[mask_intersect,:] * dt[mask_intersect])
                                                  - normals[mask_intersect,:] * torch.sqrt(discriminant[mask_intersect]))
        total_reflect_mask = ~mask_intersect

        cos_theta_i = -torch.sum(v_in * normals, dim=1).reshape(-1,1)
        cos_theta_t = -torch.sum(refracted_directions * normals,dim=1).reshape(-1,1)

        e_s = (cos_theta_t * eta2 - cos_theta_i * eta1) / \
                torch.clamp(cos_theta_t * eta2 + cos_theta_i * eta1, min=1e-10 )
        e_p = (cos_theta_t * eta1 - cos_theta_i * eta2) / \
                torch.clamp(cos_theta_t * eta1 + cos_theta_i * eta2, min=1e-10 )
        attenuate = torch.clamp(0.5 * (e_s * e_s + e_p * e_p), 0, 1)

        return refracted_directions,attenuate,total_reflect_mask


    def reflection(self,v_in,normals):
        # checked
        v_in = v_in/torch.linalg.norm(v_in,ord=2,dim=1).reshape(-1,1)
        normals = normals/torch.linalg.norm(normals,ord=2,dim=1).reshape(-1,1)

        reflected_dir = v_in - 2 * self.dot(v_in,normals) * normals   # 从交互点向外
        return reflected_dir


    def sampleEnvLight(self,dirs):
        n_rays,_ = dirs.shape
        dirs = dirs/torch.linalg.norm(dirs,ord=2,dim=1).reshape(-1,1)

        delta_angle = 10
        cam_dir = self.cam_loc.repeat(n_rays,1)
        cam_dir = self.mueller.normalize(cam_dir)
        delta_angle = torch.tensor((90 - delta_angle) / 180.0 * torch.pi).cuda()
        delta_value = torch.cos(delta_angle)
        values = torch.sum(dirs * cam_dir,dim=1).reshape(-1,1)
        light_mask = values > delta_value
        light = torch.zeros(dirs.shape[0], 1).cuda()
        light[light_mask] = 1.0
        light[~light_mask] = 0.1



        return light

    def dot(self,vec_1,vec_2):
        return torch.sum(vec_1 * vec_2,dim = 1).reshape(-1,1)

    def get_intersection_points(self,sdf, ray_directions,start_points,mask_intersect,isInternal,isOutSphere=False):
        sdf_values = sdf(start_points[mask_intersect])
        mask_idx = torch.nonzero(mask_intersect).flatten()
        sampler_out_mask = sdf_values > 0
        sampler_internal_mask = sdf_values < 0
        if(isInternal):
            mask_intersect[mask_idx[sampler_out_mask]] = False
        else:
            mask_intersect[mask_idx[sampler_internal_mask]] = False

        sdf_values = sdf(start_points[mask_intersect])
        sdf_interal_mask = sdf_values < 0 # 获取所有在内部的mask
        sdf_out_mask = sdf_values > 0
        if(isInternal):
            pass
        else:
            assert torch.sum(sdf_interal_mask) == 0, 'error start points sdf values!'

        ray_directions = ray_directions/torch.linalg.norm(ray_directions,ord=2,dim=1).reshape(-1,1)

        n_rays,_ = start_points.shape
        sampler_min_max = torch.zeros((n_rays,2)).cuda()

        # sphere_intersections  就是near 和 far
        sphere_intersections,sphere_mask_intersect = rend_util.get_sphere_intersections_single(start_points,ray_directions)

        sampler_mask = mask_intersect & sphere_mask_intersect
        sampler_mask = sampler_mask.reshape(-1)

        pts = start_points + sphere_intersections[:,0].reshape(-1,1) * ray_directions  # 5120条光线的最近点pts和最远点pts2
        pts2 = start_points + sphere_intersections[:,1].reshape(-1,1) * ray_directions
        r0 = torch.abs(1.0 - torch.linalg.norm(pts,ord=2,dim=1))
        r1 = torch.abs(1.0 - torch.linalg.norm(pts2,ord=2,dim=1))
        if(isOutSphere):
            sphere_intersections = sphere_intersections
            values,indices = torch.max(sphere_intersections,dim=1)
        else:
            values,indices = torch.min(torch.stack((r0,r1),dim=1),dim=1)

        sphere_distances = torch.zeros((n_rays,1)).cuda()  # (n_rays,1)
        r0_mask = (indices==0).reshape(-1)
        r1_mask = (indices==1).reshape(-1)
        sphere_distances[r0_mask,0] = sphere_intersections[r0_mask,0]
        sphere_distances[r1_mask,0] = sphere_intersections[r1_mask,1]

        sphere_points = start_points + sphere_distances.reshape(-1,1) * ray_directions

        sampler_min_max[:,1] = sphere_distances.reshape(-1)


        # 下面这个函数通过输入相机坐标、方向，利用sdf输出结果，预测与物体的表面点位置。
        # 不太了解sampler_pts,sampler_pts_neg的具体区别，他们两个的值相差不大。猜测neg代表离相机更近的采样点。但他们都代表交点。
        # sampler_dists,sampler_dists_neg代表从相机出发到采样点的距离。
        sampler_pts,sampler_pts_neg,sampler_net_obj_mask,sampler_dists,sampler_dists_neg = self.ray_sampler(sdf=sdf,
                                                                          interact_points=start_points,
                                                                          ray_directions=ray_directions,
                                                                          sampler_min_max=sampler_min_max,
                                                                          sampler_mask = sampler_mask,
                                                                          isInternal=isInternal
                                                                          )

        mask_left_out = ~sampler_net_obj_mask & ~sphere_mask_intersect
        # 我的理解是，mask_left_out表示真正有效的光线。因为在ray_sampler代码中，使用输出的一条光线上100个采样点的，最小的sdf对应的采样点。
        # 但有些时候，最小的sdf对应的采样点，这个sdf值都大于0，说明这条光线上根本没有穿过物体。这与事实情况相悖，事实情况是，我这可是自己手工
        # 标注的mask，理论上，它肯定是穿过了物体的，一定会有一个sdf是0。（突然意识到会不会是100个采样点数量不够？）
        if mask_left_out.sum() > 0:
            cam_left_out = start_points[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            sampler_dists[mask_left_out] = -torch.bmm(rays_left_out.view(-1,1,3),cam_left_out.view(-1,3,1)).squeeze()
            sampler_dists_neg[mask_left_out] = -torch.bmm(rays_left_out.view(-1,1,3),cam_left_out.view(-1,3,1)).squeeze()
            sampler_pts[mask_left_out] = cam_left_out + sampler_dists[mask_left_out].unsqueeze(1) * rays_left_out
            sampler_pts_neg[mask_left_out] = cam_left_out + sampler_dists_neg[mask_left_out].unsqueeze(1) * rays_left_out

        mask = ~sampler_net_obj_mask & sphere_mask_intersect
        if mask.sum() > 0:
            min_dis = sphere_intersections[:,0]
            max_dis = sphere_intersections[:,1]

            min_mask_points,min_mask_dist = self.minimal_sdf_points(n_rays,sdf,start_points,ray_directions,mask,min_dis,max_dis)
            sampler_pts[mask] = min_mask_points
            sampler_pts_neg[mask] = min_mask_points

            sampler_dists_neg[mask] = min_mask_dist
            sampler_dists[mask] = min_mask_dist

        sampler_net_obj_mask_neg = sampler_net_obj_mask.clone()
        sampler_sdf = sdf(sampler_pts)
        sampler_sdf_neg = sdf(sampler_pts_neg)
        mask_idx = torch.nonzero(sampler_net_obj_mask).flatten()
        sampler_out_mask = sampler_sdf[sampler_net_obj_mask] > 0
        sampler_internal_mask = sampler_sdf[sampler_net_obj_mask] < 0
        sampler_out_mask_neg = sampler_sdf_neg > 0
        sampler_internal_mask_neg = sampler_sdf_neg < 0

        if(isInternal):
            sampler_net_obj_mask[mask_idx[sampler_internal_mask]] = False
            sampler_net_obj_mask_neg = sampler_net_obj_mask_neg & sampler_internal_mask_neg

        else:
            sampler_net_obj_mask[mask_idx[sampler_out_mask]] = False
            sampler_net_obj_mask_neg = sampler_net_obj_mask_neg & sampler_out_mask_neg

        return sampler_pts,sampler_pts_neg,sampler_net_obj_mask,sampler_net_obj_mask_neg,sampler_dists,sampler_dists_neg,sphere_points,sphere_intersections

    def ray_sampler(self, sdf, interact_points, ray_directions, sampler_min_max, sampler_mask,isInternal):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''
        # ray_directions: (n_rays,3) float
        # interact_points: (n_rays,3) float
        # sampler_min_max: (n_rays,2) float distances
        # sampler_mask: (n_rays,) bool

        sdf_values = sdf(interact_points[sampler_mask])
        sdf_interal_mask = sdf_values < 0
        if(isInternal):
            # print('all:',torch.sum(sdf_out_mask) + torch.sum(sdf_interal_mask))
            pass
            # print('torch.sum(sdf_out_mask):',torch.sum(sdf_out_mask))
            # assert torch.sum(sdf_out_mask) == 0, 'error start points sdf values!'
        else:
            assert torch.sum(sdf_interal_mask) == 0, 'error start points sdf values!'

        n_rays,_ = interact_points.shape
        sampler_pts = torch.zeros(n_rays, 3).cuda().float()
        sampler_dists = torch.zeros(n_rays).cuda().float()
        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1,-1) #(1,100)
        pts_intervals = sampler_min_max[:,0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:,1] - sampler_min_max[:,0]).unsqueeze(-1)  # (n_rays,100)
        points = interact_points.reshape(n_rays,1,3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(1)  # (n_rays,100,3)

        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points[sampler_mask,:,:] # (n_valid_rays,100,3)
        pts_intervals = pts_intervals[sampler_mask,:]  # (n_valid_rays,100)

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1,3),100000,dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1,self.n_steps)  # (n_valid_rays,100)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # (n_valid_rays,100)
        if(isInternal):
            sampler_pts_ind = torch.argmax(tmp, -1)

        else:
            sampler_pts_ind = torch.argmin(tmp, -1)

        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]),sampler_pts_ind,:]  # 计算出一条光线上第一个sdf<0的点
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]),sampler_pts_ind]  # 上述点相对于相机的distant



        if(isInternal):
            net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]),sampler_pts_ind] > 0) # (n_valid_rays,)
        else:
            net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]),sampler_pts_ind] < 0) # (n_valid_rays,)

        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False


        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        sampler_pts_neg = sampler_pts.clone()
        sampler_dists_neg = sampler_dists.clone()
        if(n_secant_pts > 0):
            # Get secant z predictions
            # z_high就是相机到由sdf预测的表面的距离
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts] # (n_valid_rays2,) t_(i+1)
            # sdf_high就是预测的表面点，对应的sdf值。这个值是之前已经计算过了，选了一个最近的再拿出来再次利用。
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]),sampler_pts_ind][secant_pts] # (n_valid_rays2,) f(t_(i+1))
            # z_low也是距离，会比z_high小一点点，可以理解成相对于z_high,更近相机的1个单位的距离
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1] # (n_valid_rays2,) t_i
            # sdf_low同理，比sdf_high更近一个单位，所以sdf_low显正值（对应的采样点在物体外部），sdf_high显负值（对应的采样点在物体内部）
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1] # (n_valid_rays2,) f(t_i)
            # loc_secant表示采样点的坐标。其对应的采样点和z_high和sdf_high对应的采样点，三者一样，都在物体内部
            loc_secant = interact_points[mask_intersect_idx[secant_pts]] # (n_valid_rays2,3)
            # ray_directions_secant表示这个采样点的方向。
            ray_directions_secant = ray_directions.reshape(-1,3)[mask_intersect_idx[secant_pts]]
            z_pred_secant,z_pred_neg_secant = self.secant(sdf_low,sdf_high,z_low,z_high,loc_secant,ray_directions_secant,sdf,isInternal)

            sampler_pts[mask_intersect_idx[secant_pts]] = loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_pts_neg[mask_intersect_idx[secant_pts]] = loc_secant + z_pred_neg_secant.unsqueeze(-1) * ray_directions_secant

            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant
            sampler_dists_neg[mask_intersect_idx[secant_pts]] = z_pred_neg_secant

        return sampler_pts,sampler_pts_neg, sampler_net_obj_mask, sampler_dists,sampler_dists_neg
    def secant(self,sdf_low, sdf_high,z_low, z_high, loc_secant, ray_directions, sdf ,isInternal):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''
        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps + 1):
            p_mid = loc_secant + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)

            if(isInternal):
                ind_low = sdf_mid < 0
            else:
                ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
            if(isInternal):
                ind_high = sdf_mid > 0
            else:
                ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low



        z_pred = z_high
        z_pred_neg = z_low

        return z_pred,z_pred_neg

    def minimal_sdf_points(self, num_pixels, sdf, start_points, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = start_points[mask,:]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist

    def normalize(self,a):
        # a: (n_rays,3)
        return a / torch.linalg.norm(a,ord = 2,dim=1).reshape(-1,1)

    def get_bottom_points(self,sdf,num_pixels):

        up = torch.tensor([0.0,0.0,-1.0]).reshape(-1,3).cuda().float()
        i = self.mueller.stokes_basis(up).reshape(-1,3)
        j = self.mueller.normalize(torch.cross(i, up)).reshape(-1,3)
        k = up
        phi = torch.randint(0,360,(num_pixels,1)).cuda().float()
        theta = torch.randint(0,80,(num_pixels,1)).cuda().float()
        phi = phi / 180.0 * torch.pi
        theta = theta / 180.0 * torch.pi
        x = 1 * torch.sin(theta) * torch.cos(phi)
        y = 1 * torch.sin(theta) * torch.sin(phi)
        z = 1 * torch.cos(theta)
        sample_dirs = x * i + y * j + z * k
        sample_dirs = self.mueller.normalize(sample_dirs)

        start_pts = torch.tensor([0.0,0.0,0.9]).cuda().float().repeat(num_pixels,1)

        first_pts, first_pts_neg, first_net_mask, first_net_mask_neg, first_dists, first_dists_neg, first_sphere_points = self.get_intersection_points(
            sdf=sdf,
            ray_directions=sample_dirs,
            start_points=start_pts,
            mask_intersect=torch.ones((num_pixels,)).cuda().bool(),
            isInternal=False,
            isOutSphere=False
            )

        return start_pts,first_pts,first_net_mask