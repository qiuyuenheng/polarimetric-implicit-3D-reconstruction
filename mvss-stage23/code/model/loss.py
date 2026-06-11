import torch
from torch import nn
from torch.nn import functional as F
import utils

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, aolp_render_weight_init, alpha, stokes_render_weight_init, use_refract_ray=False):
        super().__init__()

        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.aolp_render_weight = 0.0
        self.aolp_render_weight_init = aolp_render_weight_init
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l1_loss_c = nn.L1Loss(reduction='none')  # color
        self.use_refract_ray = use_refract_ray
        self.stokes_render_weight = 0.0
        self.stokes_render_weight_init = stokes_render_weight_init
        self.color_render_weight = 1.0


    def get_aolp_render_loss(self,stokes_r, aolp_gt, network_object_mask,
                             object_mask, valid_mask, percentage, transparent_confidence):
        mask = network_object_mask & object_mask & valid_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        aolp_render = 0.5 * torch.atan2(stokes_r[:,2], stokes_r[:,1] + 1e-6)
        aolp_predict = torch.remainder(aolp_render, torch.pi) # (num_pixels,) # (0,pi)
        # aolp_predict = aolp_predict[network_object_mask & object_mask & valid_mask]

        temp = aolp_gt.reshape(-1).clone()
        temp[network_object_mask & object_mask] = aolp_predict[network_object_mask & object_mask]
        aolp_predict = temp[mask]

        aolp_gt = aolp_gt.reshape(-1)[mask] # (0,pi)

        loss = torch.abs(aolp_predict - aolp_gt)
        loss_invalid_mask = (loss > torch.pi / 6.0)
        percentage = percentage[mask].reshape(-1)
        transparent_confidence = transparent_confidence[mask].reshape(-1)
        loss_weighted= loss * percentage * (1 - transparent_confidence)
        loss_weighted[loss_invalid_mask] = 0.0
        loss_weighted = loss_weighted.sum()
        loss_weighted = loss_weighted / float(loss.shape[0])
        return loss_weighted

    def get_aolp_loss(self,pred_normal,aolp_gt, dolp_gt):
        """aolp已经是0到pi.
            pred_normal  (num_ray,3)
            aolp_gt  (num_ray,)

        """

        aolp_0 = aolp_gt + torch.pi / 2
        aolp_1 = aolp_gt - torch.pi / 2
        aolp_0 = torch.remainder(aolp_0, torch.pi * 2)
        aolp_1 = torch.remainder(aolp_1, torch.pi * 2)

        # mask_invalid_pixels = torch.all(mask_tensor < 255, dim=1)
        y = pred_normal[:, 1]
        x = pred_normal[:, 0]
        phi = torch.atan2(y, x)  # (batchSize,H,W) (-pi,pi)
        phi = torch.remainder(phi, torch.pi * 2)

        error_0 = torch.min(torch.abs(phi - aolp_0),
                            torch.pi * 2 - torch.abs(phi - aolp_0))  # (bs,H,W)
        error_1 = torch.min(torch.abs(phi - aolp_1),
                            torch.pi * 2 - torch.abs(phi - aolp_1))
        error = torch.min(error_0, error_1)
        error = error * dolp_gt
        return error.sum()/error.shape[0]


    def get_aolp_loss_gb(self, pred_normal, aolp_gt, dolp_gt):
        """
            aolp已经是0到pi.
            pred_normal  (num_ray,3)
            aolp_gt  (num_ray,)

        """
        if len(pred_normal) == 0:
            return torch.tensor([0.0]).cuda()

        aolp_0 = aolp_gt
        aolp_1 = aolp_gt + torch.pi / 2 * 1
        aolp_2 = aolp_gt + torch.pi / 2 * 2
        aolp_3 = aolp_gt + torch.pi / 2 * 3
        aolp_0 = torch.remainder(aolp_0, torch.pi * 2)  # 真实aolp算出来的4个方位角
        aolp_1 = torch.remainder(aolp_1, torch.pi * 2)
        aolp_2 = torch.remainder(aolp_2, torch.pi * 2)
        aolp_3 = torch.remainder(aolp_3, torch.pi * 2)

        y = pred_normal[:, 1]
        x = pred_normal[:, 0]
        phi = torch.atan2(y, x)  # 预测的normal算出来的方位角
        phi = torch.remainder(phi, torch.pi * 2)

        error_0 = torch.min(torch.abs(phi - aolp_0), torch.pi * 2 - torch.abs(phi - aolp_0))
        error_1 = torch.min(torch.abs(phi - aolp_1), torch.pi * 2 - torch.abs(phi - aolp_1))
        error_2 = torch.min(torch.abs(phi - aolp_2), torch.pi * 2 - torch.abs(phi - aolp_2))
        error_3 = torch.min(torch.abs(phi - aolp_3), torch.pi * 2 - torch.abs(phi - aolp_3))

        error = torch.min(torch.min(error_0, error_1), torch.min(error_2, error_3))  # 选最小的error
        loss_invalid_mask = (error > torch.pi / 2.0)
        error = error * dolp_gt
        # error = error * percentage[:,0]
        error[loss_invalid_mask] = 0.0
        return error.sum() / error.shape[0]


    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        # 不满足mask的sdf值都应大于0
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        # mask_loss = 1 * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt,
        #                                                                   reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_bottom_normal_loss(self,bottom_normals_predict,bottom_normal_target,bottom_mask):
        """
        我不知道这个有啥用。
        """
        n_rays,_ = bottom_normals_predict.shape
        target_bottom_normals = bottom_normal_target.repeat(n_rays,1)

        loss_orientation = torch.min(torch.sqrt(((target_bottom_normals - bottom_normals_predict) ** 2).sum(dim=1)),
                                     torch.sqrt(((target_bottom_normals + target_bottom_normals) ** 2).sum(dim=1)))
        loss_orientation[~bottom_mask] = 0.0
        loss = loss_orientation.sum() / float(bottom_mask.shape[0])
        return loss

    def get_normal_loss(self, pred_normal, gt_normal, mask=None, var=None):
        # 对真实值法向量和预测值法向量进行归一化
        # true_vectors_normalized = F.normalize(gt_normal, dim=1)
        predicted_vectors_normalized = F.normalize(pred_normal, dim=1)

        # 计算余弦相似性度量
        # cos_similarities = F.cosine_similarity(true_vectors_normalized.squeeze(0), predicted_vectors_normalized.squeeze(0),
        #                                        dim=1)
        cos_similarities = F.cosine_similarity(gt_normal.squeeze(0),predicted_vectors_normalized.squeeze(0),
                                               dim=1)
        cos_similarities[~var] = 1
        cos_similarities[cos_similarities < 0] = 1
        # 计算平均损失
        # loss = 1 - cos_similarities.mean()
        loss = torch.mean(1 - cos_similarities)
        return loss

    def smooth(self, pred_normal, mask):
        import numpy as np
        from PIL import Image
        import cv2

        # 首先，计算预测的法向量
        eps = 1e-10
        pred_normal = pred_normal.permute(0, 2, 1).contiguous().view(1, 3, 1028, 1232)
        pred_normal = torch.clip(pred_normal, -1.0 + eps, 1.0 - eps)
        # 计算mask
        mask = mask.reshape(1028,1232).detach().cpu().numpy()
        # mask_pil = Image.fromarray((mask * 255).astype(np.uint8))  # 验证mask是否正确
        # mask_pil.save("D:\Desktop/note\mask.png")
        diff_map = torch.zeros_like(pred_normal).cuda()

        # 腐蚀一次mask
        kernel = np.ones((3, 3), np.uint8)  # 矩形结构
        mask = cv2.erode(mask.astype(np.uint8), kernel)  # 腐蚀
        # cv2.imwrite("D:\Desktop/note\eroded_mask.png", mask*255)
        mask = torch.from_numpy(mask).cuda()  # 测试结果为，完美！

        # 设置一个平均卷积核，卷出一张特征图
        kernel_size = 5  # 有必要调整。越大得到的mean map就越模糊
        mean_kernel = torch.ones([1, 1, kernel_size, kernel_size]) / kernel_size ** 2
        mean_kernel = mean_kernel.cuda()
        mean_kernel = nn.Parameter(data=mean_kernel, requires_grad=False)
        mean_map_1 = nn.functional.conv2d(pred_normal[:,0,:,:].unsqueeze(0), mean_kernel, padding=kernel_size // 2)
        mean_map_2 = nn.functional.conv2d(pred_normal[:,1,:,:].unsqueeze(0), mean_kernel, padding=kernel_size // 2)
        mean_map_3 = nn.functional.conv2d(pred_normal[:,2,:,:].unsqueeze(0), mean_kernel, padding=kernel_size // 2)
        mean_map = torch.cat([mean_map_1,mean_map_2,mean_map_3], dim=1)

        # temp = (mean_map.squeeze(0).permute(1,2,0).detach().cpu().numpy() + 1) * 127.5
        # mean_map_pil= Image.fromarray(temp.astype(np.uint8))
        # mean_map_pil.save("D:\Desktop/note\mean_map.png")

        # temp1 = (pred_normal.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) * 127.5
        # pred_normal_pil = Image.fromarray(temp1.astype(np.uint8))
        # pred_normal_pil.save("D:\Desktop/note\pred_normal_map.png")

        mask = mask.expand(1,3,1028,1232)
        diff_map[mask] = torch.abs(pred_normal - mean_map)[mask]  # 预测normal和光滑normal之间的差异。

        # 根据mask，找到对应位置，计算loss
        diff_map = diff_map.permute(0,2,3,1)
        indices = torch.nonzero(mask[0][0])# 获取掩码为 True 的像素点的索引
        masked_pixels = diff_map[0,indices[:, 0], indices[:, 1],:]# 获取掩码为 True 的像素点的坐标

        distances = torch.norm(masked_pixels,p=2,dim=1)# 计算欧氏距离
        smooth_reg = distances.sum()/masked_pixels.shape[0]# 计算欧氏距离的总和
        return smooth_reg

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask, transparent_confidence):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        transparent_confidence = transparent_confidence[network_object_mask & object_mask]

        rgb_loss = (self.l1_loss_c(rgb_values, rgb_gt).sum(1) * transparent_confidence).sum() / float(
            object_mask.shape[0])
        return rgb_loss

    def get_stokes_loss(self, render_s0, render_s1, render_s2, gt_s0,
                        gt_s1, gt_s2, network_object_mask, object_mask,
                        transparent_confidence):
        valid_mask = network_object_mask & object_mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        gt_s0 = gt_s0[valid_mask].unsqueeze(1)
        gt_s1 = gt_s1[valid_mask].unsqueeze(1)
        gt_s2 = gt_s2[valid_mask].unsqueeze(1)
        transparent_confidence = transparent_confidence[valid_mask]
        s0_loss = self.l1_loss(render_s0[valid_mask], gt_s0) \
                  / float(gt_s0.shape[0])
        s1_loss = self.l1_loss(render_s1[valid_mask], gt_s1) \
                  / float(gt_s0.shape[0])
        s2_loss = self.l1_loss(render_s2[valid_mask], gt_s2) \
                  / float(gt_s0.shape[0])
        # s0_loss = (self.l1_loss_c(render_s0[valid_mask], gt_s0).sum(1) * transparent_confidence).sum() \
        #           / float(gt_s0.shape[0])
        # s1_loss = (self.l1_loss_c(render_s1[valid_mask], gt_s1).sum(1) * transparent_confidence).sum() \
        #           / float(gt_s0.shape[0])
        # s2_loss = (self.l1_loss_c(render_s2[valid_mask], gt_s2).sum(1) * transparent_confidence).sum() \
        #           / float(gt_s0.shape[0])
        # s0_loss = self.l1_loss(render_s0[valid_mask], gt_s0) / float(gt_s0.shape[0])
        # s1_loss = self.l1_loss(render_s1[valid_mask], gt_s1) / float(gt_s0.shape[0])
        # s2_loss = self.l1_loss(render_s2[valid_mask], gt_s2) / float(gt_s0.shape[0])
        stokes_loss = s0_loss+s1_loss+s2_loss
        return stokes_loss

    def forward(self, model_inputs,model_outputs,ground_truth,epoch,sampling_idxs=None):
        # normal_gt = ground_truth['normal'].cuda()  # [bs,num_pixels,3]
        aolp_gt = model_outputs['gt_aolp'][:,0].unsqueeze(0)
        # dolp_gt = model_outputs['gt_dolp'][:,0].unsqueeze(0)
        # s0_gt = (model_outputs['gt_color'][:,0].unsqueeze(0) + 1.0) / 2.0
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']  # 这是真实mask值
        # network_normals = model_outputs['normals'] # [num_pixels,3]
        # s0_render = model_outputs['s0'].cuda()
        stokes_r = model_outputs['stokes_r']
        pred_color = torch.zeros((100, 3))
        gt_color = torch.zeros((100, 3))
        color_loss = torch.tensor([0]).cuda()
        stokes_loss = torch.tensor([0]).cuda()
        smooth_reg = torch.tensor([0]).cuda()
        aolp_loss = torch.tensor([0]).cuda()  # 几何loss
        aolp_render_loss = torch.tensor([0]).cuda()
        transparent_confidence = model_outputs['transparent_confidence'] * 5
        max_value = transparent_confidence.max()
        if max_value > 0:  # 此时可见性代码的是有效的。
            transparent_confidence = 1 - transparent_confidence / max_value
            model_outputs['gt_aolp'] = model_outputs['fused_aolp']
        else:
            transparent_confidence = torch.ones_like(transparent_confidence)
            model_outputs['gt_aolp'] = model_outputs['gt_aolp']


        if 'rgb_values' in model_outputs and self.color_render_weight != 0.0:
            pred_color = model_outputs['rgb_values']
            gt_color = ground_truth['rgb'].cuda()

        pose = model_inputs['pose']

        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        pred_normal = utils.rend_util.get_normals_cam(model_outputs['grad_theta'].unsqueeze(0), pose)
        pred_normal = torch.clip(pred_normal, -1.0 + 1e-10, 1.0 - 1e-10)

        # 如果训练的是透明物体的话
        if self.use_refract_ray and stokes_r is not None and self.aolp_render_weight != 0.0:
            reflection_intensity = model_outputs['reflection_intensity'].cuda()
            transmittance_intensity = model_outputs['transmittance_intensity'].cuda()
            # stokes_r = model_outputs['stokes_r'].cuda()

            valid_mask = (aolp_gt > 0.0).reshape(-1)
            # 原版的percentage
            I_sum = reflection_intensity + transmittance_intensity
            percentage = reflection_intensity / I_sum
            percentage[torch.isnan(percentage)] = 0.0
            percentage = torch.clamp(percentage, 0, 1)
            aolp_render_loss = self.get_aolp_render_loss(stokes_r, aolp_gt, network_object_mask, object_mask,
                                                         valid_mask, percentage, transparent_confidence)
            if (network_object_mask & object_mask).sum() != 0:
                aolp_loss = self.get_aolp_loss_gb(pred_normal[0], aolp_gt[:, network_object_mask & object_mask].squeeze(0),
                                                  percentage[network_object_mask & object_mask, 0])

        if 'rgb_values' in model_outputs and self.color_render_weight != 0.0:
            color_loss = self.get_rgb_loss(pred_color, gt_color, network_object_mask, object_mask,transparent_confidence)
        if 's0_render' in model_outputs and self.stokes_render_weight != 0.0 and model_outputs['s0_render'] != None:
            stokes_loss = self.get_stokes_loss(model_outputs['s0_render'], model_outputs['s1_render'],
                                            model_outputs['s2_render'], model_inputs['gt_s0'][0,:,0].cuda(),
                                            model_inputs['gt_s1'][0,:,0].cuda(), model_inputs['gt_s2'][0,:,0].cuda(),
                                            network_object_mask, object_mask,
                                            transparent_confidence)


        # loss = self.mask_weight * mask_loss + self.eikonal_weight * eikonal_loss +\
        #        self.aolp_render_weight * aolp_render_loss + normal_loss * normal_weight
        # loss = self.mask_weight * mask_loss + self.eikonal_weight * eikonal_loss
        mask_weight = self.mask_weight-epoch/1000.0 if epoch <= 1000 else 1.0
        loss = mask_weight * mask_loss + self.eikonal_weight * eikonal_loss + \
               self.aolp_render_weight * aolp_render_loss

        if 'rgb_values' in model_outputs and self.color_render_weight != 0.0:
            loss = loss + self.color_render_weight * color_loss
        if 's0_render' in model_outputs and self.stokes_render_weight != 0.0 and model_outputs['s0_render'] != None:
            if torch.isnan(stokes_loss).any():
                loss = loss
            else:
                loss = loss + self.stokes_render_weight * stokes_loss
        if torch.isnan(loss).any():
            print("Loss is NaN")
            print(mask_loss)
            print(color_loss)
            print(eikonal_loss)
            print(stokes_loss)
            print(model_outputs['s0_render'].shape)
            # loss = torch.tensor([0]).cuda()
        return {
            'loss': loss,
            'aolp_render_loss': self.aolp_render_weight * aolp_render_loss,
            'eikonal_loss': self.eikonal_weight * eikonal_loss,
            'mask_loss': self.mask_weight * mask_loss,
            'aolp_loss': aolp_loss,
            'color_loss': self.color_render_weight * color_loss,
            'stokes_loss': self.stokes_render_weight * stokes_loss
        }

    # def forward(self, model_inputs, model_outputs,ground_truth,epoch,sampling_idxs=None):
    #     rgb_gt = ground_truth['rgb'].cuda()
    #     network_object_mask = model_outputs['first_net_mask']
    #     object_mask = model_outputs['object_mask']
    #
    #     rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
    #     mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
    #     eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
    #
    #     loss = 1 * rgb_loss + \
    #            self.eikonal_weight * eikonal_loss + \
    #            self.mask_weight * mask_loss
    #
    #     return {
    #         'loss': loss,
    #         'rgb_loss': 0 * rgb_loss,
    #         'eikonal_loss': eikonal_loss,
    #         'mask_loss': mask_loss,
    #     }