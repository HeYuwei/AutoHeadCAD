from imgaug import augmenters as iaa
import cv2
import numpy as np
import torch
from types import FunctionType
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F


class GradCam(object):
    def __init__(self, model, detach=True):
        self.model = model
        self.opt = model.opt
        self.vis_layer_names = model.opt.vis_layer_names
        self.init_vis_layers(detach)

    def _add_hook(self, net, detach, prefix=''):
        '''
        prefix: (module.)*
        '''
        if hasattr(net, '_modules'):
            for m_name, module in net._modules.items():
                if m_name != 'module':
                    new_prefix = prefix + m_name + '.'
                else:
                    new_prefix = prefix
                current_module_name = prefix + m_name
                if current_module_name in self.vis_layer_names:
                    if detach:
                        save_output_code = compile('def save_output' + m_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + current_module_name + '\"]["output"].append(output.detach());', "<string>", "exec")
                    else:
                        save_output_code = compile('def save_output' + m_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + current_module_name + '\"]["output"].append(output);', "<string>", "exec")
                    func_space = {'self': self}
                    func_space.update(globals())
                    save_output = FunctionType(
                        save_output_code.co_consts[0], func_space, "save_output")
                    h = module.register_forward_hook(save_output)
                    self.forward_hook_handles.append(h)

                    save_gradient_code = compile(
                        'def save_gradient' + m_name +
                        '(module, input_grad, output_grad): '
                        'vis_info = getattr(self, "vis_info");'
                        'vis_info[\"' + current_module_name + '\"]["grad"].append(output_grad[0]);', "<string>", "exec")
                    save_gradient = FunctionType(
                        save_gradient_code.co_consts[0], func_space, "save_gradient")
                    h = module.register_backward_hook(save_gradient)
                    self.backward_hook_handles.append(h)
                self._add_hook(module, detach, new_prefix)

    def add_hook(self, detach):
        for net_name in self.model.net_names:
            net = getattr(self.model, 'net_' + net_name)
            self._add_hook(net, detach)

    def init_vis_layers(self, detach):
        self.vis_info = {}
        for m_name in self.vis_layer_names:
            self.vis_info[m_name] = {}
            self.vis_info[m_name]['output'] = []
            self.vis_info[m_name]['grad'] = []
            self.vis_info[m_name]['cam'] = []
            self.vis_info[m_name]['vis_img'] = []

        self.forward_hook_handles = []
        self.backward_hook_handles = []
        self.add_hook(detach)

    def remove_hook(self):
        for h in self.forward_hook_handles:
            h.remove()
        self.forward_hook_handles = []
        for h in self.backward_hook_handles:
            h.remove()
        self.backward_hook_handles = []

    def cal_grad(self, y, t_label, retain_graph=False, create_graph=False):
        """
        Args:
            model:
            imgs: NHWC
            t_label: target label to be visualized
        """
        model = self.model
        output = y
        one_hots = torch.zeros(output.shape[0], output.shape[1]).cuda(
            model.opt.gpu_ids[0])
        one_hots[:, t_label] = 1
        ys = torch.sum(one_hots * output)
        ys.backward(retain_graph=retain_graph, create_graph=create_graph)

    def cal_cam(self, visualize=True, scores=None):
        self.cat_info()
        for key in self.vis_info.keys():
            grads_val = self.vis_info[key]['grad'] # (batch, F, 16, 16)
            feature = self.vis_info[key]['output'] # (batch, F, 16, 16)
            weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)
            cam = weights * feature
            cam = torch.sum(cam, dim=1)
            cam = F.relu(cam) # (batch, 16, 16)
            # normalize to (0, 1)
            tmp = cam.view(cam.shape[0], -1)
            max_value = torch.max(tmp, dim=1)[0] # (batch)
            max_value = max_value.unsqueeze(dim=1).unsqueeze(dim=1) # (batch, 1, 1)
            cam = torch.div(cam, max_value)
            # weighted by score
            if scores is not None:
                scores = scores.cuda(device=cam.device).view(-1, 1, 1)
                cam.mul_(scores)
            # threshold cam
            cam = torch.where(cam>self.opt.cam_thresh, cam, torch.zeros_like(cam))
            # save to vis_info
            if visualize:
                cam = cam.cpu().numpy()
                self.vis_info[key]['cam'] = cam
            else:
                self.vis_info[key]['cam'] = cam

    def show_cam_on_image(self, imgs):
        '''
        imgs: (N, h, w, 3)
        '''
        imgs = imgs / 255.
        iaa_resize = iaa.Resize(
            {"height": imgs.shape[1], "width": imgs.shape[2]}, interpolation="linear")
        for key in self.vis_info.keys():
            cams = self.vis_info[key]['cam']
            cams = np.transpose(cams, (1, 2, 0))
            vis_imgs = []
            heatmap_imgs = []
            mask_imgs = []
            masks = iaa_resize.augment_image(cams) * 255
            masks = np.transpose(masks, (2, 0, 1))
            masks = masks.astype('uint8')
            for i in range(imgs.shape[0]):
                img = imgs[i]  # (3, 512, 512)
                if self.model.opt.input_mode in ['bag', 'bag_middle']:
                    heatmap = cv2.applyColorMap(masks[i // 3], cv2.COLORMAP_JET)
                    mask_imgs.append(masks[i // 3])
                else:
                    heatmap = cv2.applyColorMap(masks[i], cv2.COLORMAP_JET)
                    mask_imgs.append(masks[i])
                heatmap_imgs.append(heatmap)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.float32(img)
                cam = cam / np.max(cam)
                vis_img = np.uint8(255 * cam)
                vis_imgs.append(vis_img)

            self.vis_info[key]['mask_img'] = mask_imgs
            self.vis_info[key]['vis_img'] = vis_imgs
            self.vis_info[key]['heatmap_img'] = heatmap_imgs

    def reset_info(self):
        for m_name in self.vis_layer_names:
            self.vis_info[m_name] = {}
            self.vis_info[m_name]['output'] = []
            self.vis_info[m_name]['grad'] = []
            self.vis_info[m_name]['cam'] = []
            self.vis_info[m_name]['imgs'] = []

    def cat_info(self):
        ''' concatenate tensor list into one tensor'''
        for m_name in self.vis_layer_names:
            if isinstance(self.vis_info[m_name]['cam'], list) and len(self.vis_info[m_name]['cam']) > 0:
                self.vis_info[m_name]['cam'] = torch.cat(
                    self.vis_info[m_name]['cam'], dim=0)
            if isinstance(self.vis_info[m_name]['output'], list) and len(self.vis_info[m_name]['output']) > 0:
                self.vis_info[m_name]['output'] = torch.cat(
                    self.vis_info[m_name]['output'], dim=0)
            if isinstance(self.vis_info[m_name]['grad'], list) and len(self.vis_info[m_name]['grad']) > 0:
                self.vis_info[m_name]['grad'] = torch.cat(
                    self.vis_info[m_name]['grad'], dim=0)