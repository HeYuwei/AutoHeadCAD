import json
from util import *
from sklearn.metrics import roc_curve
import torch.nn as nn
from .base_model import BaseModel
from .model_option import model_dict
import torch.nn.functional as F
from util.loss import FocalLoss
from schedulers import get_scheduler


class CTBaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):        
        # data
        parser.add_argument('--train_dataset_dir', type=str, default=None)
        parser.add_argument('--valid_dataset_dir', type=str, default=None)
        parser.add_argument('--data_norm_type', type=str, default="competition")
        parser.add_argument('--target_focus', type=int, default=0, help='0, 1, 2, 3')
        parser.add_argument('--input_mode', type=str, default='multiwindow', choices=('single', 'multiwindow'))
        parser.add_argument('--multiwindows', default=[[[1400,600],[100,80],[40,40]], [[90,40],[30,30],[10,30]], [[750,500],[750,500],[750,500]],[[1400,600],[100,80],[40,40]]])
        parser.add_argument('--vis_ww', type=int, default=750)
        parser.add_argument('--vis_wl', type=int, default=500)

        # model
        parser.add_argument('--method_name', type=str, default='mil')
        parser.add_argument('--net_name', type=str, default="resnet", help='resnet name')
        parser.add_argument('--depth', type=int, default=18)
        parser.add_argument('--frozen_stages', type=int, default=0)
        parser.add_argument('--bottleneck_dim', type=int, default=256, help='bottleneck dim')
        parser.add_argument('--neg_ratio', type=float, default=2)
        parser.add_argument('--sample_wwl', type=int, default=1)
        parser.add_argument('--fc_mult', type=float, default=10)
        parser.add_argument('--pool_mode', type=str, default='gated_attention', choices=('max','average','exp_attn','gated_attention'))
        parser.add_argument('--topk', type=int, default=-1)

        # test
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--save_stat_info', type=int, default=1)
        parser.add_argument('--valid_with_single_slice', type=int, default=0, help = 'If the value is 1, the scan level probability is '
                                                                                     'obtained by selecting the max slice-level probability')
        # loss
        parser.add_argument('--loss_type', type=str, default='focal')
        parser.add_argument('--focal_alpha', type=float, default=0.3)
        parser.add_argument('--focal_gamma', type=float, default=1.5)
        parser.add_argument('--inner_bs', type=int, default=20)
        parser.add_argument('--reinit_data', type=int, default=0)
        return parser

    def __init__(self, opt):
        # call the initialization method of BaseModel
        super().__init__(opt)

        if opt.target_focus == 1:
           opt.topk = 3
           opt.pool_mode = 'exp_attn' 

        model = model_dict[self.opt.method_name]['name']
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            param_dict[param_name] = getattr(self.opt, param_name)

        self.net_res = model(class_num=opt.num_classes, bottleneck_dim=opt.bottleneck_dim, **param_dict)
        self.net_names = ['res']

        self._define_metrics('train', self.opt.dataset_mode)
        self._define_metrics('valid', self.opt.v_dataset_mode)
        self._define_metrics('test', self.opt.v_dataset_mode)

        self.valid_metric = 'series_auc'
        self.scheduler_metric = 'series_auc'

        self.buffer_g_bag_scores = []
        self.buffer_g_bag_preds = []
        self.buffer_g_bag_labels = []
        self.buffer_g_series_scores = []
        self.buffer_g_series_preds = []
        self.buffer_g_series_labels = []
        self.buffer_g_series_depth = [] # slice number of each series

    def setup(self, opt):
        self.buffer_names = self.get_buffer_names()
        self.meters = self.gen_meters()
        self.set_l_state(self.opt.l_state)
        if opt.l_state == 'train':
            self.set_optimizer(opt)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # load checkpoints
        if opt.load_dir is not None:
            load_suffix = 'optimal'
            self.load_networks(load_suffix, load_dir=opt.load_dir)

        elif not (self.opt.l_state == 'train' and not self.opt.continue_train):
            load_suffix = 'optimal'
            self.load_networks(load_suffix)
            self.print_networks(opt.verbose)

        # visualization
        if opt.visualize and opt.vis_method is not None:
            self.vis_method = vis_method_dict[opt.vis_method](self)

        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.cuda(self.gpu_ids[0])
            setattr(self, 'net_' + name, nn.DataParallel(net, opt.gpu_ids))
            net.eval()

    def _define_metrics(self, l_state, dataset_mode):
        if l_state == 'train':
            setattr(self, l_state+'_loss_names', ['c'])
        else:
            setattr(self, l_state+'_loss_names', [])
        setattr(self, l_state+'_s_metric_names', [])
        if l_state == 'valid':
            setattr(self, l_state+'_g_metric_names', [])
        else:
            setattr(self, l_state+'_g_metric_names', ['series_auc'])
        setattr(self, l_state+'_t_metric_names', [])

    def load_networks(self, epoch, load_dir=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if load_dir is None:
                    load_dir = self.save_dir
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location='cpu')
                net.load_state_dict(state_dict, strict=False)
                print('model loaded')

    def set_l_state(self, l_state):
        assert l_state in ['train', 'valid', 'test']
        self.opt.l_state = l_state
        self.loss_names = getattr(self, l_state+'_loss_names')
        self.s_metric_names = getattr(self, l_state+'_s_metric_names')
        self.g_metric_names = getattr(self, l_state+'_g_metric_names')
        self.t_metric_names = getattr(self, l_state+'_t_metric_names')

    def gen_meters(self):
        name_types = ['loss', 's_metric', 't_metric', 'g_metric']
        meters = {}
        for ntype in name_types:
            for l_state in ['train', 'valid', 'test']:
                name_list = getattr(self, l_state + '_' + ntype + '_names')
                for name in name_list:
                    meters[name] = metrics.Meter()
        return meters

    def set_input(self, data):
        if self.opt.l_state == 'train':
            self._set_input(self.opt.dataset_mode, data)
        else:
            self._set_input(self.opt.v_dataset_mode, data)

    def _set_input(self, dataset_mode, data):
        if dataset_mode == 'scan':
            batch = len(data)
            self.input_size = batch
            self.input = []
            self.label = []
            self.input_id = []
            for i in range(batch):
                self.input.append(data[i][0])
                self.label.append(data[i][-2])
                if self.opt.visualize and self.opt.vis_method is not None:
                    self.original_input = data[i][3].unsqueeze(0) # (1, N, 3, H, W)
                self.input_id.append(data[i][-1])

            self.label = torch.LongTensor(self.label).cuda(self.gpu_ids[0])
            self.series_label = self.label.clone().detach()
            self.sample_idx = data[0][-1]

        elif dataset_mode == 'cq500':
            self.input_size = 1
            self.input = [l[0] for l in data]
            self.label = [data[0][-1]]
            self.input_id = [data[0][-2]]

            self.label = torch.LongTensor(self.label).cuda(self.gpu_ids[0])
            self.series_label = self.label.clone().detach()
            self.sample_idx = data[0][-2]

        elif dataset_mode == 'vis':
            batch = len(data)
            self.input_size = batch
            self.input = []
            self.label = []
            self.bag_labels = []
            self.input_id = []
            for i in range(batch):
                self.input.append(data[i][0])
                if len(data[i][1]) > 0:
                    self.label.append(1)
                else:
                    self.label.append(0)
                if self.opt.visualize and self.opt.vis_method is not None:
                    self.bbox_list = data[i][5]
                    self.original_input = data[i][6].unsqueeze(0) # (1, N, 3, H, W)
                self.bag_labels.extend(data[i][-3])
                self.input_id.append(data[i][-1])

                self.label = torch.LongTensor(self.label).cuda(self.gpu_ids[0])
                self.positive_bag_indices1 = data[0][1]
                self.positive_image_indices1 = data[0][2]
                dcm_paths = data[0][3]
                self.series_label = self.label.clone().detach()
                self.bag_labels = torch.LongTensor(self.bag_labels).cuda(self.gpu_ids[0])
                self.sample_idx = data[0][-1]
        else:
            raise Exception("invalid dataset_mode!")

    def forward(self):
        pass

    def cal_loss(self):
        if self.opt.loss_type == 'focal':
            self.loss_c = FocalLoss(gamma=self.opt.focal_gamma, alpha=self.opt.focal_alpha, device=self.opt.gpu_ids[0])(
                self.y.cuda(self.gpu_ids[0]), self.label.cuda(self.gpu_ids[0]).long())
        else:
            self.loss_c = nn.CrossEntropyLoss()(self.y, self.label.long())

    def accuracy(self, score, label):
        if self.opt.threshold is not None:
            threshold = self.opt.threshold
        else:
            threshold = 0.5
        pred = score > threshold
        pred = pred.cpu().long()
        correct = (label.cpu().long() == pred)
        return 100.0 * correct.sum() / len(correct)

    def cal_s_metric(self):
        if 'series_accuracy' in self.s_metric_names:
            self.s_metric_series_accuracy = self.accuracy(
                self.series_score, self.series_label)

    def cal_g_metric(self):
        if 'series_precision' in self.g_metric_names:
            self.g_metric_series_precision = metrics.precision(
                self.t_metric_series_cmatrix, 1)
        if 'series_recall' in self.g_metric_names:
            self.g_metric_series_recall = metrics.recall(
                self.t_metric_series_cmatrix, 1)
        if 'series_fscore' in self.g_metric_names:
            self.g_metric_series_fscore = metrics.f_score(
                self.t_metric_series_cmatrix, 1)
        if 'series_auc' in self.g_metric_names:
            self.g_metric_series_auc = metrics.auc_score(
                self.buffer_g_series_labels, self.buffer_g_series_scores)

    def cal_t_metric(self):
        if 'bag_cmatrix' in self.t_metric_names:
            self.t_metric_bag_cmatrix = metrics.comfusion_matrix(
                self.buffer_g_bag_preds, self.buffer_g_bag_labels, self.opt.num_classes)
        if 'series_cmatrix' in self.t_metric_names:
            self.t_metric_series_cmatrix = metrics.comfusion_matrix(
                self.buffer_g_series_preds, self.buffer_g_series_labels, self.opt.num_classes)

    def transform_orinput(self):
        # self.original_input: (1, N, 3, H, W)
        self.original_input = self.original_input[0, :, 1, :, :]  # (N, 512, 512)
        self.original_input = self.original_input.unsqueeze(dim=-1)  # (N, 512, 512, 1)
        self.original_input = self.original_input.repeat_interleave(3, dim=-1).cpu().numpy()  # (N, 512, 512, 3)

    def transform_cam(self):
        for key in self.vis_method.vis_info.keys():
            cam = self.vis_method.vis_info[key]['cam'].repeat_interleave(3, dim=0)
            self.vis_method.vis_info[key]['cam'] = cam.cpu().numpy()

    def save_images(self, h_num=8, factor=0.5, font=0.5):
        def plot_highlight_box(im_list, index_list, color):
            for index in index_list:
                im = im_list[index]
                cv2.rectangle(im, (0, 0), (im.shape[0], im.shape[1]), color, 5)
                im_list[index] = im
            return im_list

        candidate_imgs = []
        orig_imgs = self.original_input
        if self.opt.v_dataset_mode != 'scan':
            plot_highlight_box(orig_imgs, self.positive_image_indices1, (0, 0, 255))
        candidate_imgs.append(orig_imgs)

        key_num = len(self.vis_method.vis_info.keys())
        for key in self.vis_method.vis_info.keys():
            cam_imgs = self.vis_method.vis_info[key]['vis_img']
            threshold = self.opt.threshold if self.opt.threshold is not None else 0.5
            candidate_imgs.append(cam_imgs)

        h_num = len(orig_imgs) if h_num == -1 else h_num
        edge = int(factor * 512)
        result_imgs = []
        start_ind = 0
        blank_image = np.zeros((512, 512, 3))
        while start_ind < len(orig_imgs):
            for candidate in candidate_imgs:
                result_imgs.extend(candidate[start_ind:start_ind+h_num])
                if start_ind + h_num > len(orig_imgs):
                    blank_num = start_ind + h_num - len(orig_imgs)
                    result_imgs.extend([blank_image]*blank_num)
            start_ind += h_num

        result_imgs = np.array(result_imgs, dtype=np.uint8)
        result_img = cat_image(result_imgs, h_num=h_num, factor=factor)
        
        # show bbox
        if hasattr(self, 'bbox_list'):
            for i, bboxes in enumerate(self.bbox_list):
                row = i // h_num
                col = i % h_num
                for box in bboxes:
                    left = int(box[0] * factor)
                    top = int(box[1] * factor)
                    right = int(box[2] * factor)
                    bottom = int(box[3] * factor)
                    cv2.rectangle(result_img, (col*edge+left, row*(key_num+1)*edge+top),
                                (col*edge+right, row*(key_num+1)*edge+bottom), (0, 0, 255), 1)

        save_name = os.path.join(self.vis_dir, '{}.png'.format(self.sample_idx))
        cv2.imwrite(save_name, result_img)

    def stat_info(self):
        if self.opt.l_state == 'train':
            self.current_dataset_mode = self.opt.dataset_mode
        else:
            self.current_dataset_mode = self.opt.v_dataset_mode

        if self.current_dataset_mode in ['scan', 'cq500']:
            # label: (1), y: (1, 2), score: (1)
            if self.opt.threshold is not None:
                self.series_pred = self.series_score > self.opt.threshold
                self.series_pred = self.series_pred.cpu().long()
            else:
                scores = F.softmax(self.series_y.cpu(), dim=1)
                self.series_pred = torch.argmax(scores, dim=1)
            self.buffer_g_series_scores.extend(
                self.series_score.cpu().view(-1).tolist())
            self.buffer_g_series_labels.extend(
                self.series_label.cpu().long().view(-1).tolist())
            self.buffer_g_series_preds.extend(
                self.series_pred.view(-1).tolist())
            self.buffer_g_bag_scores.extend(
                self.bag_scores.cpu().view(-1).tolist())
            self.buffer_ginput_ids.extend(self.input_id)
            self.buffer_g_series_depth.append(len(self.bag_scores))

        elif self.current_dataset_mode == 'vis':
            # label: (1), all_labels: (N), y: (1, 2), bag_ys:(N, 2), score: (1), bag_scores: (N)
            if self.opt.threshold is not None:
                self.bag_preds = self.bag_scores > self.opt.threshold
                self.bag_preds = self.bag_preds.cpu().long()
                self.series_pred = self.series_score > self.opt.threshold
                self.series_pred = self.series_pred.cpu().long()
            else:
                series_scores = F.softmax(self.series_y.cpu(), dim=1)
                self.series_pred = torch.argmax(series_scores, dim=1)
                bag_scores = F.softmax(self.bag_ys.cpu(), dim=1)
                self.bag_preds = torch.argmax(bag_scores, dim=1)
                
            self.buffer_g_bag_scores.extend(
                self.bag_scores.cpu().view(-1).tolist())
            self.buffer_g_bag_labels.extend(
                self.bag_labels.cpu().long().view(-1).tolist())
            self.buffer_g_bag_preds.extend(self.bag_preds.view(-1).tolist())
            self.buffer_g_series_scores.extend(
                self.series_score.cpu().view(-1).tolist())
            self.buffer_g_series_labels.extend(
                self.series_label.cpu().long().view(-1).tolist())
            self.buffer_g_series_preds.extend(
                self.series_pred.view(-1).tolist())
            self.buffer_ginput_ids.extend(self.input_id)
            self.buffer_g_series_depth.append(len(self.bag_labels))

        else:
            raise Exception('no such dataset mode!')
        
        if self.opt.visualize and self.opt.vis_method is not None and self.opt.l_state != 'train':
            bag_scores = self.bag_scores.view(-1) if self.current_dataset_mode != 'scan' else None
            self.visualize(bag_scores)

    def visualize(self, bag_scores=None):
        if self.opt.vis_method == 'original':
            self.transform_orinput()
            for key in self.vis_method.vis_info.keys():
                self.vis_method.vis_info[key]['vis_img'] = self.original_input
            self.save_images()
            self.vis_method.reset_info()
        else:
            self.vis_method.cal_cam(visualize=True, scores=bag_scores)
            self.transform_orinput()
            self.vis_method.show_cam_on_image(self.original_input)
            self.save_images()
            self.vis_method.reset_info()

    def save_stat_info(self):
        if not self.opt.save_stat_info:
            return
        f = open(osp.join(self.save_dir, '{}_stat_info.json'.format(self.opt.l_state)), 'w')
        scores = []
        start_ind = 0
        for i in range(len(self.buffer_g_series_depth)):
            depth = self.buffer_g_series_depth[i]
            sample_idx = self.buffer_ginput_ids[i]
            label = self.buffer_g_series_labels[i]
            series_score = self.buffer_g_series_scores[i]
            bag_scores = self.buffer_g_bag_scores[start_ind:start_ind+depth]
            record = {'sample_idx': sample_idx, 
                        'label': label,
                        'score': series_score,
                        'bag_scores': bag_scores}
            if len(self.buffer_g_bag_labels) > 0:
                record['bag_labels'] = self.buffer_g_bag_labels[start_ind:start_ind+depth]
            if hasattr(self, 'buffer_g_weights'):
                weights = self.buffer_g_weights[start_ind:start_ind+depth]
                record['weights'] = weights
            scores.append(record)
            start_ind += depth
        json.dump(scores, f)
        f.close()

    def next_epoch(self):
        if self.opt.reinit_data:
            self.opt.dataset.dataset.__init__(self.opt, 'train')