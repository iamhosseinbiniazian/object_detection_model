"""Train Faster-RCNN end to end."""
import argparse
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from detection import AUTDetection

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b_voc',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='aut',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='faster_rcnn_resnet50_v1b_voc_best.params',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')

    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')

    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')

    args = parser.parse_args()
    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 8000
        args.wd = float(args.wd) if args.wd else 1e-4
        num_gpus = len(args.gpus.split(','))
        if num_gpus == 1:
            args.lr_warmup = -1
        else:
            args.lr *= num_gpus
            args.lr_warmup /= num_gpus
    elif args.dataset == 'aut':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    return args


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        # pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    elif dataset.lower()== 'aut':
        train_dataset = AUTDetection(
            splits=[('train')])
        val_dataset = AUTDetection(
            splits=[('train')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_workers, multi_stage):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            train_transform(net.short, net.max_size, net, ashape=net.ashape,
                            multi_stage=multi_stage)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        net.hybridize(static_alloc=args.static_alloc)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                        det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha





if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(1100)
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'bn':
            kwargs['num_devices'] = len(args.gpus.split(','))
    net_name = '_'.join(('faster_rcnn', args.network))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True, **kwargs)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, test_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, test_data = get_dataloader(
        net, train_dataset, test_dataset, FasterRCNNDefaultTrainTransform,
        FasterRCNNDefaultValTransform, args.batch_size, args.num_workers, args.use_fpn)

    # training
    map_name, mean_ap = validate(net, test_data, ctx, eval_metric, args)
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    print(' Validation: \n{}'.format(val_msg))