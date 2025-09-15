"""
train.py 有以下操作
1.加载数据集（图片，标签，还有一些框的信息）
    建立一个类，类中定义读取
2.网络，比如vgg16
3. 设置训练的优化器等参数
3.设置权重保存路径
4.设置tensorboard保存路径
"""
import os
import time
import torch
import shutil
import argparse
import datetime
from config import cfg
from Utils.logger import setup_logger
from dataset import get_segmentation_dataset
from model import get_segmentation_Model
from Utils.loss import get_segmentation_loss
from Utils.loss import *
from Utils.lr_scheduler import WarmupPolyLR
from Utils.score import SegmentationMetric
from Utils.distributed import *
import torch.utils.data as data


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--model', type=str, default='DBCNet',
                    help='model name (default: fcn32s)')
    parser.add_argument('--dataset', type=str, default='shvpli',
                    choices=['bijie', 'landslide4Sense', 'luding'],
                    help='dataset name (default: pascal_voc)')
    args = parser.parse_args()

    return args


# python train.py --model FCN_8 --backbone vgg16 --dataset potsdam

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset and dataloader
        data_kwargs = {'height': cfg.TRAIN.WEIGHT, 'width': cfg.TRAIN.WIDTH, 'classes': cfg.TRAIN.CLASSES}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // cfg.TRAIN.BATCH_SIZE
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters)
        val_sampler = make_data_sampler(val_dataset, False)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TRAIN.BATCH_SIZE)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.TRAIN.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.TRAIN.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_Model(name=args.model, nclass=cfg.TRAIN.CLASSES).to(self.device)
        # 需要的时候接着训练
        if cfg.TRAIN.RESUME:
            pth_filename = '{}_{}.pth'.format(args.model, args.dataset)
            resume_path = os.path.join(cfg.DATA.WEIGHTS_PATH, pth_filename)

            print('Resuming training, loading {}...'.format(resume_path))
            self.model.load_state_dict(torch.load(resume_path))

        # create criterion
        self.loss_ce = get_segmentation_loss(args.model, use_ohem=cfg.LOSS.USE_OHEM, aux=cfg.LOSS.AUX,
                                             aux_weight=cfg.LOSS.AUX_WEIGHT, ignore_index=-1).to(self.device)
        self.loss_dc = DiceCeLoss().to(self.device)


        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        params_list.append({'params': self.model.parameters(), 'lr': cfg.TRAIN.LR})
        if hasattr(self.model, 'pre_trained'):
            params_list.append({'params': self.model.pre_trained.parameters(), 'lr': cfg.TRAIN.LR})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': cfg.TRAIN.LR * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=cfg.TRAIN.LR,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=self.max_iters,
                                         power=0.9,
                                         warmup_factor=cfg.TRAIN.WARMUP_FACTOR,
                                         warmup_iters=cfg.TRAIN.WARMUP_ITERS,
                                         warmup_method=cfg.TRAIN.WARMUP_METHOD)


        self.metric = SegmentationMetric(train_dataset.classes)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = cfg.TRAIN.EPOCHS, self.max_iters
        log_per_iters, val_per_iters = cfg.TRAIN.LOG_ITER, cfg.TRAIN.VAL_EPOCH * self.iters_per_epoch
        save_per_iters = cfg.TRAIN.SAVE_EPOCH * self.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, labels) in enumerate(self.train_loader):
            #print(images.shape, targets.shape)
            iteration = iteration + 1
            self.lr_scheduler.step(iteration)

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss_ce = self.loss_ce(outputs, labels)
            # loss_dc = self.loss_dc(outputs[0], labels)

            loss_total = dict(loss=loss_ce)

            losses = sum(loss for loss in loss_total.values())


            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_total)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.metric.update(outputs[0], labels)
            pixAcc, m_IOU = self.metric.get()

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || pixAcc:{:.4f} || m_IOU:{:.4f} || || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(), pixAcc, m_IOU,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
                save_checkpoint(self.model, self.args, is_best=False)

            if not cfg.TRAIN.SKIP_VAL and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, labels) in enumerate(self.val_loader):
            image = image.to(self.device)
            label = labels.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], label)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.DATA.WEIGHTS_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    filename = os.path.join(directory, filename)
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger("semantic_segmentation", cfg.TRAIN.LOG_DIR, get_rank(), filename='{}_{}_log.txt'.format(
        args.model, args.dataset))
    logger.info(args)
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
