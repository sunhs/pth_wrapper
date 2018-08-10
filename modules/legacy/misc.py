import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from my_modules.modules import utils

plt.switch_backend('agg')


def my_multilabel_soft_margin_loss(inputs, targets, config):
    """Inputs and targets are both tensors."""
    weights = torch.ones(inputs.size()).cuda(config.DEFAULT_GPU)
    # n_ele = inputs.nelement()

    for i in range(targets.size(0)):
        for j in range(targets.size(1)):
            if targets[i, j].item() == 1:
                weights[i, j] = config.W_POS
            elif targets[i, j].item() == 0:
                weights[i, j] = config.W_NEG
            elif targets[i, j].item() == -1:
                weights[i, j] = 0

    return functional.binary_cross_entropy(
        # functional.softmax(inputs, dim=1),
        functional.sigmoid(inputs),
        targets,
        weights,
        size_average=False) / inputs.size(0)

    # return functional.binary_cross_entropy(
    #     torch.sigmoid(inputs).contiguous().view(n_ele),
    #     targets.contiguous().view(n_ele),
    #     weights.contiguous().view(n_ele),
    #     size_average=False) / inputs.size(0)


# def my_precision_sum(predictions, targets):
#     """Both of predictions and targets are Tensors"""
#     assert predictions.size() == targets.size()
#     n_samples = predictions.size(0)
#     prec_sum = 0

#     with torch.no_grad():
#         predictions = predictions.chunk(n_samples, dim=0)
#         targets = targets.chunk(n_samples, dim=0)

#     for i in range(n_samples):
#         p, t = predictions[i].squeeze(), targets[i].squeeze()
#         true_positive = np.where(t.cpu().numpy() == 1)[0]

#         if not len(true_positive):
#             continue

#         _, sortind = torch.sort(p, dim=0, descending=True)
#         correct = 0

#         for i in range(len(true_positive)):
#             if sortind[i].item() in true_positive:
#                 correct += 1

#         prec_sum += correct / max(len(true_positive), 1)

#     return prec_sum


def my_precision_sum(predictions, targets):
    """Both of predictions and targets are cuda tensors"""
    predictions = predictions.cpu()
    targets = targets.cpu()
    assert predictions.size() == targets.size()
    n_samples = predictions.size(0)
    prec_sum = 0

    for i in range(n_samples):
        p, t = predictions[i], targets[i]
        true_positive = np.where(t.numpy() == 1)[0]

        if len(true_positive) == 0:
            continue

        _, sortind = torch.sort(p, dim=0, descending=True)
        correct = 0

        for i in range(len(true_positive)):
            if sortind[i] in true_positive:
                correct += 1

        prec_sum += correct / max(len(true_positive), 1)

    return prec_sum


def visualize(log, train_log, model_dir):
    """"Matplotlib version."""
    plt.plot(
        list(range(len(log['train']['loss']))),
        log['train']['loss'],
        'r',
        label='train_loss')
    plt.plot(
        list(range(len(log['test']['loss']))),
        log['test']['loss'],
        'g',
        label='test_loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss.jpg'))
    plt.clf()

    plt.plot(
        list(range(len(log['train']['prec']))),
        log['train']['prec'],
        'b',
        label='train_prec')
    plt.plot(
        list(range(len(log['test']['prec']))),
        log['test']['prec'],
        'y',
        label='test_prec')
    plt.ylim([0, 0.7])
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'prec.jpg'))
    plt.clf()

    plt.plot(list(range(len(log['test']['map']))), log['test']['map'])
    plt.ylim([0, 0.3])
    plt.savefig(os.path.join(model_dir, 'map.jpg'))
    plt.clf()

    if not train_log:
        return

    data = np.concatenate(train_log)
    plt.plot(list(range(len(data))), data)
    plt.savefig(os.path.join(model_dir, 'train_log.jpg'))
    plt.clf()


def email_images(loss, prec, map_value, epoch, config):
    sender_info = {
        'email': 'report1832@163.com',
        'password': 'wushuangjianji0',
        'server': 'smtp.163.com'
    }
    receiver = config.EMAIL_ADDR

    subject = '{}/{} epoch_{}'.format(
        os.path.basename(config.PROJ_ROOT_DIR),
        os.path.basename(config.__file__).rsplit('.')[0], epoch)

    text = '<h3>map: {:.3f}</h3><h3>prec: {:.3f}</h3><h3>loss: {:.3f}</h3>'.format(
        map_value if map_value is not None else -1, prec, loss)
    image_html = '<br><img src="cid:loss"><img src="cid:prec"><img src="cid:map"><br>'
    content = text + image_html

    images = [{
        'cid': 'loss',
        'path': os.path.join(config.MODEL_DIR, 'loss.jpg')
    }, {
        'cid': 'prec',
        'path': os.path.join(config.MODEL_DIR, 'prec.jpg')
    }, {
        'cid': 'map',
        'path': os.path.join(config.MODEL_DIR, 'map.jpg')
    }]
    utils.send_email(sender_info, receiver, subject, content, images)


def write_voc_result(outputs, imdb, output_dir):
    # outputs: numpy.ndarray  [batch, num_classes]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = [
        'jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
        'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking'
    ]

    for i, class_name in enumerate(classes):
        file_path = os.path.join(output_dir,
                                 'comp9_action_test_{}.txt'.format(class_name))
        with open(file_path, 'w') as f:
            for m in range(imdb['test']['names'].size):
                img_name, pid = imdb['test']['names'][m].rsplit('_', 1)
                line = '{} {} {:.6f}\n'.format(img_name, pid, outputs[m][i])
                f.write(line)
                f.flush()

                
class OCRHandler:
    def __init__(self, epoch, mode, num_batch, ds_size, config):
        self.config = config

        with open(os.path.join(self.config.MODEL_DIR, 'log.pkl'), 'rb') as f:
            log = pickle.load(f)
        with open(os.path.join(self.config.MODEL_DIR, 'train_log.pkl'),
                  'rb') as f:
            train_log = pickle.load(f)

        self.log = log
        self.train_log = train_log
        self.epoch = epoch
        self.mode = mode
        self.num_batch = num_batch
        self.ds_size = ds_size
        self.batch_losses = []
        self.epoch_loss = 0

    def cleanup_batch(self, data, outputs, loss, batch, hz):
        _, labels, _ = data
        loss_value = loss.item()
        self.epoch_loss += loss_value

        if self.mode == 'train':
            self.batch_losses.append(loss_value)
        elif self.mode == 'test':
            # TODO: Add it.
            return

        if self.mode == 'train':
            color_code = '\033[1;31m'
        elif self.mode == 'test':
            color_code = '\033[1;34m'

        # construct bar string
        num_bar = round(batch / self.num_batch * 100)
        bar_str = ' ['
        for _ in range(num_bar):
            bar_str += '='
        for _ in range(100 - num_bar):
            bar_str += ' '
        bar_str += '] '

        batch_info = color_code + '{b:3d}/{n_b:3d}'.format(
            b=batch, n_b=self.num_batch) + bar_str + \
            'loss {l:.3f}, speed {hz:.1f} Hz'.format(
                l=loss_value, hz=hz) + \
            '\033[0m'
        sys.stdout.write('\033[K')
        print(batch_info, end='\r')

    def cleanup_epoch(self):
        sys.stdout.write('\033[K')

        if self.mode == 'train':
            color_code = '\033[1;31m'
            self.train_log.append(self.batch_losses)
            with open(
                    os.path.join(self.config.MODEL_DIR, 'train_log.pkl'),
                    'wb') as f:
                pickle.dump(self.train_log, f)
        elif self.mode == 'test':
            # TODO: Add it.
            return
            # color_code = '\033[1;34m'
            # map_value = self.mapmeter.map()
            # print(color_code, 'map:', map_value, '\033[0m')
            # self.log['test']['map'].append(map_value)

        self.epoch_loss /= self.num_batch
        epoch_info = 'loss {l:.3f}'.format(l=self.epoch_loss)
        print(color_code, epoch_info, '\033[0m')
        self.log[self.mode]['loss'].append(self.epoch_loss)
        with open(os.path.join(self.config.MODEL_DIR, 'log.pkl'), 'wb') as f:
            pickle.dump(self.log, f)

        # plot
        # visualize(self.log, self.train_log, self.config.MODEL_DIR)

        # if getattr(self.config, 'EMAIL', False) and self.mode == 'test':
        #     email_images(self.epoch_loss, self.prec, map_value, self.epoch,
        #                  self.config)
