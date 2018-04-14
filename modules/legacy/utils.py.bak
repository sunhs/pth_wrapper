import email.mime.image
import email.mime.multipart
import email.mime.text
import smtplib
import math
import os
import random
import warnings

import numpy as np
import skimage
import skimage.io
import skimage.transform
import torch
from torch.autograd import Variable
import torch.nn as nn


def parse_time(duration):
    hours = duration / 3600
    minutes = duration % 3600 / 60
    seconds = duration % 3600 % 60
    return '%dh:%dmin:%ds' % (hours, minutes, seconds)


def check_latest_state(state_dir, prefix):
    max_state = -1
    f_names = os.listdir(state_dir)
    for f_name in f_names:
        if not f_name.startswith(prefix):
            continue
        epoch = int(os.path.splitext(f_name)[0].split('_')[1])
        if epoch > max_state:
            max_state = epoch
    return max_state


def load_state_dict(model, pretrain_path, state_dir, prefix):
    model.initialize()  # initialize new layers if necessary, or do other stuffs
    latest_state = check_latest_state(state_dir, prefix)

    if latest_state != -1:
        load_path = os.path.join(
            state_dir, '{}_{}.pth'.format(prefix, latest_state))
        print('==========>> resume from {}'.format(load_path))
        model.load_state_dict(torch.load(load_path))

    elif pretrain_path:
        print('==========>> loading from pretrain: {}'.format(pretrain_path))
        model_state = model.state_dict()
        state_dict = torch.load(pretrain_path)
        for name, param in state_dict.items():
            if name not in model_state:
                print('Key missing: {}'.format(name))
                continue
            if param.size() != model_state[name].size():
                print('Size not match: {}'.format(name))
                continue
            if isinstance(param, (nn.Parameter, Variable)):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)

    else:
        print('==========>> build from scratch')

    print('==========>> done loading initializing model')
    return latest_state


def resize(image):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return skimage.img_as_ubyte(
            skimage.transform.resize(
                image, [224, 224], mode='constant'))


def multi_imread(image_paths, pool, image_size=None):
    images = pool.map(skimage.io.imread, image_paths)
    if image_size:
        images = pool.map(resize, images)
    return images


def random_fliplr(image):
    if random.random() > 0.5:
        return np.fliplr(image)
    return image


def crop(image, size, pos='c'):
    h, w = image.shape[:2]
    new_h, new_w = size

    if pos == 'c':
        u = math.ceil((h - new_h) / 2)
        l = math.ceil((w - new_w) / 2)
    elif pos == 'tl':
        u, l = 0, 0
    elif pos == 'tr':
        u = 0
        l = w - new_w
    elif pos == 'bl':
        u = h - new_h
        l = 0
    elif pos == 'br':
        u = h - new_h
        l = w - new_w

    return image[u: u + new_h, l: l + new_w, :]


def random_crop(image, size, pos_choices):
    pos = random.choice(pos_choices)
    return crop(image, size, pos)


def random_aug(img):
    img = random_crop(img, [224, 224], ['c', 'tl', 'tr', 'bl', 'br'])
    img = random_fliplr(img)
    return img


def gen_crops(img):
    crops = []
    for pos in ('c', 'tl', 'tr', 'bl', 'br'):
        _crop = crop(img, [224, 224], pos)
        crops.append(_crop)
        crops.append(np.fliplr(_crop))
    return np.stack(crops)  # [10, 224, 224, 3]


class MAP:
    def __init__(self):
        self.scores = None
        self.targets = None

    def add(self, scores, targets, copy=False):
        """Scores and target are both numpy arrays"""
        if self.scores is None or self.targets is None:
            if not copy:
                self.scores = scores
                self.targets = targets
            else:
                self.scores = scores.copy()
                self.targets = targets.copy()
            return

        self.scores = np.concatenate([self.scores, scores])
        self.targets = np.concatenate([self.targets, targets])

    def value(self):
        """Deprecated."""
        if self.scores is None:
            return 0

        ap = torch.zeros(self.scores.size(1))

        for k in range(self.scores.size(1)):
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, dim=0, descending=True)
            ground_truth = targets[sortind]
            cum_positive, cum_num = 0, 0
            prec = []

            for i in range(ground_truth.size(0)):
                if ground_truth[i] == -1:
                    continue
                cum_num += 1
                if ground_truth[i] == 1:
                    cum_positive += 1
                    prec.append(cum_positive / cum_num)

            ap[k] = sum(prec) / max(cum_positive, 1)

        return ap.mean()

    def map(self):
        # copied from https://github.com/zxwu/lsvc2017
        probs = self.scores
        labels = self.targets
        mAP = np.zeros((probs.shape[1],))

        for i in range(probs.shape[1]):
            iClass = probs[:, i]
            iY = labels[:, i]
            idx = np.argsort(-iClass)
            iY = iY[idx]
            count = 0
            ap = 0.0
            skip_count = 0
            for j in range(iY.shape[0]):
                if iY[j] == 1:
                    count += 1
                    ap += count / float(j + 1 - skip_count)
                if iY[j] == -1:
                    skip_count += 1
                if count != 0:
                    mAP[i] = ap / count
        return np.mean(mAP)
        # return mAP


def occupy_gpu(model, trainer, config):
    gpus, default_gpu = config.GPUS, config.DEFAULT_GPU
    model.train()
    model.cuda(default_gpu)

    if len(gpus) > 1:
        _model = nn.DataParallel(
            model, gpus, output_device=default_gpu)
    else:
        _model = model

    data_loader = trainer.setup_dataloader('train')
    inputs, labels = next(iter(data_loader))
    inputs, labels = trainer.format_data(inputs, labels, 'train')
    _ = _model(inputs)
    import ipdb
    ipdb.set_trace()


def submodule_params(model, submodule_str):
    children = submodule_str.split('.')

    if len(children) == 1 and \
       isinstance(getattr(model, submodule_str), nn.Parameter):
        return [submodule_str]

    module = model

    for child in children:
        module = getattr(module, child)

    return [('{}.{}'.format(submodule_str, e[0]), e[1])
            for e in module.named_parameters()]


def get_param_groups(model, config):
    param_groups = []
    all_vars = list(model.named_parameters())
    optional_var_names = []
    default = None

    for cfg_param_group in config.PARAM_GROUPS:
        if cfg_param_group['params'][0] == 'default':
            default = cfg_param_group.copy()
            continue

        module_names = cfg_param_group['params']
        named_params = []
        for module_name in module_names:
            named_params.extend(submodule_params(model, module_name))
        optional_var_names.extend([e[0] for e in named_params])
        param_group = cfg_param_group.copy()
        param_group['params'] = [e[1] for e in named_params]
        param_groups.append(param_group)

    default_var_list = [
        e[1] for e in all_vars if e[0] not in optional_var_names]

    if default and default_var_list:
        default['params'] = default_var_list
        param_groups.append(default)

    return param_groups


def send_email(sender_info, receiver, subject, content, images=None):
    """Send email.

    Example content with images:
    'This is an example.
     <img src="cid:image1">'
    In this example, you provide the cid `image1` for your image. So you should
    provide it and the image path for the `images` parameter.

    Args:
      sender_info (dict): {'email':    *sender email address*,
                           'password': *sender email password*,
                           'server':   *sender smtp server* (i.e. smtp.163.com)}
      receiver     (str): Receiver email address.
      subject      (str): Email subject.
      content      (str): Text or HTML content. If images are provided, each of
        them should be attached with a cid (in your HTML tag). This cid and the
        image file path should be provided for the parameter `images`.
      images      (list): Each element is a dict with keys `path`, `cid`.
    """
    msg_root = email.mime.multipart.MIMEMultipart('related')
    msg_root['Subject'] = subject
    msg_root['From'] = sender_info['email']
    msg_root['To'] = receiver

    msg_text = email.mime.text.MIMEText(content, 'html', 'utf-8')
    msg_root.attach(msg_text)

    if images is not None:
        for image in images:
            with open(image['path'], 'rb') as f:
                msg_image = email.mime.image.MIMEImage(f.read())
            msg_image.add_header('Content-ID', '<{}>'.format(image['cid']))
            msg_root.attach(msg_image)

    smtp = smtplib.SMTP()
    smtp.connect(sender_info['server'])
    smtp.login(sender_info['email'], sender_info['password'])
    smtp.sendmail(sender_info['email'], receiver, msg_root.as_string())
    smtp.quit()
