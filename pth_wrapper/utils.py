import email.mime.image
import email.mime.multipart
import email.mime.text
import smtplib
import os

import numpy as np
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

    if not os.path.exists(state_dir) or not os.path.isdir(state_dir):
        return max_state

    if not prefix:
        raise ValueError("Specified state_dir but no prefix!")

    f_names = os.listdir(state_dir)
    for f_name in f_names:
        if not f_name.startswith(prefix):
            continue
        epoch = int(f_name.split('-')[1])
        if epoch > max_state:
            max_state = epoch
    return max_state


def init_model(
    model, state_dir='', prefix='', index=None, pretrain_path=''
):
    model.initialize()
    latest_state = check_latest_state(state_dir, prefix)

    if latest_state != -1:
        if index is not None:
            assert index >= 0 and index <= latest_state
            latest_state = index
        load_path = os.path.join(
            state_dir, '{}_{}.pth'.format(prefix, latest_state)
        )
        print('==========>> resume from {}'.format(load_path))
        model.load_state_dict(torch.load(load_path))

    elif pretrain_path and os.path.exists(pretrain_path):
        print('==========>> loading from pretrain: {}'.format(pretrain_path))
        model.load_state_dict(torch.load(pretrain_path), strict=False)

    else:
        print('==========>> build from scratch')

    print('==========>> done')
    return latest_state


class MAP:
    def __init__(self):
        self.scores = None
        self.targets = None

    def add(self, scores, targets, copy=False):
        """Scores and target are both cuda tensors"""
        scores = scores.cpu().numpy()
        targets = targets.cpu().numpy()
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

    def map(self):
        # copied from https://github.com/zxwu/lsvc2017
        probs = self.scores
        labels = self.targets
        mAP = np.zeros((probs.shape[1], ))

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
        _model = nn.DataParallel(model, gpus, output_device=default_gpu)
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
        return [(submodule_str, getattr(model, submodule_str))]

    module = model

    for child in children:
        module = getattr(module, child)

    return [
        ('{}.{}'.format(submodule_str, e[0]), e[1])
        for e in module.named_parameters()
    ]


def get_param_groups(model, config):
    param_groups = []
    default_param_group = None
    all_named_params = list(model.named_parameters())
    non_default_param_names = []

    for cfg_param_group in config.PARAM_GROUPS:
        if cfg_param_group['params'][0] == 'default':
            default_param_group = cfg_param_group.copy()
            continue

        module_names = cfg_param_group['params']
        grp_named_params = []
        for module_name in module_names:
            grp_named_params.extend(submodule_params(model, module_name))
        non_default_param_names.extend([e[0] for e in grp_named_params])

        if cfg_param_group['lr'] == 0:
            for e in grp_named_params:
                e[1].requires_grad = False
            continue

        param_group = cfg_param_group.copy()
        param_group['params'] = [
            e[1] for e in grp_named_params if e[1].requires_grad
        ]
        param_groups.append(param_group)

    default_params = [e[1] for e in all_named_params if e[0] not in non_default_param_names]

    if default_param_group and default_params:
        if default_param_group['lr'] == 0:
            for e in default_params:
                e.requires_grad = False
        else:
            default_param_group['params'] = [
                e for e in default_params if e.requires_grad
            ]
            param_groups.append(default_param_group)

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
