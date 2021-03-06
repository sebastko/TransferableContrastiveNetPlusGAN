import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn

from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import argparse
import os
import glob
import random

from dataset_loader import FeatDataLayer, DATA_LOADER
from models import _AttributeNet, _RelationNet, _param

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY', help='FLO')
parser.add_argument('--dataroot', default='../datasets/data_resnet',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')

parser.add_argument('--gpu', default='1', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, default=6278, help='manual seed')
parser.add_argument('--resume', type=str, help='the model to resume')

parser.add_argument('--z_dim', type=int, default=100, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=500)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval', type=int, default=1000)

opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.REG_W_LAMBDA = 0

opt.lr = 0.00001
opt.batch_size = 32  # 512

""" hyper-parameter for testing"""

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)


def train():
    param = _param()
    dataset = DATA_LOADER(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), opt)
    result = Result()
    result_gzsl = Result()

    print
    dataset.att_dim, dataset.feature_dim
    APnet = _AttributeNet(dataset.att_dim, dataset.feature_dim).cuda()
    APnet.apply(weights_init)
    print(APnet)

    Rnet = _RelationNet(dataset.feature_dim).cuda()
    Rnet.apply(weights_init)
    print(Rnet)

    exp_info = 'GBU_{}'.format(opt.dataset)
    exp_params = 'Rls{}'.format(opt.REG_W_LAMBDA)

    out_dir = 'Result_baseline/{:s}'.format(exp_info)
    out_subdir = 'Result_baseline/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('Result_baseline'):
        os.mkdir('Result_baseline')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log_{:s}_{}.txt'.format(exp_info, opt.exp_idx)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            APnet.load_state_dict(checkpoint['state_dict_AP'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    optimizerAP = optim.Adam(APnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    nets = [APnet, Rnet]

    acc = 0
    H = 0
    S = 0
    U = 0

    for it in range(start_step, 50000 + 1):
        blobs = data_layer.forward()
        batch_feats = blobs['data']  # image data
        batch_labels = blobs['labels'].astype(int)  # class labels

        support_attr = dataset.attribute[dataset.train_class].cuda()
        class_num = dataset.train_class.shape[0]

        batch_images = torch.from_numpy(batch_feats).cuda()

        batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        batch_ext = torch.transpose(batch_ext, 0, 1)

        # forward
        semantic_proto = APnet(support_attr)
        semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(opt.batch_size, 1, 1)
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)

        # label transform
        support_labels = dataset.train_class.numpy()
        re_batch_labels = np.zeros_like(batch_labels)
        for i in range(class_num):
            re_batch_labels[batch_labels == support_labels[i]] = i
        re_batch_labels = torch.LongTensor(re_batch_labels)

        bce = nn.BCELoss().cuda()
        one_hot_labels = torch.zeros(opt.batch_size, class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
        loss = bce(relations, one_hot_labels)

        reset_grad(nets)

        loss.backward()

        optimizerAP.step()
        optimizerR.step()

        if it % opt.evl_interval == 0 and it >= 100:
            APnet.eval()
            Rnet.eval()
            acc = eval_test(it, APnet, Rnet, dataset, param, result)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                # best_acc = result.acc_list[-1]
                save_model(it, APnet, Rnet, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_ZSL_Acc_{:.2f}.tar'.format(result.acc_list[-1]))

            H, S, U = eval_test_gzsl(it, APnet, Rnet, dataset, param, result_gzsl)
            if result_gzsl.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                # best_acc_gzsl = result.acc_list[-1]
                save_model(it, APnet, Rnet, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl.best_acc,
                                                                                                 result_gzsl.best_acc_S_T,
                                                                                                 result_gzsl.best_acc_U_T))

            APnet.train()
            Rnet.train()

        if it % opt.save_interval == 0 and it:
            save_model(it, APnet, Rnet, opt.manualSeed, log_text,
                       out_subdir + '/Iter_{:d}.tar'.format(it))
            cprint('Save model to ' + out_subdir + '/Iter_{:d}.tar'.format(it), 'red')

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-{}; Loss: {:.3f}; ZSL: {:.3f}; H: {:.3f}; S: {:.3f}; U: {:.3f};  ' \
                .format(it, loss.data[0], acc, H, S, U)
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')


def save_model(it, APnet, Rnet, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_AP': APnet.state_dict(),
        'state_dict_R': Rnet.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def eval_test(it, APnet, Rnet, dataset, param, result):
    support_attr = torch.from_numpy(dataset.test_att).cuda()
    class_num = support_attr.shape[0]

    # label transform
    support_labels = np.array(dataset.unseenclasses)
    test_labels = np.array(dataset.test_unseen_label)

    test_images = dataset.test_unseen_feature.cuda()
    preds = np.zeros_like(test_labels)

    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        batch_ext = torch.transpose(batch_ext, 0, 1)

        # forward
        semantic_proto = APnet(support_attr)
        semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        prob = relations.data.cpu().numpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    grounds = test_labels
    predicts = support_labels[preds]

    acc = np.zeros(class_num)
    for i in range(class_num):
        index = grounds == support_labels[i]
        acc[i] = (predicts[index] == support_labels[i]).mean()
    acc = acc.mean() * 100

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("Accuracy is {:.2f}%".format(acc))
    return acc


def eval_test_gzsl(it, APnet, Rnet, dataset, param, result):
    support_attr = dataset.attribute.cuda()
    class_num = support_attr.shape[0]

    # label transform
    support_labels = np.array(dataset.allclasses)

    """  S -> T
    """
    test_labels = np.array(dataset.test_seen_label)
    test_images = dataset.test_seen_feature.cuda()
    preds = np.zeros_like(test_labels)

    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        batch_ext = torch.transpose(batch_ext, 0, 1)

        # forward
        semantic_proto = APnet(support_attr)
        semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        prob = relations.data.cpu().numpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    label_T = test_labels
    num_seen_classes = dataset.ntrain_class
    seen_classes = np.array(dataset.seenclasses)

    acc = np.zeros(num_seen_classes)
    for i in range(num_seen_classes):
        acc[i] = (preds[label_T == seen_classes[i]] == seen_classes[i]).mean()
    acc_S_T = acc.mean() * 100

    """  U -> T
    """
    test_labels = np.array(dataset.test_unseen_label)
    test_images = dataset.test_unseen_feature.cuda()
    preds = np.zeros_like(test_labels)

    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        batch_ext = torch.transpose(batch_ext, 0, 1)

        # forward
        semantic_proto = APnet(support_attr)
        semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        prob = relations.data.cpu().numpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    label_T = test_labels
    num_unseen_classes = dataset.ntest_class
    unseen_classes = np.array(dataset.unseenclasses)

    acc = np.zeros(num_unseen_classes)
    for i in range(num_unseen_classes):
        acc[i] = (preds[label_T == unseen_classes[i]] == unseen_classes[i]).mean()
    acc_U_T = acc.mean() * 100

    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True

    print("H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(acc, acc_S_T, acc_U_T))
    return acc, acc_S_T, acc_U_T


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


if __name__ == "__main__":
    train()