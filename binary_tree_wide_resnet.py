import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Binary_Tree_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, k=2, strides=(2, 1)):
        super(Binary_Tree_Conv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.strides = strides
        modules = []
        for i in six.moves.range(k):
            modules += [('bn_relu_conv{}_leaf'.format(i), BN_ReLU_Conv(in_channel, int(out_channel / 2), 3, strides[i], 1))]
            modules += [('bn_relu_conv{}_branch'.format(i), BN_ReLU_Conv(in_channel, int(out_channel / 2), 3, strides[i], 1))]
            in_channel = int(out_channel / 2)
            out_channel = int(out_channel / 2)
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.strides:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        pure_x = x
        H = []
        for i in six.moves.range(self.k):
            H.append(self['bn_relu_conv{}_leaf'.format(i)](x, train=train))
            x = self['bn_relu_conv{}_branch'.format(i)](x, train=train)
        self.H = H
        H.append(x)
        h = F.concat(H, axis=1)
        x = h + Binary_Tree_Conv.concatenate_zero_pad(self.maybe_pooling(pure_x), h.data.shape, h.volatile, type(h.data))
        return x


class BitBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel=3, out_channels=(64, 64), k=6, strides=((1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))):
        super(BitBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.k = k
        self.strides = strides
        modules = []
        for i in six.moves.range(len(out_channels)):
            modules += [('binary_tree_conv{}'.format(i), Binary_Tree_Conv(in_channel, out_channels[i], k=k, strides=strides[i]))]
            in_channel = out_channels[i]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        for i in six.moves.range(len(self.out_channels)):
            x = self['binary_tree_conv{}'.format(i)](x, train=train)
        return x


class BitNet(nutszebra_chainer.Model):

    def __init__(self, category_num, out_channels=(16 * 4, 32 * 4, 64 * 4), N=(3, 3, 3), K=(4, 4, 4), strides=(1, 2, 2)):
        super(BitNet, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, 16, 3, 1, 1))]
        in_channel = 16
        for i in six.moves.range(len(out_channels)):
            k = K[i]
            out_channel = (out_channels[i], ) * N[i]
            stride = [[1 for i in six.moves.range(k)] for _ in six.moves.range(N[i])]
            stride[0][0] = strides[i]
            modules.append(('bitblock{}'.format(i), BitBlock(in_channel, out_channels=out_channel, k=k, strides=stride)))
            in_channel = out_channel[-1]
        modules.append(('bn_relu_conv', BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.out_channels = out_channels
        self.N = N
        self.K = K
        self.strides = strides
        self.name = 'truncated_wide_networks_by_binary_trees_{}_{}_{}'.format(category_num, K[0], out_channels[0])

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        [link.weight_initialization() for _, link in self.modules[1:]]

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        count += int(np.sum([link.count_parameters() for _, link in self.modules[1:]]))

        return count

    def __call__(self, x, train=False):
        h = self.conv1(x)
        for i in six.moves.range(len(self.out_channels)):
            h = self['bitblock{}'.format(i)](h, train=train)
            print(h.data.shape)
        h = self.bn_relu_conv(h, train=train)
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
