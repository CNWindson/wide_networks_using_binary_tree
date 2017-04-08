import sys
sys.path.append('./trainer')
import argparse
import nutszebra_cifar10
import binary_tree_wide_resnet
import nutszebra_data_augmentation
import nutszebra_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=200,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=128,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=1,
                        help='divid batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=1,
                        help='divid batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='leraning rate')
    parser.add_argument('--d', '-d', type=int,
                        default=4,
                        help='d in https://arxiv.org/abs/1704.00509')
    parser.add_argument('--k', '-k', type=int,
                        default=6,
                        help='k in https://arxiv.org/abs/1704.00509')
    parser.add_argument('--n', '-n', type=int,
                        default=2,
                        help='n in https://arxiv.org/abs/1704.00509')

    args = parser.parse_args().__dict__
    print(args)
    lr = args.pop('lr')
    d = args.pop('d')
    k = args.pop('k')
    n = args.pop('n')

    print('generating model')
    model = binary_tree_wide_resnet.BitNet(10, out_channels=(16 * d, 32 * d, 64 * d), N=(n, ) * 3, K=(k, ) * 3, strides=(1, 2, 2))
    print('Done')
    print('Number of parameters: {}'.format(model.count_parameters()))
    optimizer = nutszebra_optimizer.OptimizerWideResBinaryTree(model, lr=lr)
    args['model'] = model
    args['optimizer'] = optimizer
    args['da'] = nutszebra_data_augmentation.DataAugmentationCifar10NormalizeSmall
    main = nutszebra_cifar10.TrainCifar10(**args)
    main.run()
