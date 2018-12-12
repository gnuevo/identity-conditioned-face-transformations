import os
import argparse
from solver import Solver
from data_loader_celeba import get_loader
from torch.backends import cudnn
import datetime
from collections import OrderedDict
import sys
import json


def str2bool(v):
    return v.lower() in ('true')


def write_configuration(configuration, file):
    data = OrderedDict()
    cmd = ' '.join(sys.argv)
    data['time'] = configuration.time
    data['cmd'] = cmd
    data['command'] = OrderedDict()
    for arg in vars(configuration):
        data['command'][arg] = getattr(configuration, arg)
    with open(file, 'w') as f:
        f.write(json.dumps(data, indent=4))


def read_validation(code):
    """Reads the information in the 'validation' argument from the command line

    Args:
        code:

    Returns: a tuple with the validation information, it can be 2 integers
    or two lists of images

    """
    error = False
    # try '<int>,<int>'
    values = code.split(',')
    if len(values) == 2:
        try:
            num_val_input = int(values[0])
            num_val_cond = int(values[1])
            return num_val_input, num_val_cond
        except:
            error = True

    # try 'path to validation file'
    try:
        with open(code, 'r') as f:
            data = json.load(f)
        try:
            val_input = data['validation']['samples']['input']
            val_cond = data['validation']['samples']['conditioning']
            return val_input, val_cond
        except KeyError:
            print("ERROR: wrong format of the validation information")
            raise
    except FileNotFoundError:
        print("Validation file not found.")
        raise
    except:
        error = True

    if error:
        raise ValueError("Unrecognised format of validation information"
                         "Remember it must be 2 int values separated by a "
                         "comma: <num_val_input>,<num_val_cond> "
                         "or the path to a file with the validation "
                         "information.")


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.validation_dir):
        os.makedirs(config.validation_dir)

    # save configuration
    write_configuration(config, os.path.join(config.path, "config.json"))

    validation = read_validation(config.validation)

    # Data loader.
    celeba_loader = get_loader(config.celeba_image_dir,
                               config.metadata_path,
                               config.crop_size,
                               config.image_size,
                               config.batch_size,
                               config.mode,
                               validation=validation)

    # save validation samples
    val_input = celeba_loader.dataset.val_input
    val_cond = celeba_loader.dataset.val_cond
    with open(os.path.join(config.path, "validation.json"), 'w') as f:
        f.write(json.dumps({
            "validation": {
                "number": (len(val_input), len(val_cond)),
                "samples": {
                    "input": val_input,
                    "conditioning": val_cond
                }
            }
        }, indent=4))

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, None, config)

    if config.mode == 'train':
        solver.train_icfat()
    elif config.mode == 'test':
        raise NotImplementedError()
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # New commands
    parser.add_argument('--margin', type=float, default=0.8, help='margin of the triplet loss')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--embed_dim', type=int, default=10, help='dimension of the id embedding')
    parser.add_argument('--crop_size', type=int, default=10, help='crop size')
    parser.add_argument('--metadata_path', type=str,
                        help='path of the id metadata file')
    parser.add_argument('--num_epochs_decay', type=int, default=10,
                        help='dummy for now')
    parser.add_argument('--decay_rate', type=float, default=10.0, help='dummy '
                                                                   'for now')
    parser.add_argument('--decay_step', type=int, default=10, help='dummy '
                                                                   'for now')
    parser.add_argument('--steps_per_epoch', type=int, default=10,
                        help='dummy for now')



    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--save_on_epoch', default=False,
                        action="store_true",
                        help='If true then --model_save_step refers to '
                             'epochs, otherwise iterations')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--validation', type=str, default="100,25",
                        help="'<num_val_input>,<num_val_cond' or path to "
                             "json file with validation information")

    # Directories.
    t = datetime.datetime.now().strftime('%G-%m-%d_%H:%M:%S')
    path = 'icGAN_{}'.format(t)
    parser.add_argument('--celeba_image_dir', type=str, default='data/CelebA_nocrop/images')
    parser.add_argument('--attr_path', type=str, default='data/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='{}/logs'.format(path))
    parser.add_argument('--model_save_dir', type=str, default='{}/models'.format(path))
    parser.add_argument('--sample_dir', type=str, default='{}/samples'.format(path))
    parser.add_argument('--result_dir', type=str, default='{}/results'.format(path))
    parser.add_argument('--validation_dir', type=str, default='{}/validation'.format(path))

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    config.time = t # starting time
    config.path = path # path to the training folder
    print(config)
    main(config)