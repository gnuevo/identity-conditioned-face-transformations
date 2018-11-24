from model import Generator
from model import Discriminator
from model import DomainDiscriminator, DomainGenerator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from collections import OrderedDict


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.c3_dim = config.embed_dim #TODO maybe rename
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_epochs = config.epochs #TODO maybe rename
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.margin = config.margin
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        # if self.dataset in ['CelebA', 'RaFD']:
        #     self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        #     self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        # elif self.dataset in ['Both']:
        #     self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
        #     self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        #
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        self.G = DomainGenerator(self.g_conv_dim, self.c3_dim, self.g_repeat_num)
        self.D_src = DomainDiscriminator(image_size=self.image_size,
                                         conv_dim=self.d_conv_dim,
                                         num_domains=1,
                                         length_domain=self.c3_dim,
                                         repeat_num=self.d_repeat_num,
                                         extra_layers=1)
        self.D_cls = DomainDiscriminator(image_size=self.image_size,
                                         conv_dim=self.d_conv_dim,
                                         num_domains=1,
                                         length_domain=self.c3_dim,
                                         repeat_num=self.d_repeat_num,
                                         extra_layers=1)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr,
                                            [self.beta1, self.beta2])
        self.dsrc_optimizer = torch.optim.Adam(self.D_src.parameters(), self.g_lr,
                                            [self.beta1, self.beta2])
        self.dcls_optimizer = torch.optim.Adam(self.D_cls.parameters(), self.g_lr,
                                            [self.beta1, self.beta2])

        # print networks
        self.print_network(self.G, 'Generator')
        self.print_network(self.D_src, 'Source Discriminator (D_src)')
        self.print_network(self.D_cls, 'Classifier Discriminator (D_cls)')

        # send to GPU if available
        self.G.to(self.device)
        self.D_src.to(self.device)
        self.D_cls.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        raise NotImplementedError()
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.dsrc_optimizer.zero_grad()
        self.dcls_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train_icfat(self):
        # TODO ###### ##  ###   ## ###############
        # TODO   ##  #  # #  # #  # ##############
        # TODO   ##  #  # #  # #  # ##############
        # TODO   ##  #  # #  # #  # ##############
        # TODO   ##   ##  ###   ## ###############
        """Train StarGAN within a single dataset."""
        # # Set data loader.
        # if self.dataset == 'CelebA':
        #     data_loader = self.celeba_loader
        # elif self.dataset == 'RaFD':
        #     data_loader = self.rafd_loader
        #
        # # Fetch fixed inputs for debugging.
        # data_iter = iter(data_loader)
        # x_fixed, c_org = next(data_iter)
        # x_fixed = x_fixed.to(self.device)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset,
        #                                   self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        # FIXME this will need to change
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        fixed_x = []
        total = 50
        for i, (batchA, batchP, batchN) in enumerate(self.celeba_loader):
            for imageA, imageP, imageN in zip(batchA, batchP, batchN):
                print('Reading debugging images', i, total)
                fixed_x.append((imageA.unsqueeze(0), imageN.unsqueeze(0)))
                total -= 1
                if total == 0: break
            if total == 0: break
        black_size = [1]
        black_size.extend(imageA.size())

        # TODO retrieve validation information
        # TODO scorer

        criterion = torch.nn.MarginRankingLoss(margin=self.margin)

        self.data_loader = self.celeba_loader
        iters_per_epoch = len(self.data_loader)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for e in range(start_iters, self.num_epochs):
            for i, (batchA, batchP, batchN) in enumerate(self.data_loader):
                # FIXME delete remanenet from the time where you had to
                # convert to variables
                imageA = batchA.to(self.device)
                imageP = batchP.to(self.device)
                imageN = batchN.to(self.device)

                # ================== Train D_cls on CelebA ================== #
                # Compute loss with real images
                _, idA = self.D_cls(imageA)
                _, idP = self.D_cls(imageP)
                _, idN = self.D_cls(imageN)

                d_loss_cls = criterion(
                    F.pairwise_distance(idA, idN),
                    F.pairwise_distance(idA, idP),
                    torch.ones((idA.size(0), 1), device=self.device)
                )

                # Compute classification accuracy of D_cls
                d = {
                    'D_cls/distance_same':
                        torch.mean(F.pairwise_distance(idA, idP)).item(),
                    'D_cls/distance_different':
                        torch.mean(F.pairwise_distance(idA, idN)).item()
                }
                log = []
                log.extend([d['D_cls/distance_same'], d['D_cls/distance_different']])
                if(i + 1) % self.log_step == 0:
                    print('Classification distances (same/different): ', end='')
                    print(log)

                # Logging
                loss = OrderedDict()
                loss['D_cls/loss_cls'] = d_loss_cls.item()

                # ================== Train D_src on CelebA ================== #

                # Compute loss with real images
                out_srcA, _ = self.D_src(imageA)
                d_loss_real = - torch.mean(out_srcA)

                # Compute loss with fake images
                fake_x = self.G(imageA, idN)
                out_src_fake, _ = self.D_src(fake_x)
                d_loss_fake = torch.mean(out_src_fake)

                # ================== Optimization =========================== #

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake

                # Backward + Optimize D's
                e_loss = d_loss + self.lambda_cls * d_loss_cls
                self.reset_grad()
                e_loss.backward()
                self.dsrc_optimizer.step()
                self.dcls_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(imageA.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * imageA.data + (
                        1 - alpha) * fake_x.data).requires_grad_(True)
                out_src, _ = self.D_src(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # FIXME don't we have to optimize D_cls as well with GP?!
                # FIXME if not, it could explain why at the end of training
                # the GP error increases so much!
                _, out_cls = self.D_cls(x_hat)
                d_loss_gp += self.gradient_penalty(out_cls, x_hat)

                # Backward + Optimize

                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.dsrc_optimizer.step()

                # Logging
                loss = OrderedDict()
                loss['D_src/loss_real'] = d_loss_real.item()
                loss['D_src/loss_fake'] = d_loss_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # ================== Train G ================== #

                if (i + 1) % self.n_critic == 0:
                    # FIXME calling the network here is unnecessary
                    _, idA = self.D_cls(imageA)
                    _, idP = self.D_cls(imageP)
                    _, idN = self.D_cls(imageN)

                    # Original-to-target and target-to-original domain
                    fake_c = idN
                    real_c = idP
                    fake_x = self.G(imageA, fake_c)
                    rec_x = self.G(fake_x, real_c)

                    # Compute losses
                    out_src, _ = self.D_src(fake_x)
                    _, idG = self.D_cls(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(imageA - rec_x))

                    # fake_label_AG = self.to_var(torch.zeros((len(idA), 1)))
                    fake_label_BG = torch.ones((len(idA), 1))

                    g_loss_cls = criterion(
                        F.pairwise_distance(idG, idA),
                        F.pairwise_distance(idG, idN),
                        torch.ones((idA.size(0), 1), device=self.device)
                    )

                    # Backward + Optimize
                    g_loss = g_loss_fake \
                             + self.lambda_rec * g_loss_rec \
                             + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                    # Compute classification accuracy of the discriminator
                    # FIXME I think there's a problem here, shouldn't it be
                    # positive_distance the one between idN and idG?
                    positive_distance = torch.sum(F.pairwise_distance(idP,
                                                                      idG)).item()
                    negative_distance = torch.sum(F.pairwise_distance(idA,
                                                                      idG)).item()
                    d = {
                        'G/Distance_same': positive_distance / len(idG),
                        'G/Distance_different': negative_distance / len(idG)
                    }
                    # FIXME scorer.add(d)

                # Print log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e + 1, self.num_epochs, i + 1, iters_per_epoch
                    )
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    # FIXME scores stuff!

            # ================== Debugging images ================== #
            if (e + 1) % self.sample_step == 0:
                debugging_samples = []
                A_images = [a for a, b in fixed_x]
                B_images = [b for a, b in fixed_x]
                for a, b in zip(A_images, B_images):
                    row = [b.to('cpu'), a.to('cpu')]
                    _, idA = self.D_cls(a)
                    _, idB = self.D_cls(b)
                    fake_image = self.G(a, idB)
                    rec_image = self.G(fake_image, idA)
                    row.append(fake_image)
                    row.append(rec_image)
                    debugging_samples.append(torch.cat(row, dim=3))
                fake_images = torch.cat(debugging_samples, dim=2)
                save_image(self.denorm(fake_images.data.cpu()),
                           os.path.join(self.sample_dir,
                                        "{}_{}_fake.png".format(e + 1, i + 1)),
                           nrow=1,
                           padding=0)
                print("Translated images and saved to {}..!".format(self.sample_dir))

            # TODO validation images!

            # ================== Checkpoints and lr decay ================== #
            # Save model checkpoints
            # TODO add posibility of saving in the middle of an epoch
            if (e + 1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_dir,
                                        "{}_{}_G.pth".format(e + 1, i + 1)))
                torch.save(self.D_src.state_dict(),
                           os.path.join(self.model_save_dir,
                                        "{}_{}_D_src.pth".format(e + 1, i + 1)))
                torch.save(self.D_cls.state_dict(),
                           os.path.join(self.model_save_dir,
                                        "{}_{}_D_cls.pth".format(e + 1, i + 1)))
                print("Saved models..!")

            if (e + 1) > (self.num_epochs - self.num_epochs_decay):
                if (e - (self.num_epochs - self.num_epochs_decay)) % \
                        self.decay_step == 0:
                    g_lr -= (self.g_lr / float(self.decay_rate))
                    d_lr -= (self.d_lr / float(self.decay_rate))
                    print("Decay learning rate to g_lr: {}, d_lr: {"
                          "}".format(g_lr, d_lr))
                    assert g_lr > 0.0
                    assert d_lr > 0.0

        # Save model checkpoints when training is done
        torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_dir,
                                        "{}_{}_G.pth".format(e + 1, i + 1)))
        torch.save(self.D_src.state_dict(),
                   os.path.join(self.model_save_dir,
                                "{}_{}_D_src.pth".format(e + 1, i + 1)))
        torch.save(self.D_cls.state_dict(),
                   os.path.join(self.model_save_dir,
                                "{}_{}_D_cls.pth".format(e + 1, i + 1)))
        print("Saved models..!")

        print("Trained finished")



            #
            #     # =================================================================================== #
            #     #                             1. Preprocess input data                                #
            #     # =================================================================================== #
            #
            #     # Fetch real images and labels.
            #     try:
            #         x_real, label_org = next(data_iter)
            #     except:
            #         data_iter = iter(data_loader)
            #         x_real, label_org = next(data_iter)
            #
            #     # Generate target domain labels randomly.
            #     rand_idx = torch.randperm(label_org.size(0))
            #     label_trg = label_org[rand_idx]
            #
            #     if self.dataset == 'CelebA':
            #         c_org = label_org.clone()
            #         c_trg = label_trg.clone()
            #     elif self.dataset == 'RaFD':
            #         c_org = self.label2onehot(label_org, self.c_dim)
            #         c_trg = self.label2onehot(label_trg, self.c_dim)
            #
            #     x_real = x_real.to(self.device)  # Input images.
            #     c_org = c_org.to(self.device)  # Original domain labels.
            #     c_trg = c_trg.to(self.device)  # Target domain labels.
            #     label_org = label_org.to(
            #         self.device)  # Labels for computing classification loss.
            #     label_trg = label_trg.to(
            #         self.device)  # Labels for computing classification loss.
            #
            #     # =================================================================================== #
            #     #                             2. Train the discriminator                              #
            #     # =================================================================================== #
            #
            #     # Compute loss with real images.
            #     out_src, out_cls = self.D(x_real)
            #     d_loss_real = - torch.mean(out_src)
            #     d_loss_cls = self.classification_loss(out_cls, label_org,
            #                                           self.dataset)
            #
            #     # Compute loss with fake images.
            #     x_fake = self.G(x_real, c_trg)
            #     out_src, out_cls = self.D(x_fake.detach())
            #     d_loss_fake = torch.mean(out_src)
            #
            #     # Compute loss for gradient penalty.
            #     alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            #     x_hat = (alpha * x_real.data + (
            #                 1 - alpha) * x_fake.data).requires_grad_(True)
            #     out_src, _ = self.D(x_hat)
            #     d_loss_gp = self.gradient_penalty(out_src, x_hat)
            #
            #     # Backward and optimize.
            #     d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            #     self.reset_grad()
            #     d_loss.backward()
            #     self.d_optimizer.step()
            #
            #     # Logging.
            #     loss = {}
            #     loss['D/loss_real'] = d_loss_real.item()
            #     loss['D/loss_fake'] = d_loss_fake.item()
            #     loss['D/loss_cls'] = d_loss_cls.item()
            #     loss['D/loss_gp'] = d_loss_gp.item()
            #
            #     # =================================================================================== #
            #     #                               3. Train the generator                                #
            #     # =================================================================================== #
            #
            #     if (i + 1) % self.n_critic == 0:
            #         # Original-to-target domain.
            #         x_fake = self.G(x_real, c_trg)
            #         out_src, out_cls = self.D(x_fake)
            #         g_loss_fake = - torch.mean(out_src)
            #         g_loss_cls = self.classification_loss(out_cls, label_trg,
            #                                               self.dataset)
            #
            #         # Target-to-original domain.
            #         x_reconst = self.G(x_fake, c_org)
            #         g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
            #
            #         # Backward and optimize.
            #         g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
            #         self.reset_grad()
            #         g_loss.backward()
            #         self.g_optimizer.step()
            #
            #         # Logging.
            #         loss['G/loss_fake'] = g_loss_fake.item()
            #         loss['G/loss_rec'] = g_loss_rec.item()
            #         loss['G/loss_cls'] = g_loss_cls.item()
            #
            # # =================================================================================== #
            # #                                 4. Miscellaneous                                    #
            # # =================================================================================== #
            #
            # # Print out training information.
            # if (i + 1) % self.log_step == 0:
            #     et = time.time() - start_time
            #     et = str(datetime.timedelta(seconds=et))[:-7]
            #     log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1,
            #                                                    self.num_iters)
            #     for tag, value in loss.items():
            #         log += ", {}: {:.4f}".format(tag, value)
            #     print(log)
            #
            #     if self.use_tensorboard:
            #         for tag, value in loss.items():
            #             self.logger.scalar_summary(tag, value, i + 1)
            #
            # # Translate fixed images for debugging.
            # if (i + 1) % self.sample_step == 0:
            #     with torch.no_grad():
            #         x_fake_list = [x_fixed]
            #         for c_fixed in c_fixed_list:
            #             x_fake_list.append(self.G(x_fixed, c_fixed))
            #         x_concat = torch.cat(x_fake_list, dim=3)
            #         sample_path = os.path.join(self.sample_dir,
            #                                    '{}-images.jpg'.format(i + 1))
            #         save_image(self.denorm(x_concat.data.cpu()), sample_path,
            #                    nrow=1, padding=0)
            #         print('Saved real and fake images into {}...'.format(
            #             sample_path))
            #
            # # Save model checkpoints.
            # if (i + 1) % self.model_save_step == 0:
            #     G_path = os.path.join(self.model_save_dir,
            #                           '{}-G.ckpt'.format(i + 1))
            #     D_path = os.path.join(self.model_save_dir,
            #                           '{}-D.ckpt'.format(i + 1))
            #     torch.save(self.G.state_dict(), G_path)
            #     torch.save(self.D.state_dict(), D_path)
            #     print('Saved model checkpoints into {}...'.format(
            #         self.model_save_dir))
            #
            # # Decay learning rates.
            # if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
            #         self.num_iters - self.num_iters_decay):
            #     g_lr -= (self.g_lr / float(self.num_iters_decay))
            #     d_lr -= (self.d_lr / float(self.num_iters_decay))
            #     self.update_lr(g_lr, d_lr)
            #     print(
            #         'Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr,
            #                                                              d_lr))

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))