import torch
import models.networks as networks
import util.util as util
from models.networks.architecture import VGG19
import  torch.nn.functional as F 

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netC, self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionHis  = networks.HistogramLoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
        
        self.vgg_encoder=VGG19(requires_grad=False)


    def forward(self, data, mode):
    
        if mode == 'generator':
            a_features=self.vgg_encoder(data['nonmakeup'],output_last_feature=False)
            b_features=self.vgg_encoder(data['makeup'],output_last_feature=False)
            warped_features = self.netC(a_features, b_features, data['label_A'], data['label_B'], data['nonmakeup_unchanged'])
            g_loss, generated = self.compute_generator_loss(data['nonmakeup'],data['makeup'],data['mask_A'], data['mask_B'], data['nonmakeup_unchanged'], warped_features, a_features) 
            return g_loss, generated, data['label_A'], data['label_B'], warped_features

        elif mode == 'discriminator':
            a_features=self.vgg_encoder(data['nonmakeup'],output_last_feature=False)
            b_features=self.vgg_encoder(data['makeup'],output_last_feature=False)
            warped_features = self.netC(a_features, b_features, data['label_A'], data['label_B'], data['nonmakeup_unchanged'])
            d_loss = self.compute_discriminator_loss(data['nonmakeup'], data['makeup'], warped_features,a_features) 
            return d_loss

        elif mode == 'inference':
            a_features, b_features, warped_features = [], [], []
            a_features.append(self.vgg_encoder(data[0]['nonmakeup'], output_last_feature=False))
            b_features.append(self.vgg_encoder(data[0]['makeup'], output_last_feature=False))
            warped_features.append(self.netC(a_features[0], b_features[0], data[0]['label_A'], data[0]['label_B'], data[0]['nonmakeup_unchanged']))
            if self.opt.demo_mode == 'normal' or self.opt.beyond_mt:
                with torch.no_grad():
                    fake_image = [self.generate_fake(warped_features[0], a_features[0])]
                return fake_image  

            if self.opt.demo_mode == 'removal':
                alpha = [ 0.3, 0.6, 1]          
                rewarped_features = []     
                warped_feature = self.netC(b_features[0], a_features[0], data[0]['label_B'], data[0]['label_A'],  data[0]['makeup_unchanged'])
                for j in range(len(alpha)):
                    rewarped_features.append( [alpha[j]*warped_feature[i] + (1-alpha[j])*b_features[0][i] for i in range(self.opt.multiscale_level)] )
                with torch.no_grad():
                    fake_images = [self.generate_fake(rewarped_features[idx],b_features[0]) for idx in range(len(rewarped_features))]
                return fake_images
            
            if self.opt.demo_mode == 'interpolate':
                alpha = [ 0.3, 0.6, 1]
                rewarped_features = []
                for j in range(len(alpha)): 
                    rewarped_features.append( [alpha[j]*warped_features[0][i] + (1-alpha[j])*a_features[0][i] for i in range(self.opt.multiscale_level)] )                     
                with torch.no_grad():
                    fake_images = [self.generate_fake(rewarped_features[idx],a_features[0]) for idx in range(len(rewarped_features))]
                return fake_images

            if self.opt.demo_mode == 'multiple_refs':
                alpha = [[0.7, 0.1, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                        [0.4, 0.1, 0.4, 0.1], [0.25, 0.25, 0.25, 0.25], [0.1, 0.4, 0.1, 0.4],
                        [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.1, 0.7]]
                for i in range(1,4):
                    b_features.append(self.vgg_encoder(data[i]['makeup'], output_last_feature=False))
                    warped_features.append(self.netC(a_features[0], b_features[i], data[0]['label_A'], data[i]['label_B'], data[0]['nonmakeup_unchanged']))
                rewarped_features = []
                for j in range(len(alpha)):
                    rewarped_features.append( [alpha[j][0]*warped_features[0][i] + alpha[j][1]*warped_features[1][i] + alpha[j][2]*warped_features[2][i] +alpha[j][3]*warped_features[3][i] for i in range(len(warped_features))] )
                with torch.no_grad():
                    fake_images = [self.generate_fake(rewarped_features[idx],a_features[0]) for idx in range(len(rewarped_features))]
                return fake_images
                
            if self.opt.demo_mode == 'partial':
                lip = [F.interpolate(data[0]['mask_A']['mask_A_lip'], scale_factor=i) for i in [0.125, 0.25, 0.5, 1]]
                eye = [F.interpolate(data[0]['mask_A']['mask_A_eye_left']+data[1]['mask_B']['mask_B_eye_right'], scale_factor=i) for i in [0.125, 0.25, 0.5, 1]]
                skin = [F.interpolate(data[0]['mask_A']['mask_A_skin'], scale_factor=i) for i in [0.125, 0.25, 0.5, 1]]
                protect = [F.interpolate(data[0]['nonmakeup_unchanged'], scale_factor=i) for i in [0.125, 0.25, 0.5, 1]]
                for i in range(1,3):
                    b_features.append(self.vgg_encoder(data[i]['makeup'], output_last_feature=False))
                    warped_features.append(self.netC(a_features[0], b_features[i], data[0]['label_A'], data[i]['label_B'], data[0]['nonmakeup_unchanged']))
                rewarped_feature = [warped_features[0][i]*lip[i] + warped_features[1][i]*eye[i] + warped_features[2][i]*skin[i] + a_features[0][i]*protect[i] for i in range(self.opt.multiscale_level)]
                with torch.no_grad():
                    fake_image = [self.generate_fake(rewarped_feature, a_features[0])]
                return fake_image
            print('|demo_mode| is invalid')
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        if opt.optim=='Adam':
            optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_G = torch.optim.RMSprop(G_params, lr = G_lr)
            optimizer_D = torch.optim.RMSprop(D_params, lr = D_lr)
            
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netC = networks.define_C(opt)
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netC, netG, netD

            
    def WGAN_clamp(self):
        for p in self.netD.parameters():
            p.data.clamp_(self.opt.clamp_lower, self.opt.clamp_upper)

    def compute_generator_loss(self, input_a, input_b, mask_a, mask_b, mask_p, warped_features,a_features):
        G_losses = {}

        fake_image = self.generate_fake(warped_features,a_features)

        pred_fake, pred_real = self.discriminate(input_a, fake_image, input_b)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, 
            for_discriminator=False) * self.opt.lambda_gan

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_gan_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] , G_losses['Match']= self.criterionVGG(input_a, fake_image,input_b, warped_features, mask_p, self.opt.content_ratio)
            G_losses['VGG'] *= self.opt.lambda_vgg    
                

        #makeup loss
        #(1)eye loss
        hisloss_eye_left = self.criterionHis(fake_image,input_b,mask_a['mask_A_eye_left'],mask_b['mask_B_eye_left'],mask_a['index_A_eye_left'],input_a)*self.opt.lambda_his_eye
        hisloss_eye_right = self.criterionHis(fake_image,input_b,mask_a['mask_A_eye_right'],mask_b['mask_B_eye_right'],mask_a['index_A_eye_right'],input_a)*self.opt.lambda_his_eye
        #(2)lip loss
        hisloss_lip=self.criterionHis(fake_image,input_b,mask_a['mask_A_lip'],mask_b['mask_B_lip'],mask_a['index_A_lip'],input_a)*self.opt.lambda_his_lip
        #(3)skin loss
        hisloss_skin=self.criterionHis(fake_image,input_b,mask_a['mask_A_skin'],mask_b['mask_B_skin'],mask_a['index_A_skin'],input_a)*self.opt.lambda_his_skin
        G_losses['Makeup']=hisloss_eye_left+hisloss_eye_right+hisloss_lip+hisloss_skin

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_a, input_b, warped_features,a_features):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(warped_features,a_features)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(input_a, fake_image, input_b)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, warped_features,a_features):
       
        fake_image = self.netG(warped_features,a_features)

        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_a, fake_image, input_b):
        fake_concat = torch.cat([input_a, fake_image], dim=1)
        real_concat = torch.cat([input_a, input_b], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0