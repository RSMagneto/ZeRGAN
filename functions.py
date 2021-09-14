from skimage import io as skimage
import torch
import pdb
import numpy
import math
from imresize import imresize
import model as models

def imreadImg(path, opt):
    x=skimage.imread(path)
    x = x.astype(float)
    if x.ndim == 2:
        x = x[:, :, None, None]
        max_x = x.max()
        x = x.transpose(2, 3, 0, 1)/max_x
    elif x.ndim == 3:
        x = x[:, :, :, None]
        x = x.transpose(3, 2, 0, 1)
        max_x = x.max()
        x = x/max_x
    x = torch.from_numpy(x)
    x = x.to(opt.device).type(torch.cuda.FloatTensor)
    x = (x-0.5)*2
    x = x.clamp(-1, 1)
    return x, max_x

def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)

def convert_image_np(inp, max_x, opt):
    inp = denorm(inp)
    if inp.shape[1]==4 or inp.shape[1]==3 or inp.shape[1]==5 or inp.shape[1] == 8:
        inp = inp*max_x
        inp = inp[-1,:,:,:]
        inp = inp.to(torch.device('cpu'))
        inp = inp.numpy().transpose((0,1,2))
    else:
        inp = inp*max_x
        inp = inp[-1, -1, :, :]
        inp = inp.to(torch.device('cpu'))
        inp = inp.numpy().transpose((0, 1))
    return inp

def convert_image_mat(inp, max_x, opt):
    inp = denorm(inp)
    inp = inp * max_x
    inp = inp[-1, :, :, :]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1, 2, 0))
    return inp

# def generate_noise(size, num_samp=1, device='cuda:0', type='gaussian', scale=1):
#     if type == 'gaussian':
#         noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
#         noise = upsampling(noise,size[1], size[2])
#     if type =='gaussian_mixture':
#         noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
#         noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
#         noise = noise1+noise2
#     if type == 'uniform':
#         noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
#     return noise

# def upsampling(im,sx,sy):
#     m = torch.nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
#     return m(im)

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# def calc_init_scale(opt):
#     in_scale = math.pow(1/2, 1/3)
#     iter_num = round(math.log(1 / opt.sr_factor, in_scale))
#     in_scale = pow(opt.sr_factor, 1 / iter_num)
#     return in_scale, iter_num

# def adjust_scales2image_SR(real_,opt):
#     opt.min_size = 64
#     opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor))) + 1
#     scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
#     opt.stop_scale = opt.num_scales - scale2stop
#     opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
#     real = imresize(real_, opt.scale1, opt)
#     opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2], real.shape[3])), 1/(opt.stop_scale))
#     scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
#     opt.stop_scale = opt.num_scales - scale2stop
#     return real

def creat_reals_pyramid(real,reals,opt):
    for i in range(0,opt.scale_num,1):
        scale = math.pow(0.5, (2 / 4) * (opt.scale_num - i - 1))
        curr_real = torch.nn.functional.interpolate(real, size=(math.ceil(real.shape[2]*scale), math.ceil(real.shape[3]*scale)), mode='bilinear')
        reals.append(curr_real)
    return reals

def weight(outMS):
    R = outMS[:, 0, :, :]
    G = outMS[:, 1, :, :]
    B = outMS[:, 2, :, :]
    N = outMS[:, 3, :, :]
    outP = 0.25 * R + 0.25 * G + 0.25 * B + 0.25 * N
    outP = outP[:, None, :, :]
    return outP

def gradientLoss_MS(middle_image,opt):
    channelsGradient_x=numpy.zeros([middle_image.shape[2],middle_image.shape[3]])
    channelsGradient_x=(torch.tensor(channelsGradient_x, requires_grad=True)).to(opt.device).type(torch.cuda.FloatTensor)
    channelsGradient_y=numpy.zeros([middle_image.shape[2],middle_image.shape[3]])
    channelsGradient_y=(torch.tensor(channelsGradient_y, requires_grad=True)).to(opt.device).type(torch.cuda.FloatTensor)
    for i in range(4):
        x,y=gradient(middle_image[:,i,:,:],opt)
        channelsGradient_x=0.25*x+channelsGradient_x
        channelsGradient_y=0.25*y+channelsGradient_y
    return channelsGradient_x[None,None,:,:],channelsGradient_y[None,None,:,:]

def gradientLoss_P(pan_image,opt):
    pan_image = pan_image[-1, :, :, :]
    grayGradient_x,grayGradient_y=gradient(pan_image, opt)
    return grayGradient_x[None, None, :, :], grayGradient_y[None, None, :, :]

def gradient(image,opt):
     image = image.to(torch.device('cpu'))
     image = numpy.array(image.detach())
     dx, dy = numpy.gradient(image[0,:,:], edge_order=1)
     dx = (torch.tensor(dx, requires_grad=True)).to(opt.device).type(torch.cuda.FloatTensor)
     dy = (torch.tensor(dy, requires_grad=True)).to(opt.device).type(torch.cuda.FloatTensor)
     return dx, dy

# def init_models(opt,i):
#     netG = models.Generator(opt).to(opt.device)
#     netG.apply(models.weights_init)
#     netG.load_state_dict(torch.load('netG_%s.pth'%(i)))
#     print(i)
#     print(netG)
#     return netG

class GuassianBlur(torch.nn.Module):
    def __init__(self, channels=4):
        super(GuassianBlur, self).__init__()
        self.channels = channels
        kernel = [[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = numpy.repeat(kernel, self.channels, axis=0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x
