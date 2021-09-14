import argparse
import model
import torch
import functions
import math
from time import *
import skimage.io
import numpy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', required=True)
    parser.add_argument('--input_ms', help='training lrms image name',  required=True)
    parser.add_argument('--input_pan', help='training pan image name', required=True)
    parser.add_argument('--channels', help='number of image channel', default=4)
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--epoch', default=5000)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='D’s learning rate')
    parser.add_argument('--lr_g', type=float, default=0.0005, help='G‘s learning rate')
    parser.add_argument('--gamma', type=float, default=0.01, help='scheduler gamma')
    parser.add_argument('--weight_rec', type=float, default=300,help='weight of rec_loss')
    parser.add_argument('--weight_back', type=float, default=30, help='weight of back_loss')
    parser.add_argument('--weight_color', type=float, default=100, help='weight of color_loss')
    parser.add_argument('--weight_gradient', type=float, default=10, help='weight of gradient_loss')
    parser.add_argument('--scale_num', type=int, default=4, help='number of scale')   # 1 2 3 5
    parser.add_argument('--nfc', type=int, default=32, help='number of filters')
    parser.add_argument('--count', type=int, default=0, help='number of scale')
    parser.add_argument('--max', type=int, default=1)
    opt = parser.parse_args()
    ms_image_ori, ms_max = functions.imreadImg('%s/%s' % (opt.input_dir, opt.input_ms), opt)
    pan_image_ori, pan_max = functions.imreadImg('%s/%s' % (opt.input_dir, opt.input_pan), opt)
    ms_image_ori_mean = ms_image_ori.mean()
    ms_image_ori_1_mean = ms_image_ori[:, 0, :, :].mean()
    ms_image_ori_2_mean = ms_image_ori[:, 1, :, :].mean()
    ms_image_ori_3_mean = ms_image_ori[:, 2, :, :].mean()
    ms_image_ori_4_mean = ms_image_ori[:, 3, :, :].mean()
    pans = []
    pans = functions.creat_reals_pyramid(pan_image_ori,pans,opt)
    in_s = ms_image_ori
    loss = torch.nn.MSELoss()
    print('start train:')
    begin_time = time()
    while opt.count < opt.scale_num:
        opt.nfc = min(opt.nfc * pow(2, math.floor(opt.count/2)), 64)
        netG = model.Generator(opt).to(opt.device)
        netG.apply(model.weights_init)
        print(netG)
        netD = model.Discriminator(opt).to(opt.device)
        netD.apply(model.weights_init)
        print(netD)
        gaussian_conv = functions.GuassianBlur().to(opt.device)
        optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)
        pan_image = pans[opt.count]
        pan_image_mean = pan_image.mean()
        ms_image = in_s
        scale = math.ceil(math.pow(2, (6 + (2 / 4) * (opt.count + 1))))
        ms_image = torch.nn.functional.interpolate(ms_image, size=(scale, scale), mode='bilinear')
        ms_image = torch.cat((pan_image, ms_image), 1)
        scale_begin_time = time()
        epoch = opt.epoch
        for i in range(epoch):
            for j in range(3):
                netD.zero_grad()
                real_out = netD(pan_image)
                errD_real = -real_out.mean()
                errD_real.backward(retain_graph=True)
                middle_image = netG(ms_image)
                fake_image = functions.weight(middle_image)
                errD_fake = netD(fake_image).mean()
                errD_fake.backward(retain_graph=True)
                gradient_penalty = functions.calc_gradient_penalty(netD, pan_image, fake_image, 0.1, opt.device)
                gradient_penalty.backward()
                optimizerD.step()
            for j in range(3):
                netG.zero_grad()
                middle_image = netG(ms_image)
                fake_image = functions.weight(middle_image)
                output = netD(fake_image)
                errG = -output.mean()
                errG.backward(retain_graph=True)
                rec_loss = opt.weight_rec * loss(fake_image, pan_image)
                rec_loss.backward(retain_graph=True)
                gaussian_conv = functions.GuassianBlur().to(opt.device)
                B_middle_image_blur = gaussian_conv(middle_image)
                B_middle_image = torch.nn.functional.interpolate(B_middle_image_blur, size=(64, 64), mode='bilinear')
                back_loss = opt.weight_back * loss(B_middle_image, ms_image_ori)
                back_loss.backward(retain_graph=True)
                middle_image_gradient_x, middle_image_gradient_y = functions.gradientLoss_MS(middle_image,opt)
                pan_image_gradient_x, pan_image_gradient_y = functions.gradientLoss_P(pan_image,opt)
                gradient_loss_x = loss(middle_image_gradient_x,pan_image_gradient_x)
                gradient_loss_y = loss(middle_image_gradient_y,pan_image_gradient_y)
                gradient_loss = opt.weight_gradient * (gradient_loss_x+gradient_loss_y)
                gradient_loss.backward(retain_graph=True)
                color_loss_1 = loss(middle_image[:, 0, :, :].mean(), ms_image_ori_1_mean)
                color_loss_2 = loss(middle_image[:, 1, :, :].mean(), ms_image_ori_2_mean)
                color_loss_3 = loss(middle_image[:, 2, :, :].mean(), ms_image_ori_3_mean)
                color_loss_4 = loss(middle_image[:, 3, :, :].mean(), ms_image_ori_4_mean)
                color_loss = opt.weight_color * (color_loss_1+color_loss_2+color_loss_3+color_loss_4)
                color_loss.backward()
                optimizerG.step()
            if i % 25 == 0 or i == (epoch - 1):
                print('%d epoch:[%d/%d]' % (opt.count, i, epoch))
                skimage.io.imsave('output/fake_sample_%s_%s.tif' % (opt.count, i),
                                  functions.convert_image_np((fake_image.detach()), pan_max, opt).astype(numpy.uint16))
                skimage.io.imsave('output/middle_image_%s_%s.tif' % (opt.count, i),
                                  functions.convert_image_np((middle_image.detach()), ms_max, opt).astype(numpy.uint16))
            schedulerD.step()
            schedulerG.step()
        torch.save(netG.state_dict(), 'output/netG_%s.pth' % (opt.count))
        torch.save(netD.state_dict(), 'output/netD_%s.pth' % (opt.count))
        scale_end_time = time()
        print('time:%d' % (scale_end_time - scale_begin_time))
        in_s = middle_image.detach()
        opt.count = opt.count+1
        del netG, netD
    end_time = time()
    run_time = end_time-begin_time
    print('run time:', run_time)
