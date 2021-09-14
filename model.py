from torch import nn
import torch

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('pad',nn.ReflectionPad2d(1)),
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=0)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        self.conv1 = ConvBlock((opt.channels+1), opt.nfc, 3, 1,1)
        self.conv2 = ConvBlock((opt.channels+1) + opt.nfc, opt.nfc, 3, 1,1)
        self.conv3 = ConvBlock((opt.channels+1) + opt.nfc*2, opt.nfc, 3, 1, 1)
        self.conv4 = ConvBlock((opt.channels+1) + opt.nfc*3, opt.nfc, 3, 1, 1)
        self.conv5 = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d((opt.channels+1)+opt.nfc*4, opt.channels, kernel_size=3, stride=1, padding=0),
           # nn.Tanh()
        )

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(torch.cat((x, h1), 1))
        h3 = self.conv3(torch.cat((x, h1, h2), 1))
        h4 = self.conv4(torch.cat((x, h1, h2, h3), 1))
        h5 = self.conv5(torch.cat((x, h1, h2, h3, h4), 1))
        return h5

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(1, N, 3, 1, 1)
        self.body = nn.Sequential()
        for i in range(3):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.nfc), max(N, opt.nfc), 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.pad=nn.ReflectionPad2d(1)
        self.tail = nn.Conv2d(max(N, opt.nfc), 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x=self.pad(x)
        x = self.tail(x)
        return x