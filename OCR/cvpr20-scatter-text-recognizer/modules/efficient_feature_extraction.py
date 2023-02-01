import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, mobilenet_v2
class EfficientNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(EfficientNet_FeatureExtractor, self).__init__()
        # self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])
        self.ConvNet = efficientnet_b0().features
        self.ConvNet[0][0] = torch.nn.Conv2d(input_channel, 32, kernel_size=(3,3),stride=(2,2), padding=(1,1), bias=False)
        self.ConvNet[-1][-3] = torch.nn.Conv2d(self.ConvNet[-1][-3].in_channels, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.ConvNet[-1][-2] = torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.TextAttentionModule = TextAttentionModule(output_channel, output_channel)
        
    def forward(self, input):
        output = self.ConvNet(input)
        return self.TextAttentionModule(output)


class TextAttentionModule(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TextAttentionModule, self).__init__()
        self.conv31 = nn.Conv2d(input_channel, output_channel, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, features):
        x = self.conv31(features)
        x = self.sigmoid(x)
        x = x*features
        return x

if __name__ == '__main__':
    # ConvNet = EfficientNet_FeatureExtractor(1, 512)
    ConvNet = mobilenet_v2()
    # ConvNet[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3,3),stride=(2,2), padding=(1,1), bias=False)
    for name, param in ConvNet.named_children():
        print(name)
        print(param)
    
    x = torch.rand((256, 3, 50, 60))
    # output = ConvNet(x)
    # print(output.size())
    pass
    