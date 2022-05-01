from tokenize import Double
import torch
import torch.nn as nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()

        """ 
        Original implementation has VALID padding which is more inconvenient to implement
        but could perform better 
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),  
            nn.BatchNorm2d(out_channels), 
            nn.ELU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.double_conv_layer = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        """ 
        forward returns image before and after applying pooling,
        this will help in the "crop and copy" part of UNet
        """
        x = self.double_conv_layer(inputs)
        pooling = self.pooling_layer(x)

        return x, pooling

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.climb_up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConvolution(in_channels=out_channels*2, out_channels=out_channels)

    def forward(self, inputs, skip_connection):
        x = self.climb_up(inputs)
        x = torch.cat((x, skip_connection), axis=1)
        x = self.double_conv(x)
        return x 


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        """ Stepping down """
        self.climb_down1 = EncoderBlock(in_channels=3, out_channels=64)
        self.climb_down2 = EncoderBlock(in_channels=64, out_channels=128)
        self.climb_down3 = EncoderBlock(in_channels=128, out_channels=256)
        self.climb_down4 = EncoderBlock(in_channels=256, out_channels=512)

        """ Bottleneck """
        self.bottleneck = DoubleConvolution(in_channels=512, out_channels=1024)
        
        """ Stepping up"""
        self.climb_up4 = DecoderBlock(in_channels=1024, out_channels=512)
        self.climb_up3 = DecoderBlock(in_channels=512, out_channels=256)
        self.climb_up2 = DecoderBlock(in_channels=256, out_channels=128)
        self.climb_up1 = DecoderBlock(in_channels=128, out_channels=64)

        """ Final convolution """
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def forward(self, inputs):
        """ Descending """
        skip_con1, pooling1 = self.climb_down1(inputs)
        skip_con2, pooling2 = self.climb_down2(pooling1)
        skip_con3, pooling3 = self.climb_down3(pooling2)
        skip_con4, pooling4 = self.climb_down4(pooling3)

        """ Bottleneck layer """
        bottleneck = self.bottleneck(pooling4)

        """ Ascending """
        transposed_conv4 = self.climb_up4(bottleneck, skip_con4)
        transposed_conv3 = self.climb_up3(transposed_conv4, skip_con3)
        transposed_conv2 = self.climb_up2(transposed_conv3, skip_con2)
        transposed_conv1 = self.climb_up1(transposed_conv2, skip_con1)

        """ Output """
        output = self.final_layer(transposed_conv1)

        return output


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    conv = UNet()
    b_neck = conv(x)
    print(b_neck.shape)
