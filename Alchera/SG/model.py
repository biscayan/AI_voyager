import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.relu(self.batch_norm2(self.conv2(self.relu(self.batch_norm1(self.conv1(x))))) + residual)
        return x

class HourGlassEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourGlassEncoding, self).__init__()
        self.conv_list = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        output = self.pooling(self.relu(self.batch_norm1(self.conv_list(x))))
        residual = output
        return residual, output

class HourGlassDecoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourGlassDecoding, self).__init__()
        self.conv_list = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.pooling = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, residual):
        output = self.relu(self.batch_norm1(self.conv_list(self.relu(x+residual))))
        output = self.pooling(output)
        return output


class HourGlassModule(nn.Module):
    def __init__(self, in_channels, channel_num_list):
        super(HourGlassModule, self).__init__()

        self.encoding_blocks = nn.ModuleList()
        for out_channels in channel_num_list[1:]:
            self.encoding_blocks.append(HourGlassEncoding(in_channels, out_channels))
            in_channels = out_channels

        self.middle1 = ResidualBlock(in_channels=in_channels)

        self.decoding_blocks = nn.ModuleList()

        channel_num_list = channel_num_list[::-1][1:]
        for out_channels in channel_num_list:
            self.decoding_blocks.append(HourGlassDecoding(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, x):
        residuals = []
        for encoding_block in self.encoding_blocks:
            residual, x = encoding_block(x)
            residuals.append(residual)

        x = self.middle1(x)

        for decoding_block in self.decoding_blocks:
            x = decoding_block(x, residuals.pop())

        return x


class HourGlassArchitecture(nn.Module):
    def __init__(self, channel_num_list, label_num):
        super(HourGlassArchitecture, self).__init__()
        self.hour_glass = HourGlassModule(channel_num_list[0], channel_num_list)

        self.conv1 = nn.Conv2d(in_channels=channel_num_list[0], out_channels=channel_num_list[0], kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channel_num_list[0])
        self.conv_predict = nn.Conv2d(in_channels=channel_num_list[0], out_channels=label_num, kernel_size=3, padding=1)
        self.conv_reverse = nn.Conv2d(in_channels=label_num, out_channels=channel_num_list[0], kernel_size=3, padding=1, bias=False)
        self.batch_norm_reverse = nn.BatchNorm2d(channel_num_list[0])

        self.conv2 = nn.Conv2d(in_channels=channel_num_list[0], out_channels=channel_num_list[0], kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channel_num_list[0])

        self.relu = nn.GELU()
        
    def forward(self, x):
        y = self.hour_glass(x)
        y = self.relu(self.batch_norm1(self.conv1(y)))

        logits = self.conv_predict(y)
        reverse = self.relu(self.batch_norm_reverse(self.conv_reverse(logits)))

        z = self.relu(self.batch_norm2(self.conv2(y)) + reverse)

        return logits, z

class StackedHourGlass(nn.Module):
    def __init__(self, stack_num, channel_num_list, label_num):
        super(StackedHourGlass, self).__init__()
        self.hour_glass_blocks = nn.ModuleList()
        for _ in range(stack_num):
            self.hour_glass_blocks.append(HourGlassArchitecture(channel_num_list, label_num))

    def forward(self, x):
        logits_list = []
        residual = 0
        for hour_glass in self.hour_glass_blocks:
            logits, x = hour_glass(x + residual)
            residual = x
            logits_list.append(logits)

        return logits_list


if __name__ == '__main__':
    #model = HourGlassArchitecture(in_channels=3, channel_num_list=[6, 12], conv_block_num_list=[2, 2], label_num=5)
    model = StackedHourGlass(stack_num=10, channel_num_list=[3, 32, 64, 128, 256], label_num=5)
    predict = model(torch.randn((5, 3, 64, 64)))
    print(predict[-1].shape)
    print(predict[0].shape, predict[1].shape)
    print(predict[0].softmax(dim=1).shape)
    print(predict[1].softmax(dim=1).shape)
    # torch.save(model.state_dict(), 'state_dict.pt')
    # torch.save(model, 'model.pt')