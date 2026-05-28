import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                    m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UpsampleCat(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 2, 2, 0, bias=False)
        initialize_weights(self.deconv, 0.1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return torch.cat([x1, x2], dim=1)


def conv_func(x, conv, blindspot):
    size = conv.kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad = nn.ConstantPad2d(padding=(0, 0, ofs, 0), value=0)
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot):
    if blindspot:
        pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x


def rotate(x, angle):
    if angle == 0:
        return x
    elif angle == 90:
        return torch.rot90(x, k=1, dims=(3, 2))
    elif angle == 180:
        return torch.rot90(x, k=2, dims=(3, 2))
    elif angle == 270:
        return torch.rot90(x, k=3, dims=(3, 2))


class EncoderStage(nn.Module):
    """One encoder stage: num_convs convolutions followed by optional max-pool."""

    def __init__(self, in_channels, out_channels, num_convs, blindspot, pool):
        super(EncoderStage, self).__init__()
        self.blindspot = blindspot
        self.has_pool = pool
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            self.convs.append(nn.Conv2d(in_ch, out_channels, 3, 1, 1))
        initialize_weights(self.convs, 0.1)

        if pool:
            self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        for conv in self.convs:
            x = self.act(conv_func(x, conv, self.blindspot))
        if self.has_pool:
            x = pool_func(x, self.maxpool, self.blindspot)
            return x, x  # onward flow and skip are the same pooled tensor
        return x, None


class DecoderStage(nn.Module):
    """One decoder stage: upsample + concat skip + double conv."""

    def __init__(self, in_channels, up_channels, skip_channels, out_channels,
                 blindspot):
        super(DecoderStage, self).__init__()
        self.blindspot = blindspot
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.up = UpsampleCat(in_channels, up_channels)
        self.conv_a = nn.Conv2d(up_channels + skip_channels, out_channels, 3, 1,
                                1)
        self.conv_b = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        initialize_weights([self.conv_a, self.conv_b], 0.1)

    def forward(self, x, skip):
        x = self.up(x, skip)
        x = self.act(conv_func(x, self.conv_a, self.blindspot))
        x = self.act(conv_func(x, self.conv_b, self.blindspot))
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48,
                 blindspot=False,
                 zero_last=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.zero_last = zero_last
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Encoder: (in_ch, out_ch, num_convs, pool, save_skip)
        enc_config = [
            (in_nc, n_feature, 2, True, True),          # stage 0
            (n_feature, n_feature, 1, True, True),      # stage 1
            (n_feature, n_feature, 1, True, True),      # stage 2
            (n_feature, n_feature, 1, True, True),      # stage 3
            (n_feature, n_feature, 1, True, False),     # stage 4: pool, no skip
            (n_feature, n_feature, 1, False, False),    # bottleneck
        ]
        self.enc_stages = nn.ModuleList()
        self._enc_save_skip = []
        for in_ch, out_ch, nconv, pool, save_skip in enc_config:
            self.enc_stages.append(
                EncoderStage(in_ch, out_ch, nconv, blindspot, pool))
            self._enc_save_skip.append(save_skip)

        # Decoder: (in_ch, up_ch, skip_ch, out_ch)
        dec_config = [
            (n_feature, n_feature, n_feature, n_feature * 2),         # up5
            (n_feature * 2, n_feature * 2, n_feature, n_feature * 2), # up4
            (n_feature * 2, n_feature * 2, n_feature, n_feature * 2), # up3
            (n_feature * 2, n_feature * 2, n_feature, n_feature * 2), # up2
        ]
        self.dec_stages = nn.ModuleList()
        for in_ch, up_ch, skip_ch, out_ch in dec_config:
            self.dec_stages.append(
                DecoderStage(in_ch, up_ch, skip_ch, out_ch, blindspot))

        # Final upsample (up1) + output stage
        self.up_out = UpsampleCat(n_feature * 2, n_feature * 2)
        self.dec_conv1a = nn.Conv2d(n_feature * 2 + in_nc, 96, 3, 1, 1)
        self.dec_conv1b = nn.Conv2d(96, 96, 3, 1, 1)
        initialize_weights([self.dec_conv1a, self.dec_conv1b], 0.1)

        # Output head
        if blindspot:
            self.nin_a = nn.Conv2d(96 * 4, 96 * 4, 1, 1, 0)
            self.nin_b = nn.Conv2d(96 * 4, 96, 1, 1, 0)
        else:
            self.nin_a = nn.Conv2d(96, 96, 1, 1, 0)
            self.nin_b = nn.Conv2d(96, 96, 1, 1, 0)
        initialize_weights([self.nin_a, self.nin_b], 0.1)
        self.nin_c = nn.Conv2d(96, out_nc, 1, 1, 0)
        if not self.zero_last:
            initialize_weights(self.nin_c, 0.1)

    def forward(self, x):
        blindspot = self.blindspot
        if blindspot:
            x = torch.cat([rotate(x, a) for a in [0, 90, 180, 270]], dim=0)

        # Encoder
        pool0 = x
        skips = []
        for i, stage in enumerate(self.enc_stages):
            x, skip = stage(x)
            if self._enc_save_skip[i]:
                skips.append(skip)

        # Decoder
        for i, stage in enumerate(self.dec_stages):
            x = stage(x, skips[-i - 1])

        # Final upsample + output
        x = self.up_out(x, pool0)
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
            x = pad(x[:, :, :-1, :])
            x = torch.split(x, split_size_or_sections=x.shape[0] // 4, dim=0)
            x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
            x = torch.cat(x, dim=1)
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        return x


if __name__ == "__main__":
    import numpy as np
    x = torch.from_numpy(np.zeros((10, 3, 32, 32), dtype=np.float32))
    print(x.shape)
    net = UNet(in_nc=3, out_nc=3, blindspot=False)
    y = net(x)
    print(y.shape)
