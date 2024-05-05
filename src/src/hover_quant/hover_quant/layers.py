import torch
from torch import nn
from torch.nn import functional as F
from hover_quant.utils import center_crop_im_batch


class _BatchNormRelu(nn.Module):
    """BatchNorm + Relu layer"""

    def __init__(self, n_channels):
        super(_BatchNormRelu, self).__init__()
        self.batch_norm = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.batch_norm(inputs))


class _HoVerNetResidualUnit(nn.Module):
    """
    Residual unit.
    See: Fig. 2(a) from Graham et al. 2019 HoVer-Net paper.
    This unit is not preactivated! That's handled when assembling units into blocks.
    output_channels corresponds to m in the figure
    """

    def __init__(self, input_channels, output_channels, stride):
        super(_HoVerNetResidualUnit, self).__init__()
        internal_channels = output_channels // 4
        if stride != 1 or input_channels != output_channels:
            self.convshortcut = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=1,
                bias=False,
            )
        else:
            self.convshortcut = None
        self.conv1 = nn.Conv2d(
            input_channels, internal_channels, kernel_size=1, bias=False
        )
        self.bnrelu1 = _BatchNormRelu(internal_channels)
        self.conv2 = nn.Conv2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bnrelu2 = _BatchNormRelu(internal_channels)
        self.conv3 = nn.Conv2d(
            internal_channels, output_channels, kernel_size=1, bias=False
        )

    def forward(self, inputs):
        skip = self.convshortcut(inputs) if self.convshortcut else inputs
        out = self.conv1(inputs)
        out = self.bnrelu1(out)
        out = self.conv2(out)
        out = self.bnrelu2(out)
        out = self.conv3(out)
        out = out + skip
        return out


class _HoVerNetEncoder(nn.Module):
    """
    Encoder for HoVer-Net.
    7x7 conv, then four residual blocks, then 1x1 conv.
    BatchNormRelu after first convolution, based on code from authors, see:
     (https://github.com/vqdang/hover_net/blob/5d1560315a3de8e7d4c8122b97b1fe9b9513910b/src/model/graph.py#L67)

     Reuturn a list of the outputs from each residual block, for later skip connections
    """

    def __init__(self):
        super(_HoVerNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.bnrelu1 = _BatchNormRelu(64)
        self.block1 = _make_HoVerNet_residual_block(
            input_channels=64, output_channels=256, stride=1, n_units=3
        )
        self.block2 = _make_HoVerNet_residual_block(
            input_channels=256, output_channels=512, stride=2, n_units=4
        )
        self.block3 = _make_HoVerNet_residual_block(
            input_channels=512, output_channels=1024, stride=2, n_units=6
        )
        self.block4 = _make_HoVerNet_residual_block(
            input_channels=1024, output_channels=2048, stride=2, n_units=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=2048, out_channels=1024, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = self.bnrelu1(out1)
        out1 = self.block1(out1)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out4 = self.conv2(out4)
        return [out1, out2, out3, out4]


class _HoVerNetDenseUnit(nn.Module):
    """
    Dense unit.
    See: Fig. 2(b) from Graham et al. 2019 HoVer-Net paper.
    """

    def __init__(self, input_channels):
        super(_HoVerNetDenseUnit, self).__init__()
        self.bnrelu1 = _BatchNormRelu(input_channels)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=128, kernel_size=1
        )
        self.bnrelu2 = _BatchNormRelu(128)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=5, padding=2
        )

    def forward(self, inputs):
        out = self.bnrelu1(inputs)
        out = self.conv1(out)
        out = self.bnrelu2(out)
        out = self.conv2(out)

        # need to make sure that inputs have same shape as out, so that we can concat
        cropdims = (inputs.size(2) - out.size(2), inputs.size(3) - out.size(3))
        inputs_cropped = center_crop_im_batch(inputs, dims=cropdims)
        out = torch.cat((inputs_cropped, out), dim=1)
        return out


class _HoverNetDecoder(nn.Module):
    """
    One of the three identical decoder branches.
    """

    def __init__(self):
        super(_HoverNetDecoder, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=5,
            padding=2,
            stride=1,
            bias=False,
        )
        self.dense1 = _make_HoVerNet_dense_block(input_channels=256, n_units=8)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False
        )
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=5,
            padding=2,
            stride=1,
            bias=False,
        )
        self.dense2 = _make_HoVerNet_dense_block(input_channels=128, n_units=4)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False
        )
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=5,
            stride=1,
            bias=False,
            padding=2,
        )

    def forward(self, inputs):
        """
        Inputs should be a list of the outputs from each residual block, so that we can use skip connections
        """
        block1_out, block2_out, block3_out, block4_out = inputs
        out = self.upsample1(block4_out)
        # skip connection addition
        out = out + block3_out
        out = self.conv1(out)
        out = self.dense1(out)
        out = self.conv2(out)
        out = self.upsample2(out)
        # skip connection
        out = out + block2_out
        out = self.conv3(out)
        out = self.dense2(out)
        out = self.conv4(out)
        out = self.upsample3(out)
        # last skip connection
        out = out + block1_out
        out = self.conv5(out)
        return out


def _make_HoVerNet_residual_block(input_channels, output_channels, stride, n_units):
    """
    Stack multiple residual units into a block.
    output_channels is given as m in Fig. 2 from Graham et al. 2019 paper
    """
    units = []
    # first unit in block is different
    units.append(_HoVerNetResidualUnit(input_channels, output_channels, stride))

    for i in range(n_units - 1):
        units.append(_HoVerNetResidualUnit(output_channels, output_channels, stride=1))
        # add a final activation ('preact' for the next unit)
        # This is different from how authors implemented - they added BNRelu before all units except the first, plus
        # a final one at the end.
        # I think this is equivalent to just adding a BNRelu after each unit
        units.append(_BatchNormRelu(output_channels))

    return nn.Sequential(*units)


def _make_HoVerNet_dense_block(input_channels, n_units):
    """
    Stack multiple dense units into a block.
    """
    units = []
    in_dim = input_channels
    for i in range(n_units):
        units.append(_HoVerNetDenseUnit(in_dim))
        in_dim += 32
    units.append(_BatchNormRelu(in_dim))
    return nn.Sequential(*units)
