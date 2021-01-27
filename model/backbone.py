import torch
import torch.nn as nn
import torch.nn.functional as F

from model.efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):
    """ Backbone Wrapper """
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(model_name)
        del model._bn1, model._conv_head, model._fc
        del model._avg_pooling, model._dropout
        self.model = model
        self.drop_connect_rate = model._global_params.drop_connect_rate

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))

        execute_next_block_check = False
        features = []
        for idx, block in enumerate(self.model._blocks):
            if execute_next_block_check:
                execute_next_block_check = False
                if block._depthwise_conv.stride == [2, 2]:
                    features.append(x)
            
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == len(self.model._blocks) - 1:
                features.append(x)
                execute_next_block_check = False
            else:
                execute_next_block_check = True

        return features[2:]

    def get_channels_list(self):
        channels = []
        for idx, block in enumerate(self.model._blocks):
            if idx == len(self.model._blocks) - 1:
                channels.append(block._block_args.output_filters)

            else:
                next_block = self.model._blocks[idx + 1]
                if next_block._block_args.stride == [2]:
                    channels.append(block._block_args.output_filters)

        return channels[2:]


if __name__ == '__main__':
    ''' quick test '''

    backbone = EfficientNet('')
