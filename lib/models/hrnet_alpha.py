"""
MIT License
Copyright (c) 2019 Microsoft
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os

import torch
import torch.nn as nn

import torch.nn.functional as F

#zychen
from itertools import chain

BatchNorm2d = nn.BatchNorm2d

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, multi_scale_output=True, alphas=None, multiply_prob=None, alpha_thr=0.5, name=''):
        # zychen: a new parameter alphas added
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.relu = nn.ReLU(False)
        
        # zychen
        self.num_blocks = num_blocks
        self.multiply_prob = multiply_prob
        self.alpha_thr = alpha_thr
        if isinstance(alphas, bool):
            # if type bool, then it means if we use alphas for fuse layer
            self.alphas = torch.zeros(num_blocks[0], num_branches, num_branches)
            self.alphas = nn.Parameter(self.alphas, requires_grad = False)
        else:
            self.alphas = nn.Parameter(alphas, requires_grad = False)
        self.interactions = self._make_interact_layers(num_branches, num_blocks, num_channels)
        
        # zychen make process fusion within different depth of the same stage
        self.within_stage = False#True
        if self.within_stage:
            self.conv1x1s = []
            for i in range(num_branches):
                self.conv1x1s.append(nn.Sequential(
                                        nn.Conv2d(
                                            num_channels[i] * num_blocks[i],
                                            num_channels[i],
                                            1, 1, 0, bias=False
                                        ),
                                        nn.BatchNorm2d(num_channels[i]),
                                        nn.ReLU(True)))
            self.conv1x1s = nn.ModuleList(self.conv1x1s)
        else:
            self.conv1x1s = None
            
        self.debug_dir = 'analysis/'
        self.all_batch = None
                

    # zychen: get architectural parameters
    def get_alphas(self):
        return self.alphas
    
    # zychen: get interaction parameters
    def get_interact_weights(self):
        return self.interactions.parameters()

    # zychen: make interaction layers
    def _make_interact_layers(self, num_branches, num_blocks, num_inchannels):
        all_blocks = []
        count = 0
        probs = torch.sigmoid(self.alphas)
        for b in range(num_blocks[0]):
            fuse_layers = []
            for i in range(num_branches):
                fuse_layer = []
                for j in range(num_branches):
                    if j > i and probs[b][i][j] > self.alpha_thr:
                        fuse_layer.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_inchannels[i],
                                    1, 1, 0, bias=False
                                ),
                                nn.BatchNorm2d(num_inchannels[i]),
                                nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                            )
                        )
                        count += 1
                    elif j < i and probs[b][i][j] > self.alpha_thr:
                        conv3x3s = []
                        for k in range(i-j):
                            if k == i - j - 1:
                                num_outchannels_conv3x3 = num_inchannels[i]
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3, 2, 1, bias=False
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3)
                                    )
                                )
                            else:
                                num_outchannels_conv3x3 = num_inchannels[j]
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(
                                            num_inchannels[j],
                                            num_outchannels_conv3x3,
                                            3, 2, 1, bias=False
                                        ),
                                        nn.BatchNorm2d(num_outchannels_conv3x3),
                                        nn.ReLU(True)
                                    )
                                )
                        count += 1
                        fuse_layer.append(nn.Sequential(*conv3x3s))
                    else:
                        fuse_layer.append(None)
                fuse_layers.append(nn.ModuleList(fuse_layer))
            all_blocks.append(nn.ModuleList(fuse_layers))
        return nn.ModuleList(all_blocks)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        if not self.multiply_prob:
            pass
        else:
            probs = torch.sigmoid(self.alphas)
            
        if self.debug_dir and self.all_batch is None:
            self.all_batch = {}
            for iblock in range(self.num_blocks[0]):
                for ibranch in range(self.num_branches):
                    self.all_batch["%d_%d"%(iblock, ibranch)] = []
                    for inputs in range(self.num_branches):
                        self.all_batch["%d_%d_%d"%(iblock, ibranch, inputs)] = []

        ys = [] #
        for iblock in range(self.num_blocks[0]):
            #original forward
            for i in range(self.num_branches):
                x[i] = self.branches[i][iblock](x[i])

            y = []
            for ibranch in range(self.num_branches):
                y.append(x[ibranch])
                if self.debug_dir:
                    self.all_batch["%d_%d"%(iblock, ibranch)].append(y[ibranch].abs().mean().item())
                for inputs in range(self.num_branches):
                    # Input from the same resolution shouldn't be added
                    if inputs == ibranch:
                        continue
                    # Add feature from different resolution with weight alpha
                    if self.interactions[iblock][ibranch][inputs] is not None:
                        if self.multiply_prob:
                            y[ibranch] = y[ibranch] + probs[iblock][ibranch][inputs] * self.interactions[iblock][ibranch][inputs](x[inputs])
                        else:
                            to_add = self.interactions[iblock][ibranch][inputs](x[inputs]) * 0.5
                            y[ibranch] = y[ibranch] + to_add
                            if self.debug_dir:
                                self.all_batch["%d_%d_%d"%(iblock, ibranch, inputs)].append(to_add.abs().mean().item())
            x = y
            ys.append(y)
        if self.conv1x1s is None:
            return x
        # adding code to process fusion inside one stage
        if self.debug_dir and len(self.all_batch["0_0"]) == 200:
            torch.save(self.all_batch, os.path.join(self.debug_dir, "feature_%s.pth"%(self.name)))
        outs = []
        for ibranch in range(len(ys[0])):
            out = torch.cat([ys[i][ibranch] for i in range(len(ys))], dim = 1)
            outs.append(self.conv1x1s[ibranch](out))
        return outs


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 extra,
                 search_config,
                 norm_eval=True,
                 zero_init_residual=False,
                 frozen_stages=-1,
                 alpha_file=None,
                 alpha_thr=0.5,
                 multiply_prob=""):
        super(HighResolutionNet, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        # for
        self.extra = extra
        self.search_config = search_config
        print(self.search_config)

        # zychen: add alpha list
        self.alphas = []
        self.share_alphas = search_config['SHARE_ALPHAS']
        self.use_fuse_alpha = search_config['USE_FUSE_ALPHA']
        self.multiply_prob = multiply_prob
        self.alpha_thr = alpha_thr
        print('Shared alphas: {}; Transition alpha: {}'.format(self.share_alphas, self.use_fuse_alpha))
        if alpha_file is not None:
            self.alpha_dict = torch.load(alpha_file)
        else:
            self.alpha_dict = None

        # stem network
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block_type = self.stage1_cfg['BLOCK']
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]

        block = blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        self.num_channels = num_channels[0]
        block_type = self.stage2_cfg['BLOCK']

        block = blocks_dict[block_type]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)
        # num_modules, num_branches, num_blocks, num_channels, block, fuse_method, num_inchannels
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block_type = self.stage3_cfg['BLOCK']

        block = blocks_dict[block_type]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block_type = self.stage4_cfg['BLOCK']

        block = blocks_dict[block_type]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=self.extra['MULTI_SCALE_OUTPUT'])

        # Classification Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)
        self.classifier = nn.Linear(2048, 1000)
        
        # zychen stage alphas
        if 'stage_alphas' in self.alpha_dict:
            self.stage_alphas = nn.Parameter(self.alpha_dict['stage_alphas'], requires_grad = False)
        elif 'module.stage_alphas' in self.alpha_dict:
            self.stage_alphas = nn.Parameter(self.alpha_dict['module.stage_alphas'], requires_grad = False)
        else:
            self.stage_alphas = None
        self._make_stage_alphas()

    #zychen: get architecture params
    def get_alphas(self):
        if len(self.alphas) > 0:
            return self.alphas
        all_alphas = []
        for stage in [self.stage2, self.stage3, self.stage4]:
            all_alphas.extend([block.get_alphas() for block in stage])
        if self.stage_alphas is not None:
            all_alphas.append(self.stage_alphas)
        return all_alphas

    def calc_alpha_loss(self, device = None):
        alphas = self.get_alphas()
        total_loss = torch.zeros([]).to(alphas[0].device)
        all_counts = 0
        for alpha in alphas:
            probs = torch.sigmoid(alpha)
            ent_loss = -(probs * torch.log(probs) + (1-probs) * torch.log(1-probs))
            ent_loss = torch.where(probs > 0.005, ent_loss, torch.zeros_like(ent_loss))
            ent_loss = torch.where(probs < 0.995, ent_loss, torch.zeros_like(ent_loss))
            total_loss += torch.sum(ent_loss)
            all_counts += probs.shape[0] * probs.shape[1] * probs.shape[2]
        total_loss /= all_counts
        return total_loss
        
    #zychen: get interaction params
    def get_interact_weights(self):
        all_interact_weights = []
        for stage in [self.stage2, self.stage3, self.stage4]:
            all_interact_weights.extend([block.get_interact_weights() for block in stage])
        return chain(*all_interact_weights)

    #zychen: get model params (no alphas)
    def get_model_weights(self, lr=None):
        all_weights = []
        for k, v in self.named_parameters():
            if 'alphas' in k:
                continue
            if lr is None:
                all_weights.append(v)
            else:
                tmp_lr = lr if 'interactions' in k else lr * 0.2
                all_weights.append({"params": [v], "lr": tmp_lr})
        return all_weights
    
    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer
    
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _frozen_stages(self):
        # frozen stage  1 or stem networks
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1, self.conv2, self.bn2]:
                for param in m.parameters():
                    param.requires_grad = False
        if self.frozen_stages == 1:
            for param in self.layer1.parameters():
                param.requires_grad = False

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]

        # zychen: add config if one can share connections within the same stage.
        if self.share_alphas:
            alp = torch.zeros(num_blocks[0] + 1 if self.use_fuse_alpha else num_blocks[0], num_branches, num_branches)
            #alp[:-1, :, :] = 0.2
            #alp[-1, :, :] = 0.8
            self.alphas.append(nn.Parameter(alp), requires_grad = False)

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            # load alpha
            alpha_key = "stage%d.%d.alphas"%(num_branches, i)
            if self.alpha_dict is not None:
                this_alpha = self.alpha_dict[alpha_key] if alpha_key in self.alpha_dict else self.alpha_dict["module."+alpha_key]
            else:
                this_alpha = None
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     reset_multi_scale_output,
                                     alphas=this_alpha,
                                     multiply_prob=self.multiply_prob,
                                     alpha_thr=self.alpha_thr,
                                     name="stage%dblock%d"%(num_branches, i))
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_stage_alphas(self, num_channels = None):
        if self.stage_alphas is None:
            self.stage_links = [None] * 35
            return 

        stage_alphas = torch.sigmoid(self.stage_alphas)
        stage_links = []
        num_inchannels = [18, 36, 72, 144]

        def make_one_link(j, i):# j is input, i is output
            if stage_alphas[len(stage_links)] < self.alpha_thr:
                return None
            elif j > i:
                return nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[j],
                            num_inchannels[i],
                            1, 1, 0, bias=False
                        ),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    )
            elif j == i:
                return None
            else:
                conv3x3s = []
                for k in range(i-j):
                    if k == i - j - 1:
                        num_outchannels_conv3x3 = num_inchannels[i]
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    3, 2, 1, bias=False
                                ),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            )
                        )
                    else:
                        num_outchannels_conv3x3 = num_inchannels[j]
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    3, 2, 1, bias=False
                                ),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)
                            )
                        )
                return nn.Sequential(*conv3x3s)

        for i in range(2):
            for j in range(1):
                stage_links.append(make_one_link(j, i))
        for i in range(3):
            for j in range(1):
                stage_links.append(make_one_link(j, i))
            for j in range(2):
                stage_links.append(make_one_link(j, i))
        for i in range(4):
            for j in range(1):
                stage_links.append(make_one_link(j, i))
            for j in range(2):
                stage_links.append(make_one_link(j, i))
            for j in range(3):
                stage_links.append(make_one_link(j, i))

        self.stage_links = nn.ModuleList(stage_links)
        self.__setattr__('stage_links', self.stage_links)

    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        if isinstance(pretrained, str) and pretrained:
            checkpoint = torch.load(pretrained)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)
        #    load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None or pretrained == '':
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        # init for stage_alphas
        offset = 0
        stage_alphas = torch.sigmoid(self.stage_alphas) if self.stage_alphas is not None else None

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        stage1_list = x_list[:1]

        y_list = self.stage2(x_list)
        if stage_alphas is not None:
            stage2_list = []
            for i in range(2):
                stage2_list.append(y_list[i])
                for j in range(1):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage2_list[-1] = stage2_list[-1] + self.stage_links[offset + j](stage1_list[j]) * self.stage_alphas[offset + j]
                offset += 1
            y_list = stage2_list

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        # merge between stages
        if stage_alphas is not None:
            stage3_list = []
            for i in range(3):
                stage3_list.append(y_list[i])
                for j in range(1):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage3_list[-1] = stage3_list[-1] + self.stage_links[offset + j](stage1_list[j]) * self.stage_alphas[offset + j]
                offset += 1
                for j in range(2):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage3_list[-1] = stage3_list[-1] + self.stage_links[offset + j](stage2_list[j]) * self.stage_alphas[offset + j]
                offset += 2
            y_list = stage3_list

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        # merge betwen stages
        if stage_alphas is not None:
            stage4_list = []
            for i in range(4):
                stage4_list.append(y_list[i])
                for j in range(1):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage4_list[-1] = stage4_list[-1] + self.stage_links[offset + j](stage1_list[j]) * self.stage_alphas[offset + j]
                offset += 1
                for j in range(2):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage4_list[-1] = stage4_list[-1] + self.stage_links[offset + j](stage2_list[j]) * self.stage_alphas[offset + j]
                offset += 2
                for j in range(3):
                    if i == j or self.stage_links[offset + j] is None:
                        continue
                    stage4_list[-1] = stage4_list[-1] + self.stage_links[offset + j](stage3_list[j]) * self.stage_alphas[offset + j]
                offset += 3
            y_list = stage4_list

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                 [2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y

    def train(self, mode=True):
        super(HighResolutionNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        #if is_module_wrapper(module):
        #    module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    #rank, _ = get_dist_info()
    if len(err_msg) > 0:# and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def get_cls_net(num_layers, **kwargs):
  from .hrnet_cfg import HRNET_CFGS, _C as cfg
  config_file = os.path.dirname(os.path.realpath(__file__)) + '/w18search_v2.yaml'
  cfg.defrost()
  cfg.merge_from_file(config_file)
  cfg.freeze()
  kwargs["alpha_file"]="/userhome/ctnet_model/v2.pth"#"/home/yszhu3/github/ctnet_new/exp/ctdet+entropy/coco_hrnetlink_0829entropy10/alpha_last.pth"#
  kwargs["alpha_thr"]=0.0
  model = HighResolutionNet(cfg['MODEL']['EXTRA'], cfg['MODEL']['SEARCH'], **kwargs)
  
  if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Config: ', cfg['MODEL']['EXTRA'])
    print('Loading Pretrained from :', cfg['MODEL']['PRETRAINED'])
    print('Additional Config: ', kwargs)
  if cfg['MODEL']['INIT_WEIGHTS']:
    model.init_weights(cfg['MODEL']['PRETRAINED'])

  return model


if __name__ == "__main__":
    net = get_hrnet(None, None, None)