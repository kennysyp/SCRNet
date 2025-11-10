import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
from einops import rearrange


class VecInt(nn.Module):
    def __init__(self, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, window_size=[2, 2, 2]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm_xin = nn.LayerNorm(embed_dim)
        self.norm_yin = nn.LayerNorm(embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        d, h, w = x_in.size(2), x_in.size(3), x_in.size(4)
        x_in = x_in.permute(0, 2, 3, 4, 1)
        y_in = y_in.permute(0, 2, 3, 4, 1)
        x = self.norm_xin(x_in)
        y = self.norm_yin(y_in)
        window_size = get_winsize((d, h, w), self.window_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)

        b_, n_, c_ = x_windows.shape
        kv = self.kv_proj(x_windows).reshape(b_, n_, 2, self.num_heads, c_ // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_proj(y_windows).reshape(b_, n_, self.num_heads, c_ // self.num_heads).permute(0, 2, 1, 3) * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = attn @ v

        attn_windows = attn.transpose(1, 2).reshape(b_, n_, c_)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        cross = window_reverse(attn_windows, window_size, dims)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            cross = cross[:, :d, :h, :w, :].contiguous()
        cross = self.norm_out(cross).permute(0, 4, 1, 2, 3).contiguous()
        return cross


class ConvGRU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_z = nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_r = nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False)
        self.conv_q = nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.conv_out = nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=False)

    def forward(self, h, x):
        if h == None:
            b, c, d, h, w = x.shape
            h = torch.zeros((b, c, d, h, w), dtype=x.dtype).to(x.device)
        hx = concat(h, x)

        z = self.conv_z(hx)
        z = self.sigmoid(z)
        r = self.conv_r(hx)
        r = self.sigmoid(r)
        q = self.conv_q(concat(r*h, x))
        q = self.tanh(q)

        new_h = (1-z) * h + z * q
        out = self.conv_out(new_h)
        return new_h, out


class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class RegHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(nn.init.normal_(self.reg_head.weight, mean=0, std=1e-5))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out


class WindowCorr(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.win_size = 2 * radius + 1
        self.radius = radius
        self.padding = nn.ConstantPad3d(radius, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, f"inputs shape mismatch {x.shape} and {y.shape} in window-based correlation."
        b, c, d, h, w = x.shape
        y_padded = self.padding(y)
        offset = torch.meshgrid(
            [torch.arange(0, self.win_size) for _ in range(3)]
        )
        corr = torch.cat(
            [
                torch.sum(x * y_padded[:, :, dz:dz + d, dy:dy + h, dx:dx + w], dim=1, keepdim=True)
                for dz, dy, dx in zip(offset[0].flatten(), offset[1].flatten(), offset[2].flatten())
            ], dim=1
        )
        corr *= (c ** -0.5)
        return corr


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv(out))
        return out


class ContextModule(nn.Module):
    def __init__(self, in_channel, mode):
        super().__init__()
        self.mode = mode
        if self.mode == 'deep':
            self.conv = nn.Conv3d(in_channel + in_channel // 2, in_channel, 3, 1, 1)
        elif self.mode == 'mid':
            self.conv = nn.Conv3d(in_channel * 2 + in_channel + in_channel // 2, in_channel, 3, 1, 1)
        elif self.mode == 'shallow':
            self.conv = nn.Conv3d(in_channel * 2 + in_channel, in_channel, 3, 1, 1)
        self.sa = SpatialAttentionModule()
        self.convall = DualConvBlock(3 + in_channel * 2, in_channel)

    def forward(self, flow, ca_arr, dec):
        if self.mode == 'deep':
            ca, post = ca_arr
            post_ = nnf.interpolate(post, scale_factor=0.5, mode="trilinear", align_corners=True)
            ca_all = self.conv(concat(ca, post_))
        elif self.mode == 'mid':
            pre, ca, post = ca_arr
            pre_ = nnf.interpolate(pre, scale_factor=2, mode="trilinear", align_corners=True)
            post_ = nnf.interpolate(post, scale_factor=0.5, mode="trilinear", align_corners=True)
            ca_all = self.conv(concat(pre_, ca, post_))
        elif self.mode == 'shallow':
            pre, ca = ca_arr
            pre_ = nnf.interpolate(pre, scale_factor=2, mode="trilinear", align_corners=True)
            ca_all = self.conv(concat(pre_, ca))
        attn = self.sa(ca_all)
        out = self.convall(concat(flow, dec, attn * ca_all))
        return out


class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


class GroupConv(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        if num_heads==1:
            pass
        else:
            self.g = dim // num_heads
            self.conv_all = nn.ModuleList([])
            for i in range(num_heads):
                p = i
                k = 2*p+1
                self.conv_all.append(nn.Conv3d(self.g, self.g, kernel_size=k, padding=p, stride=1))

            self.proj_out = nn.Sequential(
                nn.Conv3d(dim,dim,3,1,1),
                nn.InstanceNorm3d(dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.num_heads==1:
            return x
        else:
            b, c, d, h, w = x.shape
            input = x.permute(0, 2, 3, 4, 1).reshape(b, d, h, w, self.num_heads, self.g).permute(4, 0, 5, 1, 2, 3)
            arr = []
            for i in range(self.num_heads):
                s_i = input[i]
                s_i = self.conv_all[i](s_i)
                s_i = s_i.reshape(b, self.g, 1, d, h, w)
                arr.append(s_i)
            out = torch.cat(arr, dim=2).reshape(b, c, d, h, w)
            out = self.proj_out(out)
            return out+x


import faiss
class KM(nn.Module):
    def __init__(self, channel, num_k=7, m_iter=1000):
        super().__init__()
        self.num_k = num_k
        self.rng_seed = 3407
        self.m_iter = m_iter
        self.kmeans = faiss.Kmeans(
            d=channel,
            k=num_k,
            niter=m_iter,
            verbose=False,
            gpu=False,
            seed=3407,
        )

    def forward(self, x):
        x_data = x.contiguous().detach().cpu().float().numpy()

        self.kmeans.train(x_data)
        centroids_np = self.kmeans.centroids

        _, labels_np = self.kmeans.index.search(x_data, 1)

        labels = torch.from_numpy(labels_np).squeeze(1).long().to(x.device)
        centroids = torch.from_numpy(centroids_np).float().to(x.device)
        return labels, centroids


class ClusteringSpatialAttention(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kmeans,
                 padding=1,
                 group_num=1):
        super().__init__()
        self.padding = (padding, padding, padding, padding, padding, padding)
        self.num_k = kmeans['num_k']
        self.out_ch = out_ch
        self.w = nn.Parameter(torch.Tensor(out_ch, out_ch))
        self.km = KM(channel=out_ch, num_k=kmeans['num_k'], m_iter=kmeans['m_iter'])
        self.get_kernel = nn.Sequential(
            nn.Linear(out_ch, int(4 * out_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(int(4 * out_ch), out_ch),
            nn.Sigmoid()
        )
        self.get_bias = nn.Sequential(
            nn.Linear(out_ch, int(4 * out_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(int(4 * out_ch), out_ch),
        )
        self.proj_in = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.group = GroupConv(out_ch, group_num)
        self.proj_out = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = self.proj_in(x)
        shortcut = x

        x = self.group(x)
        x_feature = rearrange(x, 'b c d h w -> b (d h w) c')

        all_cluster_indices = []
        all_centroids = []
        for i in range(b):
            current_cluster_idx, current_centroid = self.km(x_feature[i])
            all_cluster_indices.append(current_cluster_idx)
            all_centroids.append(current_centroid)
        labels_batch = torch.stack(all_cluster_indices, dim=0)
        centroids_batch = torch.stack(all_centroids, dim=0)

        x = rearrange(x, 'b c d h w -> b c (d h w)')

        weight_all = self.get_kernel(centroids_batch)
        weight_all = weight_all.unsqueeze(2)
        weight_all = weight_all * self.w.unsqueeze(0).unsqueeze(0)
        bias_all = self.get_bias(centroids_batch)
        bias_all = bias_all.unsqueeze(-1)

        mask_matrix = nnf.one_hot(labels_batch, num_classes=self.num_k).float().permute(0, 2, 1)
        mask_matrix = mask_matrix.unsqueeze(2)

        contributions = torch.einsum('bkoi,biq->bkoq', weight_all, x)
        contributions = contributions + bias_all
        contributions = contributions * mask_matrix
        out = contributions.sum(dim=1)

        out = rearrange(out, 'b o (d h w) -> b o d h w', d=d, h=h, w=w)
        out = out + shortcut
        out = self.proj_out(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, channel_num, kmeans):
        super().__init__()
        self.csa_1 = ClusteringSpatialAttention(in_channels, channel_num, kmeans=kmeans, group_num=1)
        self.csa_2 = ClusteringSpatialAttention(channel_num, channel_num * 2, kmeans=kmeans, group_num=2)
        self.csa_3 = ClusteringSpatialAttention(channel_num * 2, channel_num * 4, kmeans=kmeans, group_num=2)
        self.csa_4 = ClusteringSpatialAttention(channel_num * 4, channel_num * 8, kmeans=kmeans, group_num=4)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        x_1 = self.csa_1(x)
        x = self.downsample(x_1)

        x_2 = self.csa_2(x)
        x = self.downsample(x_2)

        x_3 = self.csa_3(x)
        x = self.downsample(x_3)

        x_4 = self.csa_4(x)

        return [x_1, x_2, x_3, x_4]


def concat(*x):
    return torch.cat(x, dim=1)


class Model(nn.Module):
    def __init__(self, channel_num, adaptive, iter):
        super().__init__()
        self.encoder = Encoder(in_channels=1, channel_num=channel_num, kmeans=adaptive)

        self.conv_dec_1 = DualConvBlock(channel_num * 1 * 2 + 27, channel_num * 1)
        self.conv_dec_2 = DualConvBlock(channel_num * 2 * 2 + 27, channel_num * 2)
        self.conv_dec_3 = DualConvBlock(channel_num * 4 * 2 + 27, channel_num * 4)
        self.conv_dec_4 = DualConvBlock(channel_num * 8 * 2 + 27, channel_num * 8)

        self.conv_context_1 = ContextModule(channel_num * 1, mode='shallow')
        self.conv_context_2 = ContextModule(channel_num * 2, mode='mid')
        self.conv_context_3 = ContextModule(channel_num * 4, mode='mid')
        self.conv_context_4 = ContextModule(channel_num * 8, mode='deep')

        self.block_corr_1 = WindowCorr()
        self.block_corr_2 = WindowCorr()
        self.block_corr_3 = WindowCorr()
        self.block_corr_4 = WindowCorr()

        self.upsample_1 = DeconvBlock(channel_num * 2, channel_num * 1)
        self.upsample_2 = DeconvBlock(channel_num * 4, channel_num * 2)
        self.upsample_3 = DeconvBlock(channel_num * 8, channel_num * 4)

        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 2)
        self.reghead_3 = RegHead(channel_num * 4)
        self.reghead_4 = RegHead(channel_num * 8)

        self.skip_reghead_1 = RegHead(channel_num * 1)
        self.skip_reghead_2 = RegHead(channel_num * 2)
        self.skip_reghead_3 = RegHead(channel_num * 4)
        self.skip_reghead_4 = RegHead(channel_num * 8)

        self.up_flow = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.stn = SpatialTransformer_block(mode='bilinear')
        self.diff = VecInt

        self.iter = iter

        self.convgru_1 = ConvGRU(channel_num * 1 * 2, channel_num * 1)
        self.convgru_2 = ConvGRU(channel_num * 2 * 2, channel_num * 2)
        self.convgru_3 = ConvGRU(channel_num * 4 * 2, channel_num * 4)
        self.convgru_4 = ConvGRU(channel_num * 8 * 2, channel_num * 8)

        self.cross_1 = CrossAttention(channel_num * 1, channel_num * 1 // 8)
        self.cross_2 = CrossAttention(channel_num * 2, channel_num * 2 // 8)
        self.cross_3 = CrossAttention(channel_num * 4, channel_num * 4 // 8)
        self.cross_4 = CrossAttention(channel_num * 8, channel_num * 8 // 8)

        self.conv_out_1 = nn.Conv3d(channel_num * 1, channel_num * 1, 3, 1, 1)
        self.conv_out_2 = nn.Conv3d(channel_num * 2, channel_num * 2, 3, 1, 1)
        self.conv_out_3 = nn.Conv3d(channel_num * 4, channel_num * 4, 3, 1, 1)
        self.conv_out_4 = nn.Conv3d(channel_num * 8, channel_num * 8, 3, 1, 1)

    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = self.encoder(fixed)

        ca_1 = self.cross_1(x_mov_1, x_fix_1)
        ca_2 = self.cross_2(x_mov_2, x_fix_2)
        ca_3 = self.cross_3(x_mov_3, x_fix_3)
        ca_4 = self.cross_4(x_mov_4, x_fix_4)

        for i in range(self.iter['4']):
            if i == 0:
                corr_4 = self.block_corr_4(x_mov_4, x_fix_4)
                dec_4 = self.conv_dec_4(concat(x_mov_4, corr_4, x_fix_4))
                h_4, gru_4 = self.convgru_4(None, dec_4)
                dec_4 = self.conv_out_4(dec_4+gru_4)
                flow_4 = self.reghead_4(dec_4)
                flow_4 = self.diff(flow_4)
            else:
                pre_flow = flow_4
                x_warp_4 = self.stn(x_mov_4, pre_flow)
                corr_4 = self.block_corr_4(x_warp_4, x_fix_4)
                dec_4 = self.conv_dec_4(concat(x_warp_4, corr_4, x_fix_4))
                h_4, gru_4 = self.convgru_4(h_4, dec_4)
                dec_4 = self.conv_out_4(dec_4+gru_4)
                flow_4 = self.reghead_4(dec_4)
                flow_4 = flow_4 + self.stn(pre_flow, flow_4)
                flow_4 = self.diff(flow_4)
        context_4 = self.conv_context_4(flow_4, [ca_4, ca_3], dec_4)
        skip_flow_4 = self.skip_reghead_4(context_4)
        flow_4 = flow_4 + skip_flow_4
        flow_4 = self.diff(flow_4)

        for i in range(self.iter['3']):
            if i == 0:
                pre_flow = self.up_flow(flow_4)
                h_3 = self.upsample_3(h_4)
            else:
                pre_flow = flow_3
            x_warp_3 = self.stn(x_mov_3, pre_flow)
            corr_3 = self.block_corr_3(x_warp_3, x_fix_3)
            dec_3 = self.conv_dec_3(concat(x_warp_3, corr_3, x_fix_3))
            h_3, gru_3 = self.convgru_3(h_3, dec_3)
            dec_3 = self.conv_out_3(dec_3+gru_3)
            flow_3 = self.reghead_3(dec_3)
            flow_3 = flow_3 + self.stn(pre_flow, flow_3)
            flow_3 = self.diff(flow_3)
        context_3 = self.conv_context_3(flow_3, [ca_4, ca_3, ca_2], dec_3)
        skip_flow_3 = self.skip_reghead_3(context_3)
        flow_3 = flow_3 + skip_flow_3
        flow_3 = self.diff(flow_3)

        for i in range(self.iter['2']):
            if i == 0:
                pre_flow = self.up_flow(flow_3)
                h_2 = self.upsample_2(h_3)
            else:
                pre_flow = flow_2
            x_warp_2 = self.stn(x_mov_2, pre_flow)
            corr_2 = self.block_corr_2(x_warp_2, x_fix_2)
            dec_2 = self.conv_dec_2(concat(x_warp_2, corr_2, x_fix_2))
            h_2, gru_2 = self.convgru_2(h_2, dec_2)
            dec_2 = self.conv_out_2(dec_2+gru_2)
            flow_2 = self.reghead_2(dec_2)
            flow_2 = flow_2 + self.stn(pre_flow, flow_2)
            flow_2 = self.diff(flow_2)
        context_2 = self.conv_context_2(flow_2, [ca_3, ca_2, ca_1], dec_2)
        skip_flow_2 = self.skip_reghead_2(context_2)
        flow_2 = flow_2 + skip_flow_2
        flow_2 = self.diff(flow_2)

        for i in range(self.iter['1']):
            if i == 0:
                pre_flow = self.up_flow(flow_2)
                h_1 = self.upsample_1(h_2)
            else:
                pre_flow = flow_1
            x_warp_1 = self.stn(x_mov_1, pre_flow)
            corr_1 = self.block_corr_1(x_warp_1, x_fix_1)
            dec_1 = self.conv_dec_1(concat(x_warp_1, corr_1, x_fix_1))
            h_1, gru_1 = self.convgru_1(h_1, dec_1)
            dec_1 = self.conv_out_1(dec_1+gru_1)
            flow_1 = self.reghead_1(dec_1)
            flow_1 = flow_1 + self.stn(pre_flow, flow_1)
            flow_1 = self.diff(flow_1)
        context_1 = self.conv_context_1(flow_1, [ca_2, ca_1], dec_1)
        skip_flow_1 = self.skip_reghead_1(context_1)
        flow_1 = flow_1 + skip_flow_1
        flow_1 = self.diff(flow_1)

        moved = self.stn(moving, flow_1)

        return moved, flow_1