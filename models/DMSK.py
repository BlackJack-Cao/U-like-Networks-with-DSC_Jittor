import jittor as jt
import jittor.nn as nn

class StraightThroughArgmax(jt.Function):
    def execute(self, logits):
        indices = jt.argmax(logits, dim=1)[0]
        y_hard = jt.nn.one_hot(indices, num_classes=logits.shape[1])
        y_soft = nn.softmax(logits, dim=1)
        self.y_soft = y_soft
        return y_hard - y_soft.stop_grad() + y_soft
    
    def grad(self, grad_output):
        y_soft = self.y_soft
        grad_logits = grad_output * y_soft
        return grad_logits

class DynamicKernelSelection(nn.Module):
    def __init__(self, in_channel, kernel_sizes_1=[3, 5], kernel_sizes_2=[7, 9, 11]):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_sizes_1 = kernel_sizes_1
        self.kernel_sizes_2 = kernel_sizes_2
        
        self.conv_layers_1 = nn.ModuleList([
            nn.Conv2d(in_channel, in_channel, kernel_size=k, 
                     padding=k//2, groups=in_channel)
            for k in kernel_sizes_1
        ])
        
        self.conv_layers_2 = nn.ModuleList([
            nn.Conv2d(in_channel, in_channel, kernel_size=k, 
                     padding=k//2 + (k//2) * 2, dilation=3, groups=in_channel)
            for k in kernel_sizes_2
        ])
        
        self.attention_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, len(kernel_sizes_1), kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.attention_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, len(kernel_sizes_2), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def execute(self, x):
        weights_1 = self.attention_1(x)
        weights_2 = self.attention_2(x)
        output_1 = self._ste_kernel_selection(x, weights_1, self.conv_layers_1)
        output_2 = self._ste_kernel_selection(output_1, weights_2, self.conv_layers_2)
        return output_1, output_2
    
    def _ste_kernel_selection(self, x, weights, conv_layers):
        B = x.shape[0]
        num_kernels = len(conv_layers)
        logits = weights.view(B, num_kernels)
        selection = StraightThroughArgmax.apply(logits)
        output = 0
        for i, conv in enumerate(conv_layers):
            weight = selection[:, i:i+1, None, None]
            output = output + weight * conv(x)
        return output

class DMSK(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channel_proj = nn.Conv2d(in_channel, in_channel // 2, 
                                      kernel_size=1, bias=False)
        self.dynamic_kernel_selection = DynamicKernelSelection(in_channel // 2)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def execute(self, x):
        x_proj = self.channel_proj(x)
        att1, att2 = self.dynamic_kernel_selection(x_proj)
        out = jt.concat([att1, att2], dim=1)
        avg_att = jt.mean(out, dim=1, keepdims=True)
        max_att = jt.max(out, dim=1, keepdims=True)
        att = jt.concat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        out = out * att[:, 0, :, :].unsqueeze(1) + out * att[:, 1, :, :].unsqueeze(1)
        output = out + x
        return output

class DMSKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DMSK(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def execute(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x
