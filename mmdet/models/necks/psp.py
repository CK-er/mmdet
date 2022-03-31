self.psp_pool_output_size = sizes
self.stages = []
self.stages = nn.ModuleList()
self.stages.extend(
    [nn.Sequential(F.adaptive_avg_pool2d(inputs[-1], output_size=(size, size)), nn.Conv2d(in_channels[-1], out_channels, kernel_size=1, bias=False)) for size in self.psp_pool_output_size])
self.bottleneck = nn.Conv2d(out_channels * (len(self.psp_pool_output_size) + 1),  out_channels, kernel_size=1)
self.relu = nn.ReLU()


h, w = inputs[-1].size(2), inputs[-1].size(3)
priors = [F.upsample(input=stage(inputs[-1]), size=(h, w), mode='bilinear') for stage in self.stages] + [inputs[-1]]
bottle = self.bottleneck(torch.cat(priors, 1))
return self.relu(bottle)