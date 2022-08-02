import torch
import torch.nn as nn
import torch.nn.functional as F
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 6
batch_size  = 18#10
learning_rate = 1e-3
num_epochs = 500 # max epoch
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()#torch.gt(a,b)函数比较a中元素大于（这里是严格大于）b中对应元素，
                                #大于则为1，不大于则为0，这里a为Tensor，b可以为与a的size相同的Tensor或常数。

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):
#     print((mem * decay * (1. - spike)).shape,ops(x).shape)
    mem = mem* decay + ops(x)###################################mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    mem=mem-thresh*spike

    return mem, spike

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [10, 10, 10]#[10, 5, 3]#[28, 14, 7]
# fc layer
cfg_fc = [64, 6]########################

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(3200, cfg_fc[0])#（2,2,32,128）##cfg_kernel[-1] * cfg_kernel[-1] * 
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])#（128,6）

    def forward(self, input, time_window = 20):
        spikes = 0
        post_spikes=0
        batch_size=input.size(0)
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[1], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[2], cfg_kernel[2], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

#         print('input',input.size())#input torch.Size([18, 100, 10, 10])

        for step in range(input.size(1)): # simulation time steps
#             print('in:',input.float())
#             print(torch.rand(input.size(), device=device))# prob. firing
            x = input[:,step,:,:]
            x=x[:, None, :,:]
#             print('x',x.shape)#x torch.Size([18, 100, 10, 10])

            spikes=spikes+(sum(sum(sum(sum(x))))).data.cpu().numpy()
            post_spikes=post_spikes+(sum(sum(sum(sum(x))))).data.cpu().numpy()*6.25*32

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)###################################
#             print('c1_spike',c1_spike.shape)
            
            spikes=spikes+(sum(sum(sum(sum(c1_spike))))).data.cpu().numpy()
            post_spikes=post_spikes+(sum(sum(sum(sum(c1_spike))))).data.cpu().numpy()*6.25*32#*32#
#             print("spikes",spikes)
            
#             x = F.avg_pool2d(c1_spike, 2)
#             print(c1_spike.shape,c2_mem.shape,c2_spike.shape)
            c2_mem, c2_spike = mem_update(self.conv2,c1_spike, c2_mem,c2_spike)
#             print(c2_mem)
#             print('c2_spike',c2_spike.shape)
            spikes=spikes+(sum(sum(sum(sum(c2_spike))))).data.cpu().numpy()
            post_spikes=post_spikes+(sum(sum(sum(sum(c2_spike))))).data.cpu().numpy()*64
#             x = F.avg_pool2d(c2_spike, 2)
#             print(x.shape)
            y = c2_spike
            y = y.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, y, h1_mem, h1_spike)################################
#             print('h1_spike',h1_spike.shape)
            spikes=spikes+(sum(sum(h1_spike))).data.cpu().numpy()
            post_spikes=post_spikes+(sum(sum(h1_spike))).data.cpu().numpy()*6
#             print("spikes",spikes)

            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
#             print('h2_spike',h2_spike.shape)
            spikes=spikes+(sum(sum(h2_spike))).data.cpu().numpy()
#             post_spikes=post_spikes+(sum(sum(h2_spike))).data.cpu().numpy()*6
            
            h2_sumspike += h2_spike
        
        outputs = h2_sumspike / input.size(1)
#         print("spikes",spikes)
#         print('out:',outputs.shape)
#         print('outp:',outputs)
        return outputs,spikes,post_spikes


