import torch
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter()

a = torch.distributions.Normal(1,1)
a = a.sample((100,))
for i in range(10):
    writer.add_histogram('a',a.reshape(-1),i)
writer.close()