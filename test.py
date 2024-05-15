import torch
from torch.distributions import Normal


mu = torch.zeros(8, 40, 2).normal_(0,1)
sigma = torch.abs(torch.zeros(8, 40, 2).normal_(0,1))

p = Normal(mu, sigma)
print(p.rsample().shape)

y = torch.zeros(8, 40, 2).normal_(0,1)
print(p.log_prob(y).mean(dim=0).sum())

