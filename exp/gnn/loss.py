import torch


### Based on https://arxiv.org/pdf/2205.03169
def nt_xent_loss(sim, temperature):
    sim = sim / temperature
    n = sim.shape[0] // 2  # n = |user_batch|

    aligment_loss = -torch.mean(sim[torch.arange(n), torch.arange(n)+n])

    mask = torch.diag(torch.ones(2*n, dtype=torch.bool)).to(sim.device)
    sim = torch.where(mask, -torch.inf, sim)
    sim = sim[:n, :]
    distribution_loss = torch.mean(torch.logsumexp(sim, dim=1))

    loss = aligment_loss + distribution_loss
    return loss