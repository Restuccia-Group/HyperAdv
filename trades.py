import torch
import torch.nn as nn
import torch.nn.functional as F


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                step_size=0.003,
                epsilon=0.05,
                perturb_steps=10,
                beta=5.0,
                distance='l_inf',
                mode='train',
                hypercnn = False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean',log_target=True)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad = True
            if hypercnn:
                output_natural = model(x_natural,True)
                output_adv = model(x_adv,True)
                loss_kl = 0
                for j in range(len(output_adv)):
                    loss_kl += criterion_kl(F.log_softmax(output_adv[j],dim=1),
                                            F.log_softmax(output_natural[j],dim=1))
                loss_kl /= len(output_adv)
            else:    
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.log_softmax(model(x_natural), dim=1))
            model.zero_grad()
            loss_kl.backward()
            grad = x_adv.grad.data
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon).detach()
        loss_kl = 0
    else:
        raise NotImplementedError
    
    if mode == "train":
        model.train()
    
    if hypercnn:
        output_natural = model(x_natural,True)
        output_adv = model(x_adv,True)
        loss_natural = 0
        loss_robust = 0
        for k in range(len(output_natural)):
            loss_natural += F.cross_entropy(output_natural[k], y)
            loss_robust += criterion_kl(F.log_softmax(output_adv[k], dim=1),
                                        F.log_softmax(output_natural[k], dim=1))
        loss_natural /= len(output_natural)
        loss_robust /= len(output_natural)
    else:
        logits = model(x_natural)
        logits_adv = model(x_adv)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                   F.log_softmax(logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss