import time
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
# from utils.eval import accuracy
# from utils.adv import trades_loss

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# TODO: add adversarial accuracy.
def trades_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size=2.0 / 255,
    epsilon=8.0 / 255,
    perturb_steps=10,
    beta=6.0,
    clip_min=0,
    clip_max=1.0,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
        x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    if distance == "l_inf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    print(loss)
    print(loss_natural)
    # Zero gradients before the backward pass
    # optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to model parameters
    # loss.backward()
    # Assuming you have already computed gradients by calling loss.backward()
    # # Access the gradients of each parameter
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f'Parameter: {name}, Gradient Norm: {param.grad.norm()}')

    # Calling the step function on an Optimizer makes an update to its parameters
    # optimizer.step()

    return loss



def train_adv(model, device, train_loader, sm_loader, optimizer, epoch):
    for i in range(epoch):
        print("epoch",i)
        model.train()
        end = time.time()

        dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

        for i, data in enumerate(dataloader):
            if sm_loader:
                images, target = (
                    torch.cat([d[0] for d in data], 0).to(device),
                    torch.cat([d[1] for d in data], 0).to(device),
                )
            else:
                images, target = data[0].to(device), data[1].to(device)

            # basic properties of training data
            if i == 0:
                print(
                    images.shape,
                    target.shape,
                    f"Batch_size from args: {250}",
                    "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
                )
                print(f"Training images range: {[torch.min(images), torch.max(images)]}")

            output = model(images)

            # calculate robust loss
            loss = trades_loss(
                model=model,
                x_natural=images,
                y=target,
                device=device,
                optimizer=optimizer
            )

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))


            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model  # Return the trained model after all epochs are completed


        # if i % args.print_freq == 0:
        #     progress.display(i)
        #     progress.write_to_tensorboard(
        #         writer, "train", epoch * len(train_loader) + i
        #     )
        #
        # # write a sample of training images to tensorboard (helpful for debugging)
        # if i == 0:
        #     writer.add_image(
        #         "training-images",
        #         torchvision.utils.make_grid(images[0 : len(images) // 4]),
        #     )