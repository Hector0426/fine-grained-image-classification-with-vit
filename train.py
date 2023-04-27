import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, criterion_mix, alpha, epoch):
    model.train()
    for step, data in enumerate(tqdm(train_loader, desc='Training %d epoch' % epoch)):
        images, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        out110, out111, out112, out113, image_new, lam_a, lam_b, rand_index = model(images, swap=True)
        out210, out211, out212, out213 = model(image_new.detach(), swap=False)
        target_b = labels[rand_index]
        loss = criterion(out110, labels) * alpha + \
               criterion(out111, labels) * alpha + \
               criterion(out112, labels) + \
               criterion(out113, labels) + \
               torch.mean(criterion_mix(out210, labels) * lam_a + criterion_mix(out210, target_b) * lam_b) * alpha + \
               torch.mean(criterion_mix(out211, labels) * lam_a + criterion_mix(out211, target_b) * lam_b) * alpha + \
               torch.mean(criterion_mix(out212, labels) * lam_a + criterion_mix(out212, target_b) * lam_b) + \
               torch.mean(criterion_mix(out213, labels) * lam_a + criterion_mix(out213, target_b) * lam_b)
        loss.backward()
        optimizer.step()
    scheduler.step()


def valid(model, test_loader, epoch, beta=None):
    if beta is None:
        beta = [1, 1, 1, 1]
    test_total = len(test_loader.dataset)
    acc = [0] * 5
    with torch.no_grad():
        model.eval()
        for step, data in enumerate(tqdm(test_loader, desc='Test %d epoch' % epoch)):
            images, labels = data[0].cuda(), data[1].cuda()
            outs = model(images, swap=False)
            for i in range(4):
                prediction = outs[i].argmax(dim=1)
                acc[i] += torch.eq(prediction, labels).sum().float().item()
            prediction = (outs[0] * beta[0] + outs[1] * beta[1] +
                          outs[2] * beta[2] + outs[3] * beta[3]).argmax(dim=1)
            acc[4] += torch.eq(prediction, labels).sum().float().item()
        acc = [a / test_total for a in acc]
        info = '[epoch {}] ACC: {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'.format(epoch, acc[0], acc[1], acc[2], acc[3],
                                                                              acc[4])
        print(info)
