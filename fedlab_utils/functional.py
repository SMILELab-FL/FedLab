import torch

class AverageMeter(object):
    """Record train infomation"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def evaluate(model, criterion, test_loader, cuda):
    """
    Evaluate local model based on given test :class:`torch.DataLoader`
    Args:
        test_loader (torch.DataLoader): :class:`DataLoader` for evaluation
        cuda (bool): Use GPUs or not
    """
    model.eval()
    loss_sum = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted.eq(labels)).item()
            total += len(labels)
            loss_sum += loss.item()

    accuracy = correct / total
    # TODO: is it proper to use loss_sum here?? CrossEntropyLoss is averaged over each sample
    log_str = "Evaluate, Loss {}, accuracy: {}".format(loss_sum,
                                                       accuracy)
    print(log_str)
    return loss_sum, accuracy