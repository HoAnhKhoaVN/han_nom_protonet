from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch import sum


# loss averager
class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
loss_avg = Averager()

def evaluate(
    model: Module,
    evaluation_loader: DataLoader,
    device: str
):
    n_correct = 0
    length_of_data = 0
    valid_loss_avg = Averager()
    val_iter = iter(evaluation_loader)
    for batch in val_iter:
        image_tensors, labels = batch
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        images = image_tensors.to(device)
        labels = labels.to(device)


        outputs = model(images)

        cost = cross_entropy(outputs, labels)
        valid_loss_avg.add(cost)

        _, preds = outputs.max(1)

        n_correct +=sum(preds==labels).cpu().detach().numpy().tolist()
    accuracy = n_correct / float(length_of_data) * 100
    return (
        accuracy,
        valid_loss_avg.val()
    )