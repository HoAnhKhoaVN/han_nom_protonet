# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from han_nom_dataset import HanNomDataset, ClassificationDataset
from torch.utils.data import DataLoader
from mobilenet import MyMobileNetV2
from parser_util import get_parser
from evaluate import evaluate
from tqdm import tqdm
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from pickle import dump
from torch.utils.tensorboard import SummaryWriter
# from omniglot_dataset import OmniglotDataset
# from protonet import ProtoNet

def write_plk(
    fn: str,
    data
):
    with open(fn, 'wb') as f:
        dump(data, f)

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_dataset(opt, mode):
    # dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    dataset = HanNomDataset(mode = mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels, # All labels
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)

def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    # write_plk(
    #     fn=f'{mode}_dataset.plk',
    #     data= dataset
    # )
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader, dataset.classes

def init_protonet(
        num_classes: int,
        checkpoint_path: str,
        cuda: bool=False,
    ):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
    model = MyMobileNetV2(num_classes).to(device)

    # region Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path))
    # endregion

    return model

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(
        opt,
        tr_dataloader,
        model,
        optim,
        lr_scheduler,
        writer: SummaryWriter,
        val_dataloader=None,
        ):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    iteration = 0
    for epoch in range(opt.epochs):
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tr_iter:
            iteration+=1
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())


        lr_scheduler.step()
        if iteration % 500 == 0:
          avg_loss = np.mean(train_loss[-opt.iterations:])
          avg_acc = np.mean(train_acc[-opt.iterations:])
          print(f'iter: {iteration}')
          print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
          writer.add_scalar('training loss',
                              avg_loss,
                              epoch)
          
          writer.add_scalar('Train Acc',
                      avg_acc,
                      epoch)

          # region Validation on task
          if val_dataloader is None:
              continue
          val_iter = iter(val_dataloader)
          model.eval()
          for batch in val_iter:
              x, y = batch
              x, y = x.to(device), y.to(device)
              model_output = model(x)
              loss, acc = loss_fn(model_output, target=y,
                                  n_support=opt.num_support_val)
              val_loss.append(loss.item())
              val_acc.append(acc.item())
          avg_loss = np.mean(val_loss[-opt.iterations:])
          avg_acc = np.mean(val_acc[-opt.iterations:])

          postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
              best_acc)
          print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
              avg_loss, avg_acc, postfix))
          
          writer.add_scalar('Avg Val Loss',
                              avg_loss,
                              iteration)
          
          writer.add_scalar('Avg Val Acc',
                      avg_acc,
                      iteration)

          if avg_acc >= best_acc:
              torch.save(model.state_dict(), best_model_path)
              best_acc = avg_acc
              best_state = model.state_dict()
          # endregion

        #   # region Validation on classification
        #   if val_classifier_dataloader is None:
        #       continue

        #   cur_acc_cls, valid_loss_cls= evaluate(
        #       model,
        #       val_classifier_dataloader,
        #       device
        #   )
        #   postfix = ' (Best)' if cur_acc_cls >= best_acc_cls else f' (Best: {best_acc_cls})'
        #   print(f"cur_acc_cls: {cur_acc_cls} - valid_loss_cls: {valid_loss_cls} {postfix}")
        #   writer.add_scalar('Avg cur_acc_cls',
        #               cur_acc_cls,
        #               iteration)
          
        #   writer.add_scalar('Avg valid_loss_cls',
        #               valid_loss_cls,
        #               iteration)

        #   if cur_acc_cls >= best_acc_cls:
        #       torch.save(model.state_dict(), best_model_cls_path)
        #       best_acc_cls = cur_acc_cls
        #       best_state = model.state_dict()
        ## endregion
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options, opt.cuda)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    # region Tensorboard
    writer = SummaryWriter(
        log_dir=options.experiment_root
    )
    # endregion

    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader, classes = init_dataloader(options, 'train')
    val_dataloader, _ = init_dataloader(options, 'val')
    test_dataloader, _ = init_dataloader(options, 'test')


    # transform = transforms.Compose(
    # [
    #     transforms.ToTensor(),
    #     transforms.Resize((32, 32), antialias= False),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #  ])

    # val_classifier_dataloader = DataLoader(
    #     dataset= ClassificationDataset(
    #         mode= 'val',
    #         root_dir= options.dataset_root,
    #         transform = transform
    #     ),
    #     batch_size= options.batch_size_classify,
    # )

    # test_classifier_dataloader = DataLoader(
    #     dataset= ClassificationDataset(
    #         mode= 'test',
    #         root_dir= options.dataset_root,
    #         transform = transform
    #     ),
    #     batch_size= options.batch_size_classify,
    # )

    model = init_protonet(
        num_classes= len(classes),
        cuda = options.cuda,
        checkpoint_path= os.path.join(options.experiment_root, 'best_model.pth')
    )
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                writer= writer
            )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)
if __name__ == '__main__':
    main()
