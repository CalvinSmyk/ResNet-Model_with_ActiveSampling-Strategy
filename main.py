import torch.cuda
import time
import torch.optim as optim
import os
import shutil
import torchvision.transforms as T
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from ResNetModel import ResNet,Bottleneck,BasicBlock
from load_data import Data
from Dataloader import MyDataSet
from torch.utils.data import DataLoader
from preprocessing import preprocess

class Config(object):
    DATA = "CIFAR_100"
    BATCH_SIZE = 64
    NUM_EPOCH = 25
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 100
    PRINTING = 100
    TRAIN_TRANSFORM = T.Compose([T.RandomCrop(32,padding=4),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.507,0.487,0.441], std= [0.267,0.256,0.276])
                           ])
    TEST_TRANSFORM = T.Compose([T.ToTensor(),
                           T.Normalize(mean=[0.507,0.487,0.441], std= [0.267,0.256,0.276])
                           ])

class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum / self.count

def adjust_lr(optimizer,epoch):
    if epoch < 80:
        lr = Config.LEARNING_RATE
    elif epoch < 120:
        lr = Config.LEARNING_RATE *0.1
    else:
        lr = Config.LEARNING_RATE * 0.01
    for param in optimizer.param_groups:
        param['lr'] = lr

def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    bs = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / bs))
    return res

def train(train_dataloader,model,criterion,optimizer,epoch):
    batch_time = Average()
    data_time = Average()
    losses = Average()
    top1 = Average()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        input, target = input.to(device=device).float(), target.to(device)

        output = model(input).float()
        loss = criterion(output, target)

        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % Config.PRINTING == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                epoch, i, len(train_dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = Average()
    losses = Average()
    top1 = Average()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device).float(), target.to(device)

            # compute output
            output = model(input).float()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % Config.PRINTING == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/resnet100_cifar100'
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    """HERE WE TRY THE PRE THING"""

    """print('=> loading cifar100 data...')
    normalize = T.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    torchvision.datasets.CIFAR100
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            normalize,
        ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)"""

    """DONE WITH TRYING"""

    train_data,train_labels,classes = Data(download=False,path="./cifar-100-python/train",train=True).open_data()
    test_data,test_labels = Data(download=False,path="./cifar-100-python/test",train=False).open_data()
    pre =preprocess(train_data,train_labels)
    pre.order_by_class()
    pre.selection_process(400)
    train_data,train_labels = pre.get_corresponding_images_and_labels()
    train_dataset = MyDataSet(train_data, train_labels,transforms=Config.TRAIN_TRANSFORM)
    test_dataset = MyDataSet(test_data,test_labels,transforms=Config.TEST_TRANSFORM)

    train_dataloader = DataLoader(train_dataset,batch_size=Config.BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=Config.BATCH_SIZE,shuffle=False)

    device = torch.device('mps')
    model = ResNet(Bottleneck,[3,4,23,3])
    model = model.to(device=device)
    criterion = torch.nn.CrossEntropyLoss().to(device,dtype=torch.float)
    optimizer = optim.SGD(model.parameters(), Config.LEARNING_RATE, momentum=Config.MOMENTUM,
                          weight_decay=Config.WEIGHT_DECAY)

    best_prec = 0
    for epoch in range(Config.NUM_EPOCH):
        adjust_lr(optimizer,epoch)

        train(train_dataloader,model,criterion,optimizer,epoch)

        prec = validate(test_dataloader,model,criterion)

        best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'best_prec':best_prec,
            'optimizer':optimizer.state_dict(),
        }, best,fdir)













