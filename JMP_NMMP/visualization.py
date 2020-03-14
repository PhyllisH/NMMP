import visdom
import time
import numpy as np

'''
Borrowed from https://blog.csdn.net/LXX516/article/details/79019328
'''

class Visualizer(object):
    def __init__(self, env='default', port = 8097, **kwargs):
        self.vis = visdom.Visdom(server='202.120.39.165', port=port, env=env, **kwargs)
        self.index = {}         
    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name=list(d.keys())
        name_total=" ".join(name)
        x = self.index.get(name_total, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        #print(x)
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=str(name_total),#unicode
                    opts=dict(legend=name,
                        title=name_total),
                    update=None if x == 0 else 'append'
                    )
        self.index[name_total] = x + 1 

'''
https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'