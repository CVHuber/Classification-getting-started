from __future__ import print_function, absolute_import

__all__ = ['AverageMeter', 'accuracy']

# 计算及保存当前值和平均值
class AverageMeter(object):
    def __init__(self):
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

# 计算topk准确率
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    # 选出k个概率最高值
    _, pred = output.topk(maxk, 1, True, True)
    # tensor转置
    pred = pred.t()
    # 计算正确的个数
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # 计算topk值
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

