import math


def inv_lr_scheduler(optimizer, epoch, epochs, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        
        
    return optimizer

