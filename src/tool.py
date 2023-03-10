import torch

import numpy as np

class EarlyStopping:
    # """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0,cv_index = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cv_index = cv_index

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: %d out of %d'%(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                # test
                # self.early_stop = False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.5f --> %.5f).  Saving model ...'%(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), './checkpoints/checkpoint%d.pt'%self.cv_index)
        self.val_loss_min = val_loss
