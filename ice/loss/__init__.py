from __future__ import absolute_import

from .contrastive import ViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, Weighted_CrossEntropyLabelSmooth
from .triplet import TripletLoss, SoftTripletLoss

__all__ = [
    'CrossEntropyLabelSmooth',
    'Weighted_CrossEntropyLabelSmooth',
    'SoftEntropy',
    'TripletLoss',
    'SoftTripletLoss',
    'ViewContrastiveLoss']
