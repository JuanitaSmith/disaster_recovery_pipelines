""" Class to use calculate focal loss to be used with sckit-learn algorithms"""

import numpy as np

class FocalBinaryLoss:
    """
    The class calculates focal loss for multi-label binary classification that contains imbalanced data.

    This class was inspired by this [kaggle blob](https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained)
    I changed the coding from a pytorch framework, to use numpy instead to use with XGBOOST

    The gamma parameter is used to reduce the importance of majority classes,
    forcing the model to focus more on minority classes.

    A model trained with focal loss focuses relatively more on minority class patterns. As a result, it performs better.

    Args:
        gamma → float: down weighing of majority classes

    Return:
        focal_loss → float - average focal loss
    """

    def __init__(self, gamma=2):
        self.gamma = gamma


    def sigmoid(self, x):
        """ Calculate sigmoid function """
        return 1 / (1 + np.exp(-x))


    def binary_cross_entropy(self, pred, y):
        """ Calculate binary cross entropy loss """
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()


    def focal_binary_cross_entropy(self, pred, targets):
        """ Focal Multilabel Loss

         Function was inspired from a pytorch implementation, see [ref here](https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained)

         Args:
             pred: label predictions
             targets: actual labels

        Returns:
             focal_loss: focal loss for multilabel classification
        """
        # number of classes we try to predict
        num_labels = targets.shape[0]

        # Flatten the logits and targets so that we can compare two vectors
        l = pred.reshape(-1)
        t = targets.reshape(-1)

        # Sigmoid is required when calculating the loss from logits (meaning it's not probability predictions, but 1 and 0)
        # We apply sigmoid to the logits to squeeze the values between 0 and 1.
        p = self.sigmoid(l)

        # We are following now the standard binary cross entropy with logits loss. For positive examples, we'll take the sigmoid, for negative examples, we'll take 1-sigmoid. Good predictions will be close to 1, bad predictions will be close to 0.
        p = np.where(t >= 0.5, p, 1 - p)

        # Clamping the input to avoid being to close to zero or one. This is probably for numeric stability.
        clamp_p = np.clip(p, 1e-4, 1 - 1e-4)

        # Now we apply negative log - this is the BCE loss. This will convert good predictions to a loss that is close to 0, and bad predictions will go to infinity.
        logp = - np.log(clamp_p)
        # logp = - np.log(np.clip(p, 1e-4, 1-1e-4))

        loss = logp * ((1 - p) ** self.gamma)
        loss = num_labels * loss.mean()

        return loss
