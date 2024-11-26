import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeometricAdstockTransformer(BaseEstimator, TransformerMixin):
    '''
    This class represent a geometric adstock transformation that will create an adstock effect for the data.
    '''

    def __init__(self, alpha: float = 0.0, l: int = 12):
        '''
        arguments:
        -----------
        alpha:
            the adstock variable of decaying effect
        l:
            the length of the adstock effect
        '''
        self.alpha = alpha
        self.l = l

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        cycles = [
            np.append(
                arr=np.zeros(shape=x.shape)[:i],
                values=x[: x.shape[0] - i],
                axis=0
            ) 
            for i in range(self.l)
        ]
        x_cycle = np.stack(cycles, axis=0)
        w = np.array([np.power(self.alpha, i) for i in range(self.l)])
        return np.tensordot(a=w, b=x_cycle, axes=1)

class LogisticSaturationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, mu: float = 0.5):
        self.mu = mu

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return (1 - np.exp(-self.mu * x)) / (1 + np.exp(-self.mu * x))
    
class BetaHillTransformation(BaseEstimator, TransformerMixin):

    def __init__(self, K: float = 0.9, S: float = 0.75, beta: float = 0.4):
        self.K = K
        self.S = S
        self.beta = beta

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.beta - (self.K**(self.S)*self.beta/(x**(self.S) + self.K**self.S))


def jackpotgenerator(datelist):
    for i in datelist:
        if i ==datelist[0]:
            jackpot_size = [1]
        else:
            jackpot = jackpot_size[-1]
            win_prob = np.random.lognormal(mean=0, sigma=1)
            if win_prob > 2.0:
                    # pot is won, jackpot size goes to 1
                    jackpot=1
                    jackpot_size.append(jackpot)
            else:
                    #no win, increase jackpot size and go on
                    if jackpot == 15:
                        jackpot = 15
                    else:
                        jackpot +=1
                    jackpot_size.append(jackpot)
    return jackpot_size
    

    