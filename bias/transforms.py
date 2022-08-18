from bias.data import DS_IXS
import numpy as np
# Transformations on arrays


### GENERAL/ATOMIC TRANSFORMATIONS ###
def modify(ds, l):
    """
        Probabilistically swaps labels according to specifications.
        This is done in-place

        Parameters:
            * ds : Array
            * l  : Lambda object (target & pos. outcome prob.)
    """

    new_ds = np.copy(ds)
    # Iterate over dataset
    for r in range(len(ds)):
        t = ds[r]
        if l.get_outcome(t[DS_IXS.A_IX], t[DS_IXS.G_IX]) > 0:
            # Swap label if lambda = +
            new_ds[r][DS_IXS.G_IX] = int(1 - t[DS_IXS.G_IX])
    return new_ds


def addition(ds, l, gamma):
    """
        Probabilistically duplicate rows according to specifications.

        Parameters:
            * ds : Array
            * l  : Lambda object (target & pos. outcome prob.)
            * gamma : duplication distribution
    """
    new_ds = np.copy(ds)
    for r in range(len(ds)):
        t = ds[r]
        if l.get_outcome(t[DS_IXS.A_IX], t[DS_IXS.G_IX]) > 0:
            # Add gamma duplicates
            n_dupes = gamma
            dupes = np.repeat(t, n_dupes, axis=0)
            new_ds = np.append(new_ds, [dupes], axis=0)
    return new_ds

def reweighing(ds, l, gamma):
    """
        Probabilistically reweigh rows according to specifications.
        Effectively identical to addition, but doesn't actually insert duplicates & allows continuous weights.

        Parameters:
            * ds : Array
            * l  : Lambda object (target & pos. outcome prob.)
            * gamma : weight distribution
    """
    new_ds = np.copy(ds)
    for r in range(len(ds)):
        t = ds[r]
        if l.get_outcome(t[DS_IXS.A_IX], t[DS_IXS.G_IX]) > 0:
            # Add gamma duplicates
            new_ds[r][DS_IXS.W_IX] = gamma
    return new_ds


def deletion(ds, l):
    """
        Probabilistically delete rows according to specifications.

        Parameters:
            * ds : Array
            * l  : Lambda object (target & pos. outcome prob.)
    """

    to_del = list()
    for r in range(len(ds)):
        t = ds[r]
        if l.get_outcome(t[DS_IXS.A_IX], t[DS_IXS.G_IX]) > 0:
            to_del.append(r)
    return np.delete(ds, to_del, axis=0)


### REWEIGHTING METHOD ###

class Reweigher:
    """
        Groups functionality for assigning new weights to a dataset in order to make it fair.
    """

    def __init__(self, og_ds):
        self.og_ds = og_ds
        self.fair_data = None
        self._W = dict()

    def _calc_W(self, a, g):
        """
            For each combination (a, g), returns the appropriate weight.
        """
        # This weight is calculated by taking the ratio between the
        # Expected probability (P_exp(g, a)) and the Observed probability (P_obs(g, a))

        d_size = len(self.og_ds.data)
        # Expected probability of (G = g ^ A = a)
        #P_exp = len(self.og_ds.select(selection=[(DS_IXS.A_IX, a)])) / d_size
        #P_exp *= len(self.og_ds.select(selection=[(DS_IXS.G_IX, g)])) / d_size
        P_exp = self.og_ds.P(selection=(DS_IXS.A_IX, a)) * self.og_ds.P(selection=(DS_IXS.G_IX, g))

        # Observed probability of (G = g ^ A = a)
        P_obs = len(self.og_ds.select(selection=[(DS_IXS.A_IX, a), (DS_IXS.G_IX, g)])) / d_size

        return P_exp / P_obs

    def W(self, a, g):
        if a not in self._W:
            self._W[a] = dict()

        if g not in self._W[a]:
            w = self._calc_W(a, g)
            self._W[a][g] = w

        return self._W[a][g]

    def reweigh(self):
        """
            Sets the weights in the updated dataset
        """
        if self.fair_data is None:
            self.fair_data = np.copy(self.og_ds.data)

        for r in range(len(self.fair_data)):
            w = self.W(self.fair_data[r][DS_IXS.A_IX], self.fair_data[r][DS_IXS.G_IX])
            self.fair_data[r][DS_IXS.W_IX] = w

    def get_weights(self):
        """
            Returns a (flat) array corresponding to the weights of the tuples.
        """
        return self.fair_data[:,DS_IXS.W_IX]