"""
    Simple functionality to generate and analyze toy datasets.
"""
import numpy as np
import random, math


class DS_IXS:

    W_IX = 0   # Weight
    A_IX = -2  # Sensitive attr. (A)
    G_IX = -1  # Class label (G)
    X_RAN = (1, -3) # Normal features (X)


class GenDist:
    """
        Functionality to generate samples based on given distributions
    """

    def __init__(self, x_discrete=True, a_pos=0.5, g_pos=0.5, x_amt=1):
        self.x_discrete = x_discrete
        self.x_amt = x_amt
        self._x_dists = list()
        self._g_pos   = g_pos # Class (G) is uniform in [0, 1] (binary), this is the probability of G = 1
        self._a_pos   = a_pos # Sensitive attr. (A); same as above

        # P(X | a, g) = x_dists[a][g]
        if self.x_amt == 1:
            if x_discrete:
                # Poisson distribution
                self._x_dists.append(
                    {
                        0: {
                            0: 1,
                            1: 7
                        },
                        1: {
                            0: 4,
                            1: 12
                        }
                    })
            else:
                # Gaussian + discretization (round to int)
                self._x_dists.append(
                    {
                        0: {
                            0: (170, 5),
                            1: (155, 5)
                        },
                        1: {
                            0: (165, 5),
                            1: (185, 5)
                        }
                    })
        else:
            self._generate_x_dists()

    def _generate_x_dists(self):
        """
            Sets distributions for features Xi.
        """
        # x[a][g]
        # dists are generated as base_exp + mod
        # base exp depends on a,g 
        # mod depends on whether the feature is proportional or not (in [-2, 3])

        # a, g assuming proportional x
        base_exps = {
            0: {
                0: [1, 2, 3], # Lowest
                1: [7, 8, 9]    # Higher
            },
            1: {
                0: [3, 4, 5],  # 2nd lowest
                1: [11, 12, 13]   # Highest
            }
        }
        
        for i in range(self.x_amt):
            proportionate = bool(round(random.random()))
            xi_dist = None
            base_exp = None
            mod = None
            if proportionate:
                # a = 1, g = 1 imply higher Xi
                base_exp = {
                    0: {
                        0: random.choice(base_exps[0][0]),
                        1: random.choice(base_exps[0][1])
                    },
                    1: {
                        0: random.choice(base_exps[1][0]),
                        1: random.choice(base_exps[1][1])
                    }
                }
                mod = {
                    0: {
                        0: random.randint(-2, 1),
                        1: random.randint(0, 2)
                    }, 
                    1: {
                        0: random.randint(-1, 1),
                        1: random.randint(1, 3)
                    }
                }
            else:
                # a = 0, g = 0 imply higher Xi
                base_exp = {
                    0: {
                        0: random.choice(base_exps[1][1]),
                        1: random.choice(base_exps[1][0])
                    },
                    1: {
                        0: random.choice(base_exps[0][1]),
                        1: random.choice(base_exps[0][0])
                    }
                }
                mod = {
                    0: {
                        0: random.randint(1, 2),
                        1: random.randint(-1, 1)
                    }, 
                    1: {
                        0: random.randint(0, 2),
                        1: random.randint(-2, 1)
                    }
                }

            xi_dist = {
                    0: {
                        0: max(0, base_exp[0][0] + mod[0][0]),
                        1: max(0, base_exp[0][1] + mod[0][1])
                    },
                    1: {
                        0: max(0, base_exp[1][0] + mod[1][0]),
                        1: max(0, base_exp[1][1] + mod[1][1])
                    }
            }
            self._x_dists.append(xi_dist)



    def sample_a(self):
        """
            Returns a binary value for the sensitive attribute according to the configured distribution.
        """
        return int(random.random() < self._a_pos)

    def sample_g(self):
        """
            Returns a binary class label according to the configured distribution.
        """
        return int(random.random() < self._g_pos)

    def sample_x(self, a, g, ix=0):
        """
            Returns one sample from the configured distribution.
            ix specifies which feature Xi should be sampled
        """
        if self.x_discrete:
            return np.random.poisson(lam=self._x_dists[ix][a][g], size=1)[0]
        else:
            return int(random.normalvariate(*self._x_dists[ix][a][g]))

    def p_x_ag(self, x, a, g):
        """
            Returns P (x|a,g) according the the configured distribution.
            Supports multiple features in x.
        """
        p = 1 # Only supports discrete X
        if self.x_discrete:
            
            if not isinstance(x, list):
                x = [x]
            
            for i in range(len(x)): 
                # Poisson mass function:
                lam = self._x_dists[i][a][g]
                p *= ((math.pow(lam, x[i]) * math.exp(-lam)) / math.factorial(x[i]))
        else:
            # Cont. X
            p = 0
        
        return p


class Dataset:
    """
        Represents dataset (X, A, G) and surrounding functionality.
    """
    # Convention:
    # (w, X1, X2, ... Xk, A, G)
    # where 'w' (index 0) is reserved for a tuple's weight
    # A (index -2) is the sensitive feature
    # G (index -1) is the class label
    # Xi are the other features


    def __init__(self, n=10, gen_dists=None):
        self.size = n
        self.data = None
        self.weights = None
        
        if gen_dists is None:
            gen_dists = {
                "x_discrete": True,
                "a_pos": 0.5,
                "g_pos": 0.5,
                "x_amt": 1
            }
        self.generator = GenDist(**gen_dists)
        self._generate()

    @classmethod
    def from_array(cls, arr):
        """
            Returns a Dataset object for a given numpy array
        """
        ds = Dataset(n=0)
        ds.data = arr
        ds.weights = np.ones(ds.data.shape)
        ds.size = ds.data.shape[0]
        ds.generator = None # We don't know what generated the data
        return ds
        
    def _generate(self):
        """
            Generates n datapoints and sets them as the data.
            Weights default to 1.
        """
        # Resets data & weights before operation
        n_features = self.generator.x_amt + 2 # Xi + A + G
        self.data = np.zeros(shape=(self.size, n_features+1))
        self.weights = np.zeros(shape=self.size)
        
        for ix in range(self.size):
            # A & G are independent
            a = self.generator.sample_a()
            g = self.generator.sample_g()
            # X depends on G and A
            x = [self.generator.sample_x(a, g, i) for i in range(n_features-2)]

            pt = np.array([1.0, *x, a, g])
            self.data[ix] = pt
            self.weights[ix] = 1.0
            
    def _matches(self, t, selection):
        """
            True if the tuple t matches the selection.
        """
        if not isinstance(selection, list):
            selection = [selection]
        for attr_ix, val in selection:
            if t[attr_ix] != val:
                return False
        return True
            
    def P(self, selection, conditions=None, ignore_weights=False):
        """
            Returns probability/frequence of selected tuples.
            Also accepts conditions.
            
            Parameters:
                selection: pair or list of pairs (attribute index, value)
                conditions: pair or list of pairs (attribute index, value)
        """
        data = self.data
        if conditions is not None:
            data = self.select(conditions)

        p = 0
        w_total = 0
        for i in range(len(data)):
            w = 1.0 if ignore_weights else data[i][DS_IXS.W_IX]
            p += int(self._matches(data[i], selection)) * w
            w_total += w
        
        if w_total > 0:
            return p / w_total#len(data)
        else:
            return 0 # Selection|Conditions doesn't occur in the dataset
        
    
    def select(self, selection):
        """
            Returns matching datapoints.
        """
        if not isinstance(selection, list):
            selection = [selection]
        # Select on first condition
        attr_ix = selection[0][0]
        val = selection[0][1]
        data = self.data[self.data[:,attr_ix] == val]
        # Filter other conditions
        for i in range(1, len(selection)):
            attr_ix = selection[i][0]
            val = selection[i][1]
            data = data[data[:,attr_ix] == val]
        return data
    
    # Convenience: quickly select UN/UP/PN/PP groups
    def select_UN(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        return self.select([(A_IX, 0), (G_IX, 0)])
    
    def select_UP(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        return self.select([(A_IX, 0), (G_IX, 1)])
    
    def select_PN(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        return self.select([(A_IX, 1), (G_IX, 0)])
    
    def select_PP(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        return self.select([(A_IX, 1), (G_IX, 1)])

    
    # Data access
    def get_features(self):
        """
            Returns all features (including sensitive attribute) for all tuples.
        """
        return self.data[:,1:-1]
    
    def get_labels(self):
        """
            Returns labels of all tuples in the dataset
        """
        return self.data[:, DS_IXS.G_IX]

    def get_weights(self):
        """
            Returns list of weights associated with tuples in the dataset.
        """
        return self.data[:, DS_IXS.W_IX]

    def train_test_split(self, train_percent=0.7):
        """
            Returns train_X, train_Y, train_W, test_X, test_Y splits for the dataset.
        """
        X = self.get_features()
        W = self.get_weights()
        Y = self.get_labels()

        train_bound = math.floor(len(self.data) * train_percent)
        X_train = X[:train_bound]
        Y_train = Y[:train_bound]
        W_train = W[:train_bound]
        X_test = X[train_bound:]
        Y_test = Y[train_bound:]

        return X_train, Y_train, W_train, X_test, Y_test

    
    # Fairness metrics
    def disparate_impact(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        return self.P(selection=(G_IX, 1), conditions=(A_IX, 1)) \
                / self.P(selection=(G_IX, 1), conditions=(A_IX, 0))


    # Convenience: information printouts
    def general_stats(self, A_IX=DS_IXS.A_IX, G_IX=DS_IXS.G_IX):
        """
            Prints out general distribution of privileged/unprivileged positive/negative samples.
        """
        print(f"% Priv./Unpriv.: {self.P((A_IX, 1)):.3f} / {self.P((A_IX, 0)):.3f}")
        print(f"% Pos./Neg.: {self.P((G_IX, 1)):.3f} / {self.P((G_IX, 0)):.3f}")
        print(f"P(G = 1 | A = 0): {self.P(selection=(G_IX, 1), conditions=(A_IX, 0)):.3f}") # Unpriv. pos
        print(f"P(G = 0 | A = 0): {self.P(selection=(G_IX, 0), conditions=(A_IX, 0)):.3f}") # Unpriv. neg
        print(f"P(G = 1 | A = 1): {self.P(selection=(G_IX, 1), conditions=(A_IX, 1)):.3f}") # Priv. pos
        print(f"P(G = 0 | A = 1): {self.P(selection=(G_IX, 0), conditions=(A_IX, 1)):.3f}") # Priv. neg


    def conf_matrix(self, other):
        """
            Prints a confusion matrix given another dataset.
            Datasets must have the same size.
        """
        # This only makes sense when 'other' is a transformed version of this DS

        from sklearn.metrics import confusion_matrix

        Y_this  = self.get_labels()
        Y_other = other.get_labels() 

        cfm = confusion_matrix(Y_this, Y_other)
        tn, fp, fn, tp = cfm.ravel()

        print(f"TN: {tn}")
        print(f"FN: {fn}")
        print(f"TP: {tp}")
        print(f"FP: {fp}")

    
    def P_class_ax(self, c, a, x):
        """
            Returns P(Class = c | a, x) using a more precise method than simply counting.
            (Based on the theoretical value from the generating distrubitions)
        """

        if self.generator is None:
            raise Exception("Generator unknown. Dataset was likely initialised from an array.")

        # P(c | a, x) =
        # Numerator:    P( x | a, g ).P(G=g) (use generator distr.)
        # Denominator:  P( x | a, G=0).P(G=0) + P ( x | a, G=1).P(G=1)

        p_gpos = self.generator._g_pos
        p_g = p_gpos
        if c == 0:
            p_g = 1.0 - p_gpos

        return ((self.generator.p_x_ag(x, a, c) * p_g)
               / ((self.generator.p_x_ag(x, a, g=0) * (1.0 - p_gpos)) + (self.generator.p_x_ag(x, a, g=1) * p_gpos)))
