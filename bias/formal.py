"""
    Implementations of formalised functions
"""
import random

### Assumptions ###
A_IX = 1
G_IX = 2

### Helper class for lambda ###
class Lambda:
    """
        This class can be used to describe the outcomes of any operations.
        Targets a single group (un/privileged, pos-/negative), 
        has a given probability of positive outcome for targeted group.
    """
    
    def __init__(self, target_a, target_g, p_pos):
        self.target_a = target_a
        self.target_g = target_g
        self.p_pos = p_pos
    
    def set_target(self, a_val, g_val):
        """
            Single operations always target a single group.
            Any tuples that don't match this target have 0 probability of being affected.
        """
        self.target_a = a_val
        self.target_g = g_val
        
    def is_target(self, a, g):
        """
            True if given features match the target group.
        """
        return (a == self.target_a) and (g == self.target_g)
        
    def get_outcome(self, a, g):
        """
            Generates either a positive (1) or negative (0) outcome based on sensitive attribute A, and label G.
            Outcome 1 always corresponds to executing the operation.
        """
        if self.is_target(a, g):
            # Outcome as described by p_pos
            return int(random.random() < self.p_pos) 
        else:
            # Tuple belongs to a non-targeted group and should not be affected
            return 0
    
### Helper class for results of applying transformation ###
class OpOutcome:
    
    def __init__(self, freq, target_size, affected):
        self.freq     = freq         # Frequency of targeted tuples (depends on operation/lambda)
        self.n        = target_size  # Amount of tuples in target group after operation
        self.affected = affected     # Affected tuples (= nr. of times lambda was positive)
        
    def __add__(self, other):
        if isinstance(other, OpOutcome):
            return OpOutcome(self.freq + other.freq, self.n + other.n, self.affected + other.affected)
        else:
            raise ValueError(f"Can only add two OpOutcomes")
    
    def __truediv__(self, n):
        if isinstance(n, int) or isinstance(n, float):
            return OpOutcome(self.freq / n, self.n / n, self.affected / n)
        else:
            raise ValueError(f"Cannot divide OpOutcome by {type(n)}")
            
    def __str__(self):
        return f"Freq: {self.freq:.4f}\nTarget group size: {self.n}\nAffected tuples: {self.affected}"
    

### SINGLE / COUNTING FUNCTIONS ###
# These all return the new label Y, and the outcome of lambda (0 or 1, resp. - or +)
    
def f_mod(l, a, g):
    """
        Outcome (label) after modification of given tuple
        
        Parameters:
            t     : tuple to be modified
            l : distribution (lambda) that decides execution of operation
    """
    outcome = l.get_outcome(a, g)
    if outcome == 1:
        # Modify label
        return 1 - g, outcome
    elif outcome == 0:
        # Retain original label
        return g, outcome
    
    

def f_add(l, a, g):
    """
        Applies the addition (positive weighting) operation to the data.
    """
    outcome = l.get_outcome(a, g)
    if outcome == 1:
        # Generate new weight ('duplicate') and return
        return g * 2, outcome # TEMP: just duplicate once, TODO: make this actually calculate a proper weight (lambda)
    elif outcome == 0:
        # Retain original weight
        return g, outcome
    
    
def f_rem(l, a, g):
    """
        Applies the removal operation to the data.
    """
    outcome = l.get_outcome(a, g)
    if outcome == 1:
        # Remove tuple - returns 'd'
        return 'd', outcome
    elif outcome == 0:
        # Retain tuple, count using original label
        return g, outcome
    


### APPLICATIONS TO ENTIRE DATASET ###

def count_f_mod(data, l):
    """
        Returns frequency of targeted group after applying modification to the entire dataset.
        Also returns the amount of tuples in the target group (priv. / unpriv.) after application.
    """
    c = 0 # Total sum of labels
    n_tuples = 0 # Total size of target group (privileged or unprivileged) after application
    affected = 0 # Total number of positive outcomes for lambda
    for t in data:
        a = t[A_IX]
        g = t[G_IX]
        if l.target_a == a:
            y, outc = f_mod(l, a, g)
            n_tuples += 1
            affected += outc
            c += y
    c /= n_tuples
    return OpOutcome(c, n_tuples, affected)

def count_f_rem(data, l):
    c = 0 # Total sum of labels
    n_tuples = 0 # Total size of target group (privileged or unprivileged) after application
    affected = 0
    for t in data:
        a = t[A_IX]
        g = t[G_IX]
        if l.target_a == a:
            y, outc = f_rem(l, a, g)
            affected += outc
            if y == 'd': continue # Skip 'deleted' tuples
            n_tuples += 1
            c += y
    c /= n_tuples
    return OpOutcome(c, n_tuples, affected)


def avg_count_f_op(op, data, l, n = 100):
    """
        Returns the average results of n operations.
    """
    outc = OpOutcome(0, 0, 0)
    for _ in range(n):
        outc += op(data, l)
    return outc / n


### CALCULATING EXPECTED PROBABILITIES ###

def f_mod_exp(dataset, l):
    """
        Returns expected probability for the target group (per l) after modification operation.
    """
    # TODO: Allow for both scenarios (suppression and boosting)
    old_U  = dataset.select((A_IX, 0))
    old_UN = dataset.select_UN(A_IX, G_IX)
    old_UP = dataset.select_UP(A_IX, G_IX)
    
    exp_new_UN = len(old_UN) + (l.p_pos * len(old_UP))
    #print(exp_new_UN, "/", len(old_U))
    return exp_new_UN / len(old_U)
    
def f_rem_exp(dataset, l):
    
    unpriv = dataset.select((A_IX, 0))
    old_UP = dataset.select_UP(A_IX, G_IX)
    
    exp_UP = (1 - l.p_pos) * len(old_UP)
    exp_removed = len(old_UP) - exp_UP
    return exp_UP / (len(unpriv) - exp_removed)