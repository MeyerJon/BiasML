"""
    Some simple implementations for visualising the transformations of the data.
"""
import random
import matplotlib.pyplot as plt
import numpy as np

def generate_ds(n=10, a_dist=None):
    """
        Generates n datapoints,
        according to simple X,A,G model where P(G|A) = P(G)
    """
    a_dist = a_dist or (0.5, 1)
    g_independent = True
    data = list()
    for _ in range(n):
        # A = sex, 50/50 male/female
        a = min(1, max(0, random.normalvariate(*a_dist)))
        a = 'M' if a < 0.5 else 'F'
        
        # X is dependent on A
        x = round(random.normalvariate(0.9, 0.4))
        if a == "M":
            x = round(random.normalvariate(1.2, 0.5))
        x = min(2, max(0, x))
        
        g = None
        if g_independent:
            # G is independent of A (and X in this case)
            g = random.randint(0, 1)
        else:
            # G is dependent on X
            g = round(random.normalvariate(0.5, 0.3))
            if x == 2:
                g = round(random.normalvariate(0.7, 0.1))
            g = min(1, max(0, g))
            
        pt = [x, a, g]
        data.append(pt)
        
    return data
    
def matches(point, attr_ixs, vals):
    """
        True if point matches values for all given attributes
    """
    for i, a_ix in enumerate(attr_ixs):
        if point[a_ix] != vals[i]:
            return False
    return True
    
def P(data, attr_ixs, vals):
    """
        Gets probability (frequency) of data[attr_ix] = val (conjuctive)
    """
    p = 0
    for pt in data:
        p += int(matches(pt, attr_ixs, vals))
    return p / len(data)
    
def select(data, attr_ixs, vals):
    """
        Returns matching datapoints
    """
    for i, a_ix in enumerate(attr_ixs):
        data = list(filter(lambda pt : pt[a_ix] == vals[i], data))
    return data
    
def remove_coinflip(data, attr_ix, vals, w_pos = 0.5):
    """
        Use weighted coinflip to remove rows
        w_pos indicates how often coin lands on 'positive' outcome, i.e. row is removed
    """
    new_data = list()
    for pt in data:
        # Check if point matches selection
        if matches(pt, attr_ix, vals):
            # Also check if point is removed using coinflip
            flip = random.random()
            if flip < w_pos:
                continue # Continue without adding point
        new_data.append(pt)
    return new_data
        
def reassign_coinflip(data, attr_ix, vals, w_pos = 0.5):
    """
        Use weighted coinflip to flip label of selected rows
        w_pos indicates how often coin lands on 'positive' outcome, i.e. label is flipped
        Assumes label is last item in point, and binary label (0/1)
    """
    new_data = list()
    for pt in data:
        # Check if point matches selection
        l = pt[-1]
        if matches(pt, attr_ix, vals):
            # Also check if label is switched
            flip = random.random()
            if flip < w_pos:
                l = int(not bool(l)) # Flip the label
        new_data.append([v for v in pt[:-1]] + [l])
    return new_data
        
def duplicate_coinflip(data, attr_ix, vals, w_pos = 0.5):
    """
        Uses weighted coinflip to duplicate selected rows
        positive outcome = row is duplicated
    """
    new_data = list()
    for pt in data:
        # Check if point matches selection
        if matches(pt, attr_ix, vals):
            # Also check if point is duplicated using coinflip
            flip = random.random()
            if flip < w_pos:
                new_data.append(pt) # Add duplicate
        new_data.append(pt)
    return new_data

def reweight_discrete(data, attr_ix, vals, weights=None):
    """
        Emulates 'reweighting' of data by duplicating rows a random number of times (>0)
        weights should be list of [(dupes, probability)] where the probabilities add up to 1
    """
    if weights is None:
        weights = [(0, 0.5), (1, 0.5)] # Default, like coinflip
    # Format weights for use with Python's choices
    w_outcomes = [p[0] for p in weights]
    w_probs = [p[1] for p in weights]
    if abs(1 - sum(w_probs)) > 1e-9:
        raise Exception("Sum of weights must be equal to 1")
    new_data = list()
    for pt in data:
        # Check if point matches selection
        if matches(pt, attr_ix, vals):
            # Check how many dupes to add
            dupes = random.choices(population=w_outcomes, weights=w_probs, k=1)
            for _ in range(dupes[0]):
                new_data.append(pt) # Add duplicate
        new_data.append(pt)
    return new_data

def reweight_normal(data, attr_ix, vals, weights=None):
    """
        Emulates reweighting as above, but using normal distribution to decide the amount of duplicates
    """
    weights = weights or (0.5, 1)
    outcome = lambda x : int(min(max(0, round(x)), 1+weights[0]*5)) # Clamp number of dupes for single row
    new_data = list()
    for pt in data:
        # Check if point matches selection
        if matches(pt, attr_ix, vals):
            # Check how many dupes to add
            dupes = [outcome(random.normalvariate(*weights))]
            for _ in range(dupes[0]):
                new_data.append(pt) # Add duplicate
        new_data.append(pt)
    return new_data

def generate_outcomes(data, transformation, outcome_func,
                attr_ixs, vals, weights, n=20):
    """
        Runs an experiment/transformation n times and returns the outcomes
    """
    outcomes = list()
    for _ in range(n):
        new_data = transformation(data, attr_ixs, vals, w_pos=weights)
        outcomes.append(outcome_func(new_data))
    return outcomes

def avg(l):
    return sum(l)/len(l)


def weight_outcome_plot(data, outcome_func):
    X_ix = 0
    A_ix = 1
    G_ix = 2
    wgts = np.arange(0.0, 1.0, 0.05)

    # Experiment: plot influence of weight on removal
    print("Removal (1/3)...")
    outcomes = list()
    for w in wgts:
        o_rm = generate_outcomes(data, remove_coinflip, outcome_func, [A_ix, G_ix], ["M", 1], weights=w, n=2)
        outcomes.append(avg(o_rm))
    plt.plot(wgts, outcomes, marker='o', label='Remove')

    # Experiment: plot influence of weight on reassignment
    print("Reassignment (2/3)...")
    outcomes = list()
    for w in wgts:
        o_ra = generate_outcomes(data, reassign_coinflip, outcome_func, [A_ix, G_ix], ["M", 1], weights=w, n=2)
        outcomes.append(avg(o_ra))
    plt.plot(wgts, outcomes, marker='o', label='Reassign')

    # Experiment: plot influence of weight on duplication
    print("Duplication (3/3)...")
    outcomes = list()
    for w in wgts:
        o_d = generate_outcomes(data, duplicate_coinflip, outcome_func, [A_ix, G_ix], ["M", 0], weights=w, n=2)
        outcomes.append(avg(o_d))
    plt.plot(wgts, outcomes, marker='o', label='Duplicate')
    
    plt.xlabel("P(C=+)")
    plt.ylabel("P(Y=1 | A='M')")
    plt.legend()
    plt.savefig("weight_outcome_plot.png")
    plt.show()


if __name__ == "__main__":
    
    data = generate_ds(50000)
        
    X_ix = 0
    A_ix = 1
    G_ix = 2

    p_male = P(data, [A_ix], ['M'])
    p_female = P(data, [A_ix], ['F'])
    print(p_male, " + ", p_female, " = ", p_female + p_male)
    
    # Experiment: induce bias against males
    def male_bias(ds):
        return P(select(ds, [A_ix], ["M"]), [G_ix], [1])
    og_p_male_pos = male_bias(data)
    print("Original P(G=1 | A='M'):", round(og_p_male_pos, 4))
    print("Original P(G=1 | A='F'):", round(P(select(data, [A_ix], ["F"]), [G_ix], [1]), 4))
    
    #weight_outcome_plot(data, male_bias)

    #new_data = reweight_discrete(data, [A_ix, G_ix], ['M', 0], [(0, 0.5), (1, 0.3), (2, 0.2)])
    #bias = male_bias(new_data)
    #print("Reweighting:", bias)

    wts = np.arange(0.0, 10.0, 0.5)
    biases = list()
    for w in wts:
        new_data = reweight_normal(data, [A_ix, G_ix], ['M', 0], (w, 1))
        bias = male_bias(new_data)
        biases.append(bias)
    plt.plot(wts, biases, marker='o')
    plt.show()

