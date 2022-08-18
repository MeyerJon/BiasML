from bias.data import Dataset, DS_IXS
from bias.formal import Lambda
import bias.transforms as tf
import numpy as np
import math


def prep_synth_datasets(n=10000, gen_dists=None):
    """
        Returns synthetic datasets:
            - Original (fair) data
            - Transformed to introduce bias using addition, modification, removal
            - Reweighed to remove bias again (in the same order)
        Returns a dict with keys {"og", "bias": {"add", "mod", "rem"}, "fair": {"add", "mod", "rem"}}
    """
    # Creating fair dataset
    ds = Dataset(n, gen_dists=gen_dists)
    
    # Creating biased transformations of fair dataset (bias for all these is approx. identical)
    
    # Addition
    l_dist = Lambda(target_a = 1, target_g = 1, p_pos = 1.0) # Targeting privileged positive
    add_bias = tf.reweighing(ds.data, l_dist, gamma=2)
    l_dist = Lambda(target_a = 0, target_g = 0, p_pos = 1.0) # Targeting unprivileged negative
    add_bias = tf.reweighing(add_bias, l_dist, gamma=2)
    add_bias_ds = Dataset.from_array(add_bias)
    
    # Modification
    ppos = 1.0 / 3.0
    l_dist = Lambda(target_a = 1, target_g = 0, p_pos = ppos) # Targeting privileged negative
    modify_bias = tf.modify(ds.data, l_dist)
    l_dist = Lambda(target_a = 0, target_g = 1, p_pos = ppos) # Targeting unprivileged positive
    modify_bias = tf.modify(modify_bias, l_dist)
    modify_bias_ds = Dataset.from_array(modify_bias)
    
    # Removal
    l_dist = Lambda(target_a = 0, target_g = 1, p_pos = 0.5) # Targeting privileged positive
    rem_bias = tf.deletion(ds.data, l_dist)
    l_dist = Lambda(target_a = 1, target_g = 0, p_pos = 0.5) # Targeting unprivileged negative
    rem_bias = tf.deletion(rem_bias, l_dist)
    rem_bias_ds = Dataset.from_array(rem_bias)
    
    # Fixing biased datasets using reweighing technique
    add_rw = tf.Reweigher(add_bias_ds)
    add_rw.reweigh()
    add_fair_ds = Dataset.from_array(add_rw.fair_data)
    
    mod_rw = tf.Reweigher(modify_bias_ds)
    mod_rw.reweigh()
    mod_fair_ds = Dataset.from_array(mod_rw.fair_data)
    
    rem_rw = tf.Reweigher(rem_bias_ds)
    rem_rw.reweigh()
    rem_fair_ds = Dataset.from_array(rem_rw.fair_data)
    
    return {
        "og": ds,
        "bias": {
            "add": add_bias_ds,
            "mod": modify_bias_ds,
            "rem": rem_bias_ds
        },
        "fair": {
            "add": add_fair_ds,
            "mod": mod_fair_ds,
            "rem": rem_fair_ds
        }
    }
    

if __name__ == "__main__":
    
    ds = Dataset(n=10)
    print(ds.data)
