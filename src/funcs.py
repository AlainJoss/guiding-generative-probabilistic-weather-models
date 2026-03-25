
import torch
import numpy as np

def avg_over_mask(mask, state):
    avg = np.sum(mask * state) / np.sum(mask)
    return avg

def get_guidance_trajectory(drift, init_avg):
    return [init_avg + d * np.abs(init_avg) for d in drift]