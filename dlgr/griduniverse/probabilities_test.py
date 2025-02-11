import math

def normalize(arr):
    total = sum(arr)
    if total == 0:
        return [0.5, 0.5]
    return [arr[0] / total, arr[1] / total]

alpha = 0.8
player_probabilities = [0.5, 0.5]  
iterations = 0

hare1_distances = [5, 4, 3, 2, 2]  
hare2_distances = [10, 11, 12, 13, 14]  
stag_distances = [6, 5, 4, 5, 4] 

for i in range(1, len(hare1_distances)):

    delta_hare1 = hare1_distances[i] - hare1_distances[i-1]
    delta_hare2 = hare2_distances[i] - hare2_distances[i-1]
    delta_stag = stag_distances[i] - stag_distances[i-1]

    reward_stag = -delta_stag 
    reward_no_stag = -min(delta_hare1, delta_hare2) 

    exp_stag = math.exp(alpha * reward_stag)
    exp_nostag = math.exp(alpha * reward_no_stag)

    lhood_denom = exp_stag + exp_nostag
    lhood_stag = exp_stag / lhood_denom
    lhood_nostag = exp_nostag / lhood_denom

    prior_stag = player_probabilities[0] * lhood_stag
    prior_nostag = player_probabilities[1] * lhood_nostag

    player_probabilities = normalize([prior_stag, prior_nostag])

    iterations += 1

    print(f"Iteration {iterations} - Probabilities: {player_probabilities}")
