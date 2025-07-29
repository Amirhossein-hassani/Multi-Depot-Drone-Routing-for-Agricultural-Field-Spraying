# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:44:02 2025

@author: ASUS
"""

import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import numpy as np
import time
# Start the timer
start_time = time.time()


def compute_N_max(Q, time_available, theta, k_max, farm_areas):
    # Step 1: Calculate MH
    MH = Q * (time_available - theta * (k_max - 1))

    # Step 2: Sort farm areas in ascending order
    sorted_farms = sorted(farm_areas)

    # Step 3: Identify the minimum number of farms that meets MH
    total_area = 0
    N_max = 0
    for area in sorted_farms:
        total_area += area
        N_max += 1
        if total_area >= MH:
            break

    # Step 4: Return N_max
    return N_max




# Define sets and parameters
D = 3  # Number of depots 
N = 6  # Number of farms 
T = 10  # Number of days in the planning horizon 
tau = 4  # Battery duration before UAV needs to return to a depot 
theta = 1  # Battery recharge time at the depot 
time_available = 15  # Total available working time per day 
k_max = math.floor(time_available / (tau + theta)) + 1  # Maximum number of tours per day 

I = range(1, N+1)  # Farms
D_set = range(1, D+1)  # Depots
T_set = range(1, T+1)  # Days
K = range(1, k_max+1)  # Tours

Q = 2 # UAV efficiency (hectares sprayed per hour)
v = 10  # UAV speed (distance per unit time)
e = 2  # Minimum spraying amount required per visit (hectares)
A = {i:12 for i in I}  # Total area of each farm 


d_di = {(1, 1): 2, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 5,
        (2, 1): 4, (2, 2): 3, (2, 3): 2, (2, 4): 1, (2, 5): 2, (2, 6): 7, 
        (3, 1): 1, (3, 2): 2, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 1}  # Distance from depots to farms

d_id = {(f, d): d_di[d, f] for d, f in d_di}  # Same values but reversed

d_ij = {(1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 2,  
        (2, 1): 1, (2, 3): 1, (2, 4): 2, (2, 5): 3, (2, 6): 3, 
        (3, 1): 2, (3, 2): 1, (3, 4): 1, (3, 5): 2, (3, 6): 4, 
        (4, 1): 3, (4, 2): 2, (4, 3): 1, (4, 5): 1, (4, 6): 5, 
        (5, 1): 4, (5, 2): 3, (5, 3): 2, (5, 4): 1, (5, 6): 6,
        (6, 1): 2, (6, 2): 3, (6, 3): 4, (6, 4): 5, (6, 5): 6}   # Distance between farms

M = 1000  # Big number 

N_max = compute_N_max(Q, time_available, theta, k_max, A)




'''
d_di = {(1, 1): 5, (1, 2): 3, (1, 3): 3, (1, 4): 5, (1, 5): 7, (1, 6): 5, (1, 7): 5, (1, 8): 7,
        (2, 1): 2, (2, 2): 4, (2, 3): 6, (2, 4): 8, (2, 5): 2, (2, 6): 4, (2, 7): 6, (2, 8): 8,
        (3, 1): 8, (3, 2): 6, (3, 3): 4, (3, 4): 2, (3, 5): 8, (3, 6): 6, (3, 7): 4, (3, 8): 2,
        (4, 1): 7, (4, 2): 5, (4, 3): 5, (4, 4): 7, (4, 5): 5, (4, 6): 3, (4, 7): 3, (4, 8): 5}  # Distance from depots to farms 

d_id = {(1, 1): 5, (2, 1): 3, (3, 1): 3, (4, 1): 5, (5, 1): 7, (6, 1): 5, (7, 1): 5, (8, 1): 7,
        (1, 2): 2, (2, 2): 4, (3, 2): 6, (4, 2): 8, (5, 2): 2, (6, 2): 4, (7, 2): 6, (8, 2): 8,
        (1, 3): 8, (2, 3): 6, (3, 3): 4, (4, 3): 2, (5, 3): 8, (6, 3): 6, (7, 3): 4, (8, 3): 2,
        (1, 4): 7, (2, 4): 5, (3, 4): 5, (4, 4): 7, (5, 4): 5, (6, 4): 3, (7, 4): 3, (8, 4): 5}  # Distance from farms to depots 

d_ij = {(1, 2): 2, (1, 3): 4, (1, 4): 6, (1, 5): 2, (1, 6): 4, (1, 7): 6, (1, 8):8,
        (2, 1): 2, (2, 3): 2, (2, 4): 4, (2, 5): 4, (2, 6): 2, (2, 7): 4, (2, 8): 6,
        (3, 1): 4, (3, 2): 2, (3, 4): 2, (3, 5): 6, (3, 6): 4, (3, 7): 2, (3, 8): 4,
        (4, 1): 6, (4, 2): 4, (4, 3): 2, (4, 5): 8, (4, 6): 6, (4, 7): 4, (4, 8): 2,
        (5, 1): 2, (5, 2): 4, (5, 3): 6, (5, 4): 8, (5, 6): 2, (5, 7): 4, (5, 8): 6,
        (6, 1): 4, (6, 2): 2, (6, 3): 4, (6, 4): 6, (6, 5): 2, (6, 7): 2, (6, 8): 4,
        (7, 1): 6, (7, 2): 4, (7, 3): 2, (7, 4): 4, (7, 5): 4, (7, 6): 2, (7, 8): 2,
        (8, 1): 8, (8, 2): 6, (8, 3): 4, (8, 4): 2, (8, 5): 6, (8, 6): 4, (8, 7): 2}  # Distance between farms
'''






# Define decision variables
S = LpVariable.dicts("S", [(i, t, k) for i in I for t in T_set for k in K], lowBound=0, cat='Continuous')
x_ijt = LpVariable.dicts("x_ijt", [(i, j, t, k) for i in I for j in I if i != j for t in T_set for k in K], cat='Binary')
x_dit = LpVariable.dicts("x_dit", [(d, i, t, k) for d in D_set for i in I for t in T_set for k in K], cat='Binary')
x_idt = LpVariable.dicts("x_idt", [(i, d, t, k) for i in I for d in D_set for t in T_set for k in K], cat='Binary')
# Define the decision variable u_itk for subtour elimination as an integer
u_itk = LpVariable.dicts("u_itk", [(i, t, k) for i in I for t in T_set for k in K], lowBound=0, cat='Integer')
# Define binary decision variable y_it (1 if farm i is visited on day t, 0 otherwise)
y_it = LpVariable.dicts("y_it", [(i, t) for i in I for t in T_set], cat='Binary')


# Define the problem
model = LpProblem("UAV_Spraying_Optimization", LpMinimize)


# Objective function
model += lpSum(
    (x_ijt[i, j, t, k] * (d_ij[i, j] * (1/v)) +  # Time between farms
     x_dit[d, i, t, k] * (d_di[d, i] * (1/v)) +  # Time from depot to farm
     x_idt[i, d, t, k] * (d_id[i, d] * (1/v)) +  # Time from farm to depot
     (S[i, t, k] * (1/Q))  # Time spent spraying
     for i in I for j in I if i != j for d in D_set for t in T_set for k in K))






# Add constraint: UAV tour continuity
for t in T_set:
    for k in range(2, k_max+1):
        model += (lpSum(x_dit[d, i, t, k] for d in D_set for i in I) <=
                  lpSum(x_dit[d, i, t, k-1] for d in D_set for i in I))
        
        
# Add constraint: UAV trip continuity from depots to farms for k >= 2
for t in range(2, T+1):  # Starting from t=2
     # Starting from k=2
        for d in D_set:
            model += (lpSum(x_dit[d, i, t, 1] for i in I) <=  # Sum of trips for day t and tour 1
                      lpSum(x_dit[d, i, t-1, 1] for i in I))  # Sum of trips for day t-1 and tour 1

        
        
# Add constraint: Flow conservation at each farm i
for t in T_set:  # For each day
    for k in K:  # For each tour
        for i in I:  # For each farm
            model += (lpSum(x_dit[d, i, t, k] for d in D_set) +  # Trips arriving at farm i from depots
                      lpSum(x_ijt[j, i, t, k] for j in I if j != i)  # Trips arriving at farm i from other farms
                      ==
                      lpSum(x_ijt[i, j, t, k] for j in I if j != i) +  # Trips leaving farm i to other farms
                      lpSum(x_idt[i, d, t, k] for d in D_set))  # Trips leaving farm i to depots

# Add constraint: Depot flow balance for each tour k and day t
for t in T_set:  # For each day
    for k in K:  # For each tour
        model += (lpSum(x_dit[d, i, t, k] for d in D_set for i in I) ==
                  lpSum(x_idt[i, d, t, k] for d in D_set for i in I))

# Add constraint: At most one UAV trip can be initiated from depots per tour and day
for t in T_set:  # For each day
    for k in K:  # For each tour
        model += (lpSum(x_dit[d, i, t, k] for d in D_set for i in I) <= 1)

# Add constraint: Farm-to-farm transitions depend on depot departures
for t in T_set:  # For each day
    for k in K:  # For each tour
        model += (lpSum(x_ijt[j, i, t, k] for i in I for j in I if j != i) <=
                  M * lpSum(x_dit[d, i, t, k] for d in D_set for i in I))

# Add constraint: UAV total operational time per tour does not exceed battery duration τ
for t in T_set:  # For each day
    for k in K:  # For each tour
        model += (
            lpSum(S[i, t, k] * (1/Q) for i in I) + 
            lpSum(x_ijt[i, j, t, k] * (d_ij[i, j] / v) for i in I for j in I if i != j) +
            lpSum(x_dit[d, i, t, k] * (d_di[d, i] / v) for d in D_set for i in I) +
            lpSum(x_idt[i, d, t, k] * (d_id[i, d] / v) for d in D_set for i in I)
            <= tau
        )

# Add constraint: Total UAV operational time per day does not exceed available working time
for t in T_set:  # For each day
    model += (
        lpSum(
            (lpSum(S[i, t, k] * (1/Q) for i in I) +
             lpSum(x_ijt[i, j, t, k] * (d_ij[i, j] / v) for i in I for j in I if i != j) +
             lpSum(x_dit[d, i, t, k] * (d_di[d, i] / v) for d in D_set for i in I) +
             lpSum(x_idt[i, d, t, k] * (d_id[i, d] / v) for d in D_set for i in I))
            for k in K
        )
        <= time_available - theta * (lpSum(x_dit[d, i, t, k] for d in D_set for i in I for k in K) - 1)
    )


# Add constraint: Total spraying at farm i on day t equals its required area Ai if visited
for i in I:
    for t in T_set:
        model += (
            lpSum(S[i, t, k] for k in K) == A[i] * y_it[i, t]
        )

# Add constraint: Spraying amount is bounded by minimum required e and maximum A_i when visited
for i in I:
    for k in K:
        for t in T_set:
            visit_indicator = (lpSum(x_dit[d, i, t, k] for d in D_set) +
                               lpSum(x_ijt[j, i, t, k] for j in I if j != i))

            model += (e * visit_indicator <= S[i, t, k])
            model += (S[i, t, k] <= A[i] * visit_indicator)

# Add the subtour elimination constraint using the u_itk decision variable for each j
for i in I:
    for j in I:
        if i != j:
            for k in K:
                for t in T_set:
                    model += (u_itk[i, t, k] - u_itk[j, t, k] + N_max * x_ijt[i, j, t, k] <= N_max - 1)

# Add the constraint for u_itk being bounded by the sum of x_ijt variables
for i in I:
    for k in K:
        for t in T_set:
            model += (u_itk[i, t, k] <= 1 + lpSum(x_ijt[j, j_prime, t, k] for j in I for j_prime in I if j != j_prime))


# Add the constraint for u_itk being bounded by the sum of x_dit and x_ijt variables
for i in I:
    for k in K:
        for t in T_set:
            model += (lpSum(x_dit[d, i, t, k] for d in D_set) + lpSum(x_ijt[j, i, t, k] for j in I if j != i) <= u_itk[i, t, k])
            model += (u_itk[i, t, k] <= N_max * (lpSum(x_dit[d, i, t, k] for d in D_set) + lpSum(x_ijt[j, i, t, k] for j in I if j != i)))


# Add the constraint to ensure each farm is visited exactly once
for i in I:
    model += (lpSum(y_it[i, t] for t in T_set) == 1)















# Ensure the drone starts from Depot 1 at k=1, t=1
model += lpSum(x_dit[1, i, 1, 1] for i in I) == 1  # Must start from Depot 1

# Ensure the drone does NOT start from any other depot at k=1, t=1
model += lpSum(x_dit[d, i, 1, 1] for d in D_set if d != 1 for i in I) == 0  







# If the drone ends in a depot, it must start from the same depot in the next tour
for d in D_set:
    for t in T_set:
        for k in range(2, k_max + 1):  # Only for tours k >= 2
            model += (
                lpSum(x_idt[i, d, t, k-1] for i in I) >= 
                lpSum(x_dit[d, i, t, k] for i in I),
                f"Depot_Continuation_{d}_{t}_{k}"
            )



# If the drone ends at depot d on day t-1, it must start from depot d on day t for tour k
for d in D_set:
    for t in range(2, T+1):  # Starting from day 2
            model += (
                lpSum(x_idt[i, d, t-1, k_max] for i in I) >= 
                lpSum(x_dit[d, i, t, 1] for i in I),
                f"Depot_Continuation_day_{d}_{t}_{k}"
            )



####################################################################################################





# Solve the model
model.solve()

####################################################################################################




# Print the objective value
print("Optimal time:", model.objective.value())

# Print detailed information about drone movements to farms and depots
for t in T_set:
    for k in K:
        for i in I:
                
            # Check the movement from depot d to farm i
            for d in D_set:
                if x_dit[d, i, t, k].varValue == 1:
                    print(f"On day {t}, the drone moves from depot {d} to farm {i} during tour {k}.")
                
            # Check the movement from farm j to farm i
            for j in I:
                if j != i and x_ijt[j, i, t, k].varValue == 1:
                    print(f"On day {t}, the drone moves from farm {j} to farm {i} during tour {k}.")
                
            # Check the movement from farm i to depot d
            for d in D_set:
                if x_idt[i, d, t, k].varValue == 1:
                    print(f"On day {t}, the drone moves from farm {i} to depot {d} during tour {k}.")


end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Execution time: {runtime:.4f} seconds")