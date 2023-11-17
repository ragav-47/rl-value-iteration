# <p align="center"> VALUE ITERATION ALGORITHM:

## AIM:
The aim of this experiment is to apply the Value Iteration algorithm to find the optimal policy for the FrozenLake-v1 Markov Decision Process (MDP) and evaluate its performance.

## PROBLEM STATEMENT:
The problem involves navigating through the FrozenLake environment to reach the goal state while maximizing the cumulative reward. The environment presents challenges with uncertain transitions and rewards.

## VALUE ITERATION ALGORITHM:

**Step 1: Initialization**
- Initialize the state-value function V as an array of zeros.

**Step 2: Main Loop**
- While True, repeat the following steps:
  
  **Step 3: Action-Value Function Update**
  - Initialize the action-value function Q as an array of zeros.
  - For each state s:
    - For each action a:
      - For each transition (prob, next_state, reward, done) in P[s][a], update Q[s][a] using the Bellman equation.

  **Step 4: Convergence Check**
  - Check if the maximum absolute difference between the previous V and the new V (computed from Q) is less than a predefined threshold (theta). If it is, break out of the loop.

  **Step 5: Update Value Function**
  - Update the value function V with the maximum action-value from Q, i.e., V = max(Q, axis=1).

**Step 6: Policy Extraction**
- Compute the policy pi based on the action-value function Q, where pi(s) selects the action with the highest Q-value for each state.

## VALUE ITERATION FUNCTION:

```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    # Initialize the value function V as an array of zeros
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        # Initialize the action-value function Q as an array of zeros
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    # Update the action-value function Q using the Bellman equation
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Check if the maximum difference between Old V and new V is less than theta.
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        # Update the value function V with the maximum action-value from Q
        V = np.max(Q, axis=1)

    # Compute the policy pi based on the action-value function Q
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi
```
## OUTPUT:
![image](https://github.com/ragav-47/rl-value-iteration/assets/75235488/4a31418b-f1a5-4ff0-b9fe-ae9dcf24aaa3)

## RESULT:
The results of the experiment demonstrate the effectiveness of the Value Iteration algorithm in finding the optimal policy for the FrozenLake-v1 MDP. The optimal policy, value function, and success rate are presented, showing that the algorithm successfully solves the problem and achieves the desired goal.

