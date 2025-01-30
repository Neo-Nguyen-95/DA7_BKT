import numpy as np

class HMM:
    def __init__(self, pi, A, B, O_index):
        '''
        Parameter:
            π: initial probability of hidden states
            A: transition probability matrix
            B: emission probability matrix
            O_index: index sequence of observation
            psi: index of hidden state, taken infor from pi
        '''
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
        self.O_index = np.array(O_index)
        self.psi = list(np.array(range(len(pi))).reshape(-1, 1))
        
    def alpha(self, t):
        '''alpha = [α(U)
                    α(L)]
        '''
        
        b = self.B[:, self.O_index[t]].reshape(-1, 1)
    
        if t == 0:        
            return self.pi * b
        else:
            return np.dot(self.A.T, self.alpha(t-1)) * b
        
    def beta(self, t):
        '''beta = [β(U)
                   β(L)]
        '''
        
        T = len(self.O_index)-1
        
        if t == T:
            return np.ones(len(self.pi)).reshape(-1, 1)
        else:
            b_time_beta = self.B[:, self.O_index[t+1]].reshape(-1, 1) * self.beta(t+1)
            return np.dot(self.A, b_time_beta)
        
    def P_O_from_alpha(self):
        return self.alpha(len(self.O_index)-1).sum()
    
    def P_O_from_beta(self):
        return (self.beta(0) * self.pi * self.B[:, self.O_index[0]].reshape(-1, 1)).sum()
        
    def viterbi(self):

        num_state = len(self.psi)
        
        for t, obs in enumerate(self.O_index):
            if t == 0:  # Initial observation emission probability
                delta = self.pi * self.B[:, obs].reshape(-1, 1)
        
            else:
                # Take N^2 equation
                probability_observation = delta * self.A * self.B[:, obs]
                
                for current_index in range(num_state):
                    # In each group of calculation, take the max out of N results
                    # delta store max value of current_index branch
                    delta[current_index] = probability_observation[:, current_index].max()
                    
                    
                    for previous_index in range(num_state):
                        # In each group of calculation, sorted by current_index
                        # Take argmax at current_index as psi
                        # backtracking with previous_index to find previous sequence
                        if probability_observation[:, current_index].argmax() == previous_index:
                            self.psi[current_index] = np.append(self.psi[previous_index][:t], current_index)
                            
                            # Note for [:t]:
                            # only take into account up to time t of the sequence
                            # avoid update the updated psi if 2 route start from a state
        
        for state in range(num_state):
            if delta.argmax() == state:
                sequence = self.psi[state]
        
        return (delta.max(), sequence)
    
    def zeta(self, t):
        '''            U         L
        zeta = U | ζ(U, U) | ζ(U, L) |
               L | ζ(L, U) | ζ(L, L) |
        
        '''
        
        return (self.alpha(t) * self.A 
                * self.B[:, self.O_index[t+1]] * self.beta(t+1).reshape(1, -1)
                / self.P_O_from_alpha()
                )
    
    def gamma(self, t):
        '''
        gamma = γ(U)
                γ(L)
        '''
        
        return self.alpha(t) * self.beta(t) / np.sum(self.alpha(t) * self.beta(t))
        

