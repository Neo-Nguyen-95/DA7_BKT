import numpy as np
import pandas as pd

#%% CLASS

class BKTGenerator:
    def __init__(self, pi, A, B, obs_len):
        '''
        Parameter:
            
            π: initial probability of hidden states
            π =  state_1 [ 0.7 ] 
                 state_2 [ 0.6 ]
                 
            A: transition probability matrix
                            state_1    state_2
            A = state_1 [     0.3       0.7      ]
                state_2 [     0.4       0.6      ]
            
            B: emission probability matrix
                            obs_1       obs_2
            B = state_1 [     0.2         0.8   ]
                state_2 [     0.1         0.9   ]
            
            hidden_state: set of hidden states
            hidden_state = [state_1, state_2, ..., state_N]
            
            obs_len: length of generated observations
            obs_len = T
        
        '''
        
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
        self.hidden_states_set = ['Unlearned', 'Learned']
        self.unique_obs_set = ['Incorrect', 'Correct']
        self.obs_len = obs_len
        
        self.state_index_set = np.array([0, 1])
        self.obs_index_set = np.array([0, 1])
        
        self.state_mapping = {}
        for key, value in enumerate(self.hidden_states_set):
             self.state_mapping[key] = value
             
        self.obs_mapping = {}
        for key, value in enumerate(self.unique_obs_set):
             self.obs_mapping[key] = value
    
    def generate_1_sequence(self):
        
        # Initial state with initial prob
        state_index = np.random.choice(self.state_index_set, p=self.pi.flatten())
        state_index_sequence = [state_index]
        
        # Initial observation from initial state
        obs_index = np.random.choice(self.obs_index_set, p=self.B[state_index])
        obs_index_sequence = [obs_index]
        
        for _ in range(self.obs_len - 1):
            # State at t+1
            state_index = np.random.choice(self.state_index_set, p=self.A[state_index])
            state_index_sequence.append(state_index)
            
            # Observation at t+1
            obs_index = np.random.choice(self.obs_index_set, p=self.B[state_index])
            obs_index_sequence.append(obs_index)
    
        
        state_sequence = pd.Series(state_index_sequence).map(self.state_mapping)
        obs_sequence = pd.Series(obs_index_sequence).map(self.obs_mapping)
        
        return (state_index_sequence, 
                obs_index_sequence,
                state_sequence,
                obs_sequence)
    
    def generated_multi_sequences(self, num_student):
        df = pd.DataFrame({
            'student_ID': [],
            'state_index': [],
            'score': []
            })
        for i in range(num_student):
            (state_index_sequence,
             obs_index_sequence,
             state_sequence,
             obs_sequence) = self.generate_1_sequence()
            
            df_temp = pd.DataFrame({
                'student_ID': np.ones(self.obs_len) * i,
                'state_index': state_index_sequence,
                'score': obs_index_sequence
                })
            
            df = pd.concat([df, df_temp], axis='rows')
            
        return df.reset_index()
            
            
        
#%% CASE STUDY 1: BKT sequence
pi = [[1], 
      [0]]
A = [[0.75, 0.25],
     [0, 1]]
B = [[0.7, 0.3],
     [0.15, 0.85]]
obs_len = 15

bktg = BKTGenerator(pi, A, B, obs_len) 

# GENERATE SEQUENCE

df = bktg.generated_multi_sequences(30)
df = df.rename(columns = {'index': 'item_ID'})

df.to_csv('generated_student_data.csv', index=False)
