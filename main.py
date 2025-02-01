#%% IMPORT LIBRARY & MODULE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module_em import EM4BKT, multi_sequence_training
from module_hmm_general import HMM
from module_bkt_model import BKTModel

# EDA
data1 = pd.read_csv('generated_student_data.csv')
data1.head()
data1.shape
raw_score = data1.groupby('student_ID')['score'].mean() * 10
fig = plt.figure(dpi=200)
raw_score.plot(kind='hist')
plt.xlabel('Student score (out of 10)')
plt.show()

student_list = data1['student_ID'].unique()


#%% EXP 1: BKT FOR INDIVIDUAL

# PART 1: ESTIMATED KNOWLEDGE STATE WITH VITERBI ALGO
viterbi_estimated_state_index_column = np.array([])

# PART 2: ESTIMATED KNOWLEDGE STATE WITH BKT MECHANISM
bkt_learned_state_column = np.array([])
bkt_parameter = {'p_Lt': [],
                 'p_T': [],
                 'p_G': [],
                 'p_S': []}

for sid in student_list:
    # PART EM
    scores = data1[data1['student_ID'] == sid]['score'].astype(int).to_list()
    embkt = EM4BKT(scores)
    embkt.training()
    # embkt.plot_log_likelihood()

    # PART 1
    hmm = HMM(embkt.pi,
              embkt.A,
              embkt.B,
              scores)
    estimated_state_index = hmm.viterbi()[1]
    
    viterbi_estimated_state_index_column = np.append(viterbi_estimated_state_index_column, estimated_state_index)

    # PART 2
    p_Lt=embkt.pi[1]
    p_T=embkt.A[0][1]
    p_G=embkt.B[0][1]
    p_S=embkt.B[1][0]
    bkt_parameter['p_Lt'] = np.append(bkt_parameter['p_Lt'], p_Lt)
    bkt_parameter['p_T'] = np.append(bkt_parameter['p_T'], p_T)
    bkt_parameter['p_G'] = np.append(bkt_parameter['p_G'], p_G)
    bkt_parameter['p_S'] = np.append(bkt_parameter['p_S'], p_S)
    
    learn_threshold = 0.95
    
    bm = BKTModel(p_Lt, p_T, p_G, p_S, scores, learn_threshold)
    bm.get_p_L()
    bkt_learned_state_column = np.append(bkt_learned_state_column, 
                                         np.array(bm.p_L_array)
                                         )

data1['viterbi_state_index'] = viterbi_estimated_state_index_column
data1['bkt_state_index'] = (bkt_learned_state_column >= learn_threshold).astype(int)

accuracy_viterbi = (data1['state_index'] == data1['viterbi_state_index']).astype(int).mean()
accuracy_bkt = (data1['state_index'] == data1['bkt_state_index']).astype(int).mean()

# HYPOTHESIS TEST FOR ACCURACY
# H0: p_viterbi is not higher than p_bkt <=> p_viterbi <= p_bkt
# HA: p_viterbi is higher than p_bkt <=> p_viterbi > p_bkt
# significant level = 0.05 <=> z_critical = 1.645
# p_value = p(p_viterbi > p_bkt | H0)
p_viterbi = accuracy_viterbi
p_bkt = accuracy_bkt
n = len(student_list)

z_stat = (p_viterbi - p_bkt) / np.sqrt(p_bkt * (1 - p_bkt) / n)
print(z_stat)


# COMPARE AVERAGE ESTIMATION
bkt_df = pd.DataFrame(bkt_parameter)
bkt_df.describe()
# Result:
#        p_Lt   p_T   p_G    p_S
# mean   0.00   0.46  0.35   0.17

# Compare to generated data
#           p_Lt   p_T   p_G    p_S
# setting   0.10   0.25  0.3   0.15

#%% EXP 2: BKT FOR GROUP OF STUDENT
(pi, A, B, 
 log_likelihood, epoch
 ) = multi_sequence_training(df=data1, ll_eps=1e-7, max_epoch = 200)
# Result
#         p_Lt   p_T   p_G    p_S
# value   0.00   0.44  0.38   0.21

pd.Series(log_likelihood).plot()
plt.show()

# COMPARE VITERBI VS. BKT
# PART 1: ESTIMATED KNOWLEDGE STATE WITH VITERBI ALGO
viterbi_estimated_state_index_column = np.array([])

# PART 2: ESTIMATED KNOWLEDGE STATE WITH BKT MECHANISM
bkt_learned_state_column = np.array([])
bkt_parameter = {'p_Lt': [],
                 'p_T': [],
                 'p_G': [],
                 'p_S': []}

for sid in student_list:
    # PART EM
    scores = data1[data1['student_ID'] == sid]['score'].astype(int).to_list()

    # PART 1
    hmm = HMM(pi, A, B, scores)
    estimated_state_index = hmm.viterbi()[1]
    
    viterbi_estimated_state_index_column = np.append(viterbi_estimated_state_index_column, estimated_state_index)

    # PART 2
    p_Lt = pi[1]
    p_T = A[0][1]
    p_G = B[0][1]
    p_S = B[1][0]
    bkt_parameter['p_Lt'] = np.append(bkt_parameter['p_Lt'], p_Lt)
    bkt_parameter['p_T'] = np.append(bkt_parameter['p_T'], p_T)
    bkt_parameter['p_G'] = np.append(bkt_parameter['p_G'], p_G)
    bkt_parameter['p_S'] = np.append(bkt_parameter['p_S'], p_S)
    
    learn_threshold = 0.95
    
    bm = BKTModel(p_Lt, p_T, p_G, p_S, scores, learn_threshold)
    bm.get_p_L()
    bkt_learned_state_column = np.append(bkt_learned_state_column, 
                                         np.array(bm.p_L_array)
                                         )

data1['viterbi_state_index'] = viterbi_estimated_state_index_column
data1['bkt_state_index'] = (bkt_learned_state_column >= learn_threshold).astype(int)

accuracy_viterbi = (data1['state_index'] == data1['viterbi_state_index']).astype(int).mean()
accuracy_bkt = (data1['state_index'] == data1['bkt_state_index']).astype(int).mean()

# HYPOTHESIS TEST FOR ACCURACY
# H0: p_viterbi is not higher than p_bkt <=> p_viterbi <= p_bkt
# HA: p_viterbi is higher than p_bkt <=> p_viterbi > p_bkt
# significant level = 0.05 <=> z_critical = 1.645
# p_value = p(p_viterbi > p_bkt | H0)
p_viterbi = accuracy_viterbi
p_bkt = accuracy_bkt
n = len(student_list)

z_stat = (p_viterbi - p_bkt) / np.sqrt(p_bkt * (1 - p_bkt) / n)
print(z_stat)








