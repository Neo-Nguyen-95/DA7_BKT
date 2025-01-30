#%% IMPORT LIBRARY & MODULE
import numpy as np
import pandas as pd
from module_em import EM4BKT
from module_hmm_general import HMM
import matplotlib.pyplot as plt

#%% EXP 1: BKT FOR INDIVIDUAL

# EDA
data1 = pd.read_csv('generated_student_data.csv')
data1.head()
raw_score = data1.groupby('student_ID')['score'].mean() * 10
raw_score.plot(kind='hist')
plt.show()

student_list = data1['student_ID'].unique()

# blank column for estimated state
viterbi_estimated_state_index_column = np.array([])

for sid in student_list:
    scores = data1[data1['student_ID'] == sid]['score'].astype(int).to_list()
    embkt = EM4BKT(scores)
    embkt.training()

    hmm = HMM(embkt.pi,
              embkt.A,
              embkt.B,
              scores)
    estimated_state_index = hmm.viterbi()[1]
    
    viterbi_estimated_state_index_column = np.append(viterbi_estimated_state_index_column, estimated_state_index)

data1['viterbi_state_index'] = viterbi_estimated_state_index_column

accuracy = (data1['state_index'] == data1['viterbi_state_index']).astype(int).mean()


#%% EXP 2: BKT FOR GROUP OF STUDENT




