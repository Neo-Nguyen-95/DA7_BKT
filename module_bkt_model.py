import numpy as np
#%% BKT CLASS
class BKTModel:
    """Apply Bayesian Knowledge Tracing model to calculate likelihood to
    produce a correct answer.
    p(correct) = p(L) * (1 - p(S)) + (1 - p(L)) * p(G)
    """
    def __init__(self, p_Lt, p_T, p_G, p_S, scores, learn_threshold=0.95):
        self.p_Lt = p_Lt  # initial learning, default = new in learning
        self.p_T = p_T  # acquisition, default = slow move
        self.p_G = p_G  # guess, default = little guess, good behavior
        self.p_S = p_S  # slip, default = careful action
        self.scores = scores  # current answer (to update next p(Lt))
        
        self.p_L_array = []  # save the p_L0 as the first value
        
        self.learn_threshold = learn_threshold
        self.learned_detect = False
        
    
    def __str__(self):
        return f"The currect p(learned skill) is {self.p_Lt}"
    
    def p_Correct(self):
        return self.p_Lt * ( 1- self.p_S) + (1 - self.p_Lt) * self.p_G
    
    def update_p_Lt(self, score):
        p_L_given_correct = (self.p_Lt * ( 1- self.p_S)) / self.p_Correct()
        p_L_given_incorrect = (self.p_Lt * self.p_S) / (1 - self.p_Correct())
        if score == 1:
            self.p_Lt = p_L_given_correct + (1 - p_L_given_correct) * self.p_T
        else:
            self.p_Lt = p_L_given_incorrect + (1 - p_L_given_incorrect) * self.p_T
            
    def get_p_L(self):
        for score in self.scores:
            # Update p_Lt of current observation,
            # during this process, p_Correct() use past p_Lt 
            self.update_p_Lt(score)  # self.p_Lt get new value after this method
            
            if self.p_Lt >= self.learn_threshold:
                self.learned_detect = True
            
            if self.learned_detect:
                self.p_L_array.append(np.array([1]))
            else:    
                self.p_L_array.append(self.p_Lt)  # Store new value in the array
            
#%% TEST SITE
# bm = BKTModel(p_Lt = 0.2, p_T=0.4, p_G=0.1, p_S=0.3, scores=[0, 1, 1, 0, 0, 0, 0, 0])
# bm.get_p_L()
# print(bm.p_L_array)
