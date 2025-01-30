import numpy as np
import pandas as pd

# BKT RESULT
class BKTModel:
    """Apply Bayesian Knowledge Tracing model to calculate likelihood to
    produce a correct answer.
    p(correct) = p(L) * (1 - p(S)) + (1 - p(L)) * p(G)
    """
    def __init__(self, p_Lt, p_T, p_G, p_S, scores):
        self.p_Lt = p_Lt  # initial learning, default = new in learning
        self.p_T = p_T  # acquisition, default = slow move
        self.p_G = p_G  # guess, default = little guess, good behavior
        self.p_S = p_S  # slip, default = careful action
        self.scores = scores  # current answer (to update next p(Lt))
        
    
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
