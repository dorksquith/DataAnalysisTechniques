# Python examples for week 1: Porbability
import numpy as np
from numpy import random


# Frequentist Probability
S = set(range(1,11)) 
print("S= ",S)
A = {si for si in S if si % 2 == 0} 
print("A= ",A)
B = {si for si in S if si <5} # {1,2,3,4}
print("B= ",B)

P_A = len(A)/len(S)  # 0.5
P_B = len(B)/len(S)  # 0.4
P_S = len(S)/len(S) 
print("P(A) =",P_A)
print("P(B) =",P_B)
print("Total probability P(S) =",P_S)




# Subsets 
print("Is A a subset of S?", A .issubset(S) ) 
# equivalent
print("Equivalent method: Is B a subset of A?", B  <= A ) 


# Union between A and B: python uses | operator
AUB = A|B 
print ("Union AUB =",AUB)



# Intersection between A and B: python uses & operator
AnB = A&B # {2,4}
print ("Intersection AnB =",AnB)



# Probabilities of union and intersection
# For our example sets (A and B are not mutually exclusive)
P_AUB = len(AUB)/len(S) # 0.7
P_AnB = len(AnB)/len(S) # 0.2
print("P(AUB) = ",P_AUB )
print("P(A) +P(B) - P(AnB) = {} + {} - {} = {}".format(P_A, P_B, P_AnB, P_A+P_B-P_AnB ))


# Conditional
# For our example sets A and B are independent
P_AgivenB = P_AnB / P_B # 0.2/0.4 = 0.5
print("P(A|B) = ", P_AgivenB )
P_BgivenA = P_AnB / P_A 
print("P(B|A) = ", P_BgivenA )


# Bayes theorem for our sets (very boring example as A,B independent)
Posterior = P_AgivenB
Likelihood = P_BgivenA
Prior_A = P_A
Prior_B = P_B
print("P(A|B): ", Posterior)
print("P(B|A)P(A)/P(B): ", Likelihood*Prior_A / Prior_B)

