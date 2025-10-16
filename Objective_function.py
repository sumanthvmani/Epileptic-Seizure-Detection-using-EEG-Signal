import numpy as np
from Glob_Vars import Glob_Vars
from Model_MATCN_AM import Model_MATCN_AM


def Objfun_Cls(Soln):
    Feat1 = Glob_Vars.Feat_1
    Feat2 = Glob_Vars.Feat_2
    Feat3 = Glob_Vars.Feat_3
    Feat4 = Glob_Vars.Feat_4
    Target = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        steps = 200
        Eval = Model_MATCN_AM(Feat1, Feat2, Feat3, Feat4, Target,sol.astype('int'))
        Fitn[i] = 1 / (Eval[4] + Eval[7]) #  1 / (Accuracy + Precision)
    return Fitn
