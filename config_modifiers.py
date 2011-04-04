import numpy as np 

class BaseModifier(object):
    def __init__(self,modifier_params):
        self.params = modifier_params.keys()
        self.modifier_params = modifier_params
        
    def get_modifications(self,k,val):
        
        Hp =  np.arange(val,val + self.modifier_params[k]['deltamax'],self.modifier_params[k]['deltadelta']).tolist() 
        Hn =  np.arange(val,val - self.modifier_params[k]['deltamax'],-self.modifier_params[k]['deltadelta']).tolist()
               
        Hp.reverse()
        Hn.reverse()
        return weave(Hp,Hn)

def weave(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
        c.append(b[i])
    return c