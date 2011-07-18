from starflow.utils import ListUnion

def uset(x,K):
    J = [x[K_] for K_ in K]
    L = map(list,zip(*J))
    return ListUnion(L)
    

def mixup(x,Kset):

    Tset = [uset(x,K) for K in Kset]
    L = ListUnion(map(list,zip(*Tset)))
    N = len(L)/len(Kset)
    return [L[N*j:N*(j+1)] for j in range(len(Kset))]

    