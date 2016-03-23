import numpy as np
"""#################################################"""
"""###############  GP  scoring  ###################"""
"""#################################################"""
"""so far is for un-normalized"""
def kernel_function(kernelName, k_lambda,ks):
    if kernelName == "SquaredExponentialKernelType":
        m_lambda=np.identity(len(k_lambda))*k_lambda
        kernel=np.dot(np.dot(ks.transpose(),m_lambda),ks)
        return np.exp(-kernel)
    elif kernelName == "AbsoluteExponentialKernelType":
        kernel=np.dot(abs(ks.transpose()),k_lambda)
        return np.exp(-kernel)
    elif kernelName == "CubicKernelType":
        td = np.abs(ks) *k_lambda.reshape(1, len(ks))
        td[td > 1.] = 1.
        ss = 1. - td ** 2. * (3. - 2. * td)
        ans = np.prod(ss, 1)
        return ans
    elif kernelName == "LinearKernelType":
        td=k_lambda.reshape(1,len(ks))*np.abs(ks)
        td[td>1.]=1.
        ss=1.-td
        ans=np.prod(ss)
        return ans


def radi_k(DX,k_lambda,nugget,kernelName):
    size=np.shape(DX)
    row=size[0]
    col=size[1]
    K=np.zeros((row,row))
    #k_lambda=(k_lambda*k_lambda)
    #m_lambda=np.identity(col)*k_lambda
    for i in range(row):
        for j in range(row):
            ki=DX[i]
            kj=DX[j]
            ks=ki-kj
            kernel=kernel_function(kernelName,k_lambda,ks)
            K[i][j]=kernel
            if (i==j):
                K[i][j]=K[i][j]+nugget
    K_1=np.linalg.inv(K)
    return K_1

def radi_new_k(tx,DX,k_lambda,kernelName):
    size=np.shape(DX)
    row=size[0]
    col=size[1]
    size_t=np.shape(tx)
    row_t=size_t[0]
    col_t=size_t[1]
    k_T=np.zeros((row_t,row))
    k_T=np.squeeze(k_T)
    #k_lambda=(k_lambda*k_lambda)
    for i in range(row_t):
        for j in range(row):
            ki=tx[i]
            kj=DX[j]
            ks=ki-kj
            kernel=kernel_function(kernelName,k_lambda,ks)
            k_T[i][j]=kernel
    return k_T

def radi_new_x(tx,k_lambda,nugget,kernelName):
    size=np.shape(tx)
    row=size[0]
    col=size[1]
    k_tx=np.zeros(row)
    for i in range(row):
        ks=tx[i]-tx[i]
        kernel=kernel_function(kernelName,k_lambda,ks)
        k_tx[i]=kernel+nugget
    return k_tx