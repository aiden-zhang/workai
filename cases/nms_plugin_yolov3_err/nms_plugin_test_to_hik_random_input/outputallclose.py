import numpy as np
import pdb
tmp1 = np.load('./output1.npy', allow_pickle= True)
tmp2 = np.load('./output2.npy', allow_pickle= True)
pdb.set_trace()
print(f'is output[0] allclose:{np.allclose(tmp1[0], tmp2[0], rtol=1.e-5, atol=1.e-8, equal_nan=False)}')
print(f'is output[2] allclose:{np.allclose(tmp1[2], tmp2[2], rtol=1.e-5, atol=1.e-8, equal_nan=False)}')