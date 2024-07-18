import numpy as np

sr=0.3
br=0.5
L=7
P=1
scales=[10,35,70,160,400,1000,1400]
mscales=[10,35,70,125,350,760,1300]
scales=scales[::-1]
mscales=mscales[::-1]
times=[33.28/4096,101.44/1024,101.44/256,117.12/64,185.28/16,214.72/4,110.72]
times=times[::-1]

print('Theoretical time for 8xA6000')
for idx,t in enumerate(times):
    discount=mscales[idx]/scales[idx]
    print(f'{scales[idx]}: {3.6/8*t*discount*3600:.2f}Gs, discount={discount:.2f}')


accu=0
Ks=[8.8,3.7,3.5,2.6,2.15,2.4,2.5]
Ks=Ks[::-1]


print(np.mean(Ks[:-1]))


Z1=20
Z2=5+br*5+br**2*5+br**3*5
print(f'Z1={Z1},Z2={Z2}')
# K=K*Z2/Z1

# Ps=[14,4,1,0,0,0,0]
# Ps=Ps[::-1]
P=1
for i in range(L):
    # P=Ps[i]
    if i<=4:
        K=Ks[i]*Z2/Z1
    else:
        K=Ks[i]
    discount=mscales[i]/scales[i]
    K=K*discount
    accu+=times[i]*P*K
    print(f'Level {i+1}',P,f'x{mscales[i]}',f'{K*times[i]*P:.2f}Ghrs',f'Total {accu:.2f}Ghrs')
    P=int(np.ceil(P/sr))

