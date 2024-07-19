import numpy as np



def linear_budget(L,roll=0):
    db=np.cumsum(np.ones(L))[::-1]
    db/=db.sum()
    if roll>=L: 
        db=np.zeros(L)
        db[-1]=1
    else:
        db=np.roll(db,roll)
        residual=db[:roll].sum()
        db[:roll]=0
        db[roll-1]+=residual
    
    db=np.zeros(L)
    db[-1]=1
    return db

def cost_estimate(scales,costs,sr=0.25,L=4,warmup=0,title=None,mode='H100'):
    b=1
    budgets={}
    for s in scales:
        budgets[s]=int(b)
        b=np.ceil(b/sr)
    
    print('_'*80)
    print()
    if title:
        print(f'Cost Estimate for the Scale Climbing: {title}')
    print()

    def scale_time(s,warmup,db,c500,c2k):
        b=budgets[s]
        wb=np.floor(warmup*b)
        r=b-wb
        cost_weights=np.cumsum(np.ones(L)/L)
        avg=np.dot(db,cost_weights)
        rs=[]
        for i in range(L-1):
            rs.append(int(r*db[i]))
        rs.append(int(r-np.sum(rs)))
        print('Scale:',s, ' Budget:',b)
        print(f'Warmup: {int(wb)} runs, Rest: {rs}, avg. {avg:.2f}')
        cost=[w*c2k for w in cost_weights]
        st=wb*c500+np.dot(rs,cost)
        if mode=='H100':
            return st*8/3.6/3600
        elif mode=='A6000x8':
            return st/3600

    at=0
    for idx,s in enumerate(scales[::-1]):
        # warmup=0.2 #warmups[s]
        db=linear_budget(L,roll=idx)
        print(db)
        st=scale_time(s,warmup,db,*costs[s])
        at+=st
        print(f'Total: {st:.1f}, accumulated [{at:.1f}] GPUhrs ({mode})\n')
    print('_'*80)


if __name__=='__main__':
    
    costs_lower={
        14:[76,174],
        31:[190,437],
        70:[11.2*60,22.3*60],
        125:[80*60,146.3*60],
        350:[10*3600,17*3600],
        760:[30*3600,54.3*3600],
        1300:[0,137.5*3600],
    }

    costs_upper={
        14:[43,46],
        31:[112,123],
        70:[7.5*60,8.3*60],
        125:[50*60,54.3*60],
        350:[7.5*3600,7.85*3600],
        760:[30*3600,32.6*3600],
        1300:[0,92.5*3600],
    }

    # scales=[1300,760,350,125,70,31,14]
    # scales=[760,350,125,70,31,14]
    # scales=[350,125,70,31,14]
    scales=[125,70,31,14]
    sr=0.25
    mode='A6000x8'
    cost_estimate(scales,costs_lower,sr=sr,title='Lower bound',mode=mode)
    cost_estimate(scales,costs_upper,sr=sr,title='Upper bound',mode=mode)
