import numpy as np

scales=[1300,760,350,125,70,31,14]
scales=[350,125,70,31,14]
sr=0.25
budgets={}
b=1
for s in scales:
    budgets[s]=int(b)
    b=np.ceil(b/sr)
print(budgets)


warmups={
    14:0.3,
    31:0.25,
    70:0.2,
    125:0.15,
    350:0.1,
    760:0.05,
    1300:0,
}
dbs={
    14:[0.4,0.3,0.2,0.1],
    31:[0.3,0.3,0.2,0.2],
    70:[0.2,0.3,0.2,0.3],
    125:[0.1,0.2,0.3,0.4],
    350:[0,0.1,0.2,0.7],
    760:[0,0,0,1],
    1300:[0,0,0,1],
}
costs={
    14:[76,174],
    31:[190,437],
    70:[11.2*60,22.3*60],
    125:[80*60,146.3*60],
    350:[10*3600,17*3600],
    760:[30*3600,54.3*3600],
    1300:[0,137.5*3600],
}

def scale_time(b,warmup,db,c500,c2k):
    wb=np.ceil(warmup*b)
    r=b-wb
    avg=np.dot(db,[0.25,0.5,0.75,1])
    print(f'warmup: {int(wb)} runs, rest: {int(r*db[0])}, {int(r*db[1])}, {int(r*db[2])}, {int(r*db[3])}, avg. {avg:.2f}')
    st=wb*c500+r*c2k*avg
    return st*8/3.6/3600

at=0
for s in scales:
    st=scale_time(budgets[s],warmups[s],dbs[s],*costs[s])
    at+=st
    print(f'{s}: {st:.1f} H100 Ghours, accumulated {at:.1f}\n')