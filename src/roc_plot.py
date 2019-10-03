import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

plt.rc('text', usetex=True)
font = {'size'   : 22}
plt.rc('font', **font)


df1 = pd.read_csv('dr1_res.csv')
df2 = pd.read_csv('dr2_res.csv')
df3 = pd.read_csv('dr_res5.csv')

dflin = pd.read_csv('orig_linroc.csv')
dflog = pd.read_csv('orig_log.csv')
decflin = pd.read_csv('orig_dectree.csv')


dflinfull = pd.read_csv('full_linear.csv')
dflogfull = pd.read_csv('full_logistic.csv')
decflinfull = pd.read_csv('full_dectree.csv')
forfull = pd.read_csv('full_rand_for.csv')


def _rocvl(df):
    '''
    Get ROC Vals 
    '''
    f1 = df['False Positive Rates']
    f2 = df['True Positive Rates']
    return f1, f2


f1,f2 = _rocvl(df1)
f1full,f2full = _rocvl(df3)

aucorig = [0.650,0.699,0.736] 

plt.figure()
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--')
plt.plot(dflin.iloc[:,1],dflin.iloc[:,2],label='Linear')
plt.plot(dflog.iloc[:,1],dflog.iloc[:,2],label='Logistic')
plt.plot(decflin.iloc[:,1],decflin.iloc[:,2],label='Dec. Tree')
plt.plot(f1,f2,label='Random Forest') 
plt.title('Original ROC Curves')
plt.axis('equal')
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


aucfull = [0.674,0.92,0.976] 

## CHECK IF WE GOT THE CORRECT 2ND ROC CURVE 

plt.figure()
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--')
plt.plot(dflinfull.iloc[:,1],dflinfull.iloc[:,2],label='Linear')
plt.plot(dflogfull.iloc[:,1],dflogfull.iloc[:,2],label='Logistic')
plt.plot(decflinfull.iloc[:,1],decflinfull.iloc[:,2],label='Dec. Tree')
plt.plot(forfull.iloc[:,0],forfull.iloc[:,1],label='Random Forest')
plt.plot(f1full,f2full,label='Light GBT') 
plt.title('Full Feature ROC Curves')
plt.axis('equal')
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

