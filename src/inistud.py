import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import mquantiles
from scipy.stats import percentileofscore
from collections import Counter

plt.rc('text', usetex=True)

font = {'size'   : 22}

plt.rc('font', **font)

def _varplot(tmpdf,varname): 
    '''
    old vs new variable plot 
    '''
    
    newvarname = varname + '_new' 
    v1 = tmpdf[varname]
    v2 = tmpdf[newvarname]

    plt.plot(v1,v2,'bx')
    plt.xlabel('old')
    plt.ylabel('new')
    plt.title(varname)
    plt.axis('equal')
    plt.show() 

def _empcomp(df,row): 
    ''' 
    row is index of res in df
    '''
    df1 = pd.DataFrame(df.iloc[row,:][oldcompflds])
    df2 = pd.DataFrame(df.iloc[row,:][newcompflds])

    df2.index = df1.index 

    cmbdf = pd.concat([df1,df2],axis=1)
    cmbdf.columns = ['old_comp','new_comp']

    return cmbdf
    
def old_vs_new(df): 
    '''
    Old vs new salary 
    '''
    dfchg = (df.avgSalary_new - df.avgSalary)
    print "Max/Min Change"
    print dfchg.min()
    print dfchg.max()

    dfchg = dfchg[(dfchg < 50000.) & (dfchg > -50000.)] 
    dfchg.hist(bins=100)
    plt.title('Salary Difference After Leaving')
    plt.xlabel('Salary Change') 
    plt.ylabel('Freq. Count')

    plt.show()

def industry_trans(df): 
    '''
    '''
    #industs = sorted(list(df.gdSectorName.value_counts().index))


    industs = sorted(list(df.gdSectorName.value_counts().head(15).index))

    dispcols = [val.replace(' & ','/') for val in industs]

    transmat = np.zeros((len(industs),len(industs)))
    transdf = pd.DataFrame(transmat,index=industs,columns=industs)

    for i in range(len(df)):
        try:
            indu1 = df.iloc[i,:]['gdSectorName']
            indu2 = df.iloc[i,:]['gdSectorName_new']

            transdf.loc[indu1,indu2] = transdf.loc[indu1,indu2] + 1. 
        except:
            pass

    transdf.columns = dispcols
    transdf.index = dispcols

    # industry ranking 
    pltdf = np.round(100*(transdf/transdf.sum()))
    sns.heatmap(pltdf,annot=True,xticklabels=True,yticklabels=True,cbar_kws={'label':'Pct. Persons'})
    plt.title('Industry Transition Table (from row to column)')

    plt.show()

    return transdf

def metro_trans(df):
    '''
    not interesting, cols are the same!
    '''
    comps = sorted(list(df.metro.value_counts().head(89).index))

    dispcols = [val.replace(' & ','-') for val in comps]

    transmat = np.zeros((len(comps),len(comps)))
    #transdf = pd.DataFrame(transmat,index=industs,columns=industs)
    translst = []

    for i in range(len(df)):
        try:
            indu1 = df.iloc[i,:]['metro']
            indu2 = df.iloc[i,:]['metro_new']
            translst.append((indu1,indu2))
        except:
            pass

    tlst = Counter(translst)
    return tlst

def rating_comp(df):
    '''
    comp ratings before/after
    '''

    rating_flds = ['ratingBusinessOutlook',
    'ratingCareerOpportunities', 
    'ratingCeo',
    'ratingCompensationAndBenefits',
    'ratingCultureAndValues', 
    'ratingOverall',
    'ratingRecommendToFriend',
    'ratingSeniorManagement', 
    'ratingWorkLifeBalance']

    disp_flds = ['Outlook','Oppor','CEO','Comp','Culture','Overall','Friend','Mgmt','Life']
    new_rating_flds_disp = [val + '-new' for val in disp_flds]

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx()

    width = 0.4

    new_rating_flds = [val + '_new' for val in rating_flds]
    
    plt1 = df[rating_flds].mean()
    plt2 = df[new_rating_flds].mean()

    # ratings of all companies 

    ratefld = rating_flds[8]
    ratefldnew = ratefld + '_new'
    allrates = df[['employerName',ratefld]].drop_duplicates()[ratefld].values


    oldq = [percentileofscore(allrates,val) for val in df[ratefld].values ] 
    newq = [percentileofscore(allrates,val) for val in df[ratefldnew].values ] 

    plt.figure() 
    plt.subplot(1,2,1)
    plt.plot()
    plt.plot(oldq,newq,'x')
    plt.title(ratefld)

    plt.subplot(1,2,2)
    plt.hist(np.array(newq)-np.array(oldq),bins=50)
    plt.title('New less Old Rating Quantile Diff.')
    plt.show()

    # mvls = percentileofscore(df[ratefld],df[ratefld][0])
    ## Let's compare ratings from a quantile perspective 


    plt1.index = disp_flds
    plt2.index = new_rating_flds_disp

    plt1.plot(kind='bar',color='red',ax=ax,width=width,position=1)
    plt2.plot(kind='bar',color='blue',ax=ax2,width=width,position=0)

    ax.set_ylabel('Old Rating')
    ax2.set_ylabel('New Rating')

    df[rating_flds].hist(bins=30)

    tmpdf = df[new_rating_flds]
    tmpdf.columns = new_rating_flds_disp
    tmpdf.hist(bins=30)

    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    filtflds = ['theSame','resumeId'] 
    kpflds = [val for val in df.columns if val not in filtflds]

    df = df[kpflds] 

    df = df[df.leftEmployer == 1] 

    desflds = ['empAgeAtExit', 
               'leftEmployer', 
               'newMinusOldSalary',
               'rNum', 
               'salaryIncrease', 
               'startDate', 
               'startDate_new', 
               'yearFounded', 
               'yearFounded_new', 
              ]

    oldcompflds = ['avgSalary',
                   'employerName',
                   'employeesTotalNum', 
                   'endDate', 
                   'gdSectorName',
                   'jobLength_old',
                   'jobTitle', 
                   'metro',
                   'ratingBusinessOutlook',
                   'ratingCareerOpportunities', 
                   'ratingCeo',
                   'ratingCompensationAndBenefits',
                   'ratingCultureAndValues', 
                   'ratingOverall',
                   'ratingRecommendToFriend',
                   'ratingSeniorManagement', 
                   'ratingWorkLifeBalance',
                   ]

    newcompflds=['avgSalary_new',
                 'employerName_new',
                 'employeesTotalNum_new', 
                 'endDate_new', 
                 'gdSectorName_new',
                 'jobLength_new', 
                 'jobTitle_new', 
                 'metro_new', 
                 'ratingBusinessOutlook_new', 
                 'ratingCareerOpportunities_new',
                 'ratingCeo_new', 
                 'ratingCompensationAndBenefits_new', 
                 'ratingCultureAndValues_new',
                 'ratingOverall_new',
                 'ratingRecommendToFriend_new',
                 'ratingSeniorManagement_new',
                 'ratingWorkLifeBalance_new', 
                ]

    # Begin studies 
    if False:
        old_vs_new(df) 
        inttrans = industry_trans(df)
        _varplot(df,'employeesTotalNum')
    else:

        rating_comp(df)
        
