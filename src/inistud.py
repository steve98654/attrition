import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import mquantiles
from scipy.stats import percentileofscore
from matplotlib.ticker import StrMethodFormatter
from collections import Counter
from numpy.linalg import eig

plt.rc('text', usetex=True)

font = {'size'   : 22}

plt.rc('font', **font)

def normdata_construct(df):
    '''
    '''
    keepcols1 = ['leftEmployer', 
                'jobLength_old', # in months
                'avgSalary',
                'ratingBusinessOutlook',
                'ratingCareerOpportunities', 
                'ratingCeo',
                'ratingCompensationAndBenefits',
                'ratingCultureAndValues', 
                'ratingOverall',
                'ratingRecommendToFriend',
                'ratingSeniorManagement', 
                'ratingWorkLifeBalance']

    tmpdf = df[keepcols1]
    
    for col in tmpdf.columns[1:]:
        tmpdf[col] = [percentileofscore(tmpdf[col],val) for val in tmpdf[col].values ] 

    return df[keepcols1],tmpdf

def _distcomp(raw_df,col):
    '''
    '''
    attrition_yes= raw_df[raw_df['leftEmployer']==1]
    attritio_no = raw_df[raw_df['leftEmployer']==0]

    sns.kdeplot(attrition_yes[col], label="Left" ,shade=True, color="r")
    sns.kdeplot(attritio_no[col], label="Stayed" ,shade=True,color="g")
    plt.xlabel(col)
    plt.title('Attrition Vs. Rating TOP Management')


def _dstmk():
    plt.figure()
    plt.subplot(1,2,1)
    _distcomp(df,df.columns[15])
    plt.xlabel('Friend Rating')
    plt.title('Friend Rec. Distributions')
    plt.subplot(1,2,2)
    _distcomp(df,df.columns[20])
    plt.xlabel('Worklife Rating')
    plt.title('Friend Rating Distributions')
    plt.title('Worklife Distributions')

    plt.show()

def ratingPCA():  

    rting = ['ratingBusinessOutlook',
             'ratingCareerOpportunities', 
             'ratingCeo',
             'ratingCompensationAndBenefits',
             'ratingCultureAndValues', 
             'ratingOverall',
             'ratingRecommendToFriend',
             'ratingSeniorManagement', 
             'ratingWorkLifeBalance']


    disp_flds = ['Outlook','Oppor','CEO','Comp','Culture','Overall','Friend','Mgmt','Life']

    tmpdf = df.copy()[rting]

    for col in tmpdf.columns:
        tmpdf[col] = [percentileofscore(tmpdf[col],val)/100. for val in tmpdf[col].values ] 

    tmpdf.columns = disp_flds
    covmat = tmpdf.cov()

    ev,evc = eig(covmat)


    plt.figure()
    plt.subplot(1,2,1)
    sns.heatmap(tmpdf.corr(),annot=True,xticklabels=True,yticklabels=True,cbar_kws={'label':'Correlation','format':'{%.1f\%%}'})

    plt.title('Rating Corr. Matrix')
    plt.subplot(1,2,2)
    pltsrs = ev.cumsum()/ev.sum()

    sns.tsplot(pltsrs,range(1,len(ev)+1))
    plt.title('Cum. Var. Explained')
    plt.xlabel('Numbers of PCs')

    import ipdb
    ipdb.set_trace()


# 13, 15 , 19, 20   

def _varplot(tmpdf,varname,iflog=False): 
    '''
    old vs new variable plot 
    '''
    
    newvarname = varname + '_new' 
    v1 = tmpdf[varname]
    v2 = tmpdf[newvarname]

    if iflog:
        plt.plot(np.log10(v1),np.log10(v2),'bx')
    else:
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
    dfchg = (df.avgSalary_new - df.avgSalary)/df.avgSalary
    abchg = (df.avgSalary_new - df.avgSalary)

    df['relsal'] = (df.avgSalary_new - df.avgSalary)/df.avgSalary
    df['abssal'] = (df.avgSalary_new - df.avgSalary)

    print "Max/Min Change"
    print dfchg.min()
    print dfchg.max()

    #dfchg = dfchg[(dfchg < 50000.) & (dfchg > -50000.)] 

    print "Per 0 Zeros"
    dfchgvl = len(dfchg[np.abs(dfchg) < 0.0001])/5550.
    print dfchgvl

    fig,axes = plt.subplots(1,2)

    dfchg = dfchg[np.abs(dfchg) > 0.0001]
    abchg = abchg[np.abs(abchg) > 0.0001]

    dfchg.hist(bins=100,ax=axes[0])
    axes[0].set_title('Relative Salary Change')
    axes[0].set_xlabel('Salary Pct. Chg.') 
    axes[0].set_ylabel('Freq. Count')
    axes[0].grid(False)
        
    (abchg/1000.).hist(bins=100,ax=axes[1])
    plt.grid(b=None)
    axes[1].set_title('Absolute Salary Change')
    axes[1].set_xlabel('Abs. Salary Chg.') 
    axes[1].grid(False)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}$K')) # 2 decimal places
    #axes[1].set_ylabel('Freq. Count')

    #plt.xlim([-1,4])

    plt.show()

    import ipdb
    ipdb.set_trace()

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

    newdiscols = ['Acct',
                  'Bus Ser',
                  'Educ',
                  'Energy',
                  'Fin',
                  'Food',
                  'H Care',
                  'Inf Tec',
                  'Ins',
                  'Manf',
                  'Media',
                  'Pharm',
                  'Retail',
                  'Tele',
                  'Travel'] 

    transdf.columns = newdiscols
    transdf.index = newdiscols

    # industry ranking 
    pltdf = np.round(100*(transdf/transdf.sum()))
    sns.heatmap(pltdf,annot=True,xticklabels=True,yticklabels=True,cbar_kws={'label':'Job Transition Percentage','format':'{%.0f\%%}'})
    plt.title('Industry Transition (from row to column)')

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


    ## next vis 
    tdf1 = df[new_rating_flds]
    tdf2 = df[rating_flds]

    for col in tdf1.columns:
        tdf1[col] = [percentileofscore(tdf1[col],val) for val in tdf1[col].values ] 
    for col in tdf2.columns:
        tdf2[col] = [percentileofscore(tdf2[col],val) for val in tdf2[col].values ] 
        
    tdf1.columns = tdf2.columns 
    diffdf = tdf1-tdf2
    diffdf.hist(bins=30)

    plt.show()

    import ipdb
    ipdb.set_trace()

    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    filtflds = ['theSame','resumeId'] 
    kpflds = [val for val in df.columns if val not in filtflds]

    df = df[kpflds] 

    #df = df[df.leftEmployer == 1] 

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
    if True:
        old_vs_new(df) 
        #ratingPCA()
        #inttrans = industry_trans(df)
        #_varplot(df,'employeesTotalNum')
        #rating_comp(df)
        #normdata_construct(df)
    else:
        inttrans = industry_trans(df)

    ## TODO, QUANTILE NORMALIZE RATINGS COMP
    ## GOOD GRAPHIC FOR TRANS. 
    ## ANY INTERESTING VARPLOTS? 
