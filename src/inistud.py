import pandas as pd 
import matplotlib.pyplot as plt

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
    

df = pd.read_csv('data.csv')

# remove resumeId and reindex 

# Meaningful Columns 
# rNum -- ?? 

# check if job-len corresponds to start,end date difference 
# geographic studies?  Large vs small cities? 

# patterns that stem from either age or employment tenure? 

# industry specific studies 

filtflds = ['theSame','resumeId'] 
kpflds = [val for val in df.columns if val not in filtflds]

df = df[kpflds] 

df = df[df.leftEmployer == 1] 

# Build rating df and description df 

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


#rating_fields = [ ] 


## Summary Stats 
