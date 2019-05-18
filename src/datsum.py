import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import mquantiles
from scipy.stats import percentileofscore
from collections import Counter
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

plt.rc('text', usetex=True)
font = {'size'   : 22}
plt.rc('font', **font)

desflds = ['leftEmployer', 
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

oldrate1 = ['ratingCareerOpportunities', 
           'ratingCompensationAndBenefits',
           'ratingCultureAndValues', 
           'ratingOverall',
           'ratingSeniorManagement', 
           'ratingWorkLifeBalance',
           ]


oldrate2 = ['ratingBusinessOutlook',
           'ratingCeo',
           'ratingRecommendToFriend',
           ]

newcompflds=['avgSalary_new',
             'employerName_new',
             'employeesTotalNum_new', 
             'endDate_new', 
             'gdSectorName_new',
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

cols = desflds + oldcompflds + newcompflds

df = pd.read_csv('data.csv')


## Avg Sal Graph
plt.figure() 
plt.subplot(1,2,1)
sns.distplot(df['avgSalary']/1000.,norm_hist=False,bins=100,axlabel='',color='mediumblue',kde=False).set(xlim=(df['avgSalary'].min()/1000.,200)) 
plt.yticks([], [])
plt.title('Avg. Annual Salary Dist.')
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}$K')) # 2 decimal places

plt.subplot(1,2,2)
sns.distplot(df['jobLength_old'].values/12.,bins=50,color='mediumblue',kde=False)
plt.xlim([0,6.])
plt.title('Tenure Orig. Empl. (Years)')
#.set(xlim=(df['jobLength_old'].min()/12.,6)) 


plt.figure()
plt.subplot(1,2,1)
sns.distplot(df.yearFounded.dropna(),bins=40,kde=False,color='mediumblue')
plt.ylabel('Bin Count')
plt.xlabel('Orig. Year Founded')
plt.xlim([1750,2020])
plt.title('Orig. Empl. Founding Year')
#plt.grid(b=None)

plt.subplot(1,2,2)
sns.distplot(np.log10(df.employeesTotalNum[df.employeesTotalNum>1]),bins=40,kde=False,color='mediumblue')
plt.xlabel(r'$\log_{10}($Orig. Tot. Emps.$)$')
plt.title('Log of Orig. Num. Emp.')
#plt.grid(b=None)

plt.figure()
plt.subplot(1,2,1)
tmpdf = df[oldrate1]
tmpdf.columns = ['Car.', 'Comp.','Cult.', 'Over.','Mgmt.','W. L.']
dfp = tmpdf.melt(var_name='groups', value_name='vals')
sns.violinplot(x="groups",y="vals",data=dfp)
plt.xticks(rotation=90)
plt.title('Orig. Comp. Ratings [0-5]')
plt.ylabel("")
tmpdf = df[oldrate2]
tmpdf.columns = ['Out.','CEO','Friend']
plt.subplot(1,2,2)
dfp = tmpdf.melt(var_name='groups', value_name='vals')
sns.violinplot(x="groups",y="vals",data=dfp)
plt.ylabel("")
plt.title('Orig. Comp. Ratings [0-1]')
plt.xticks(rotation=90)

plt.show()
