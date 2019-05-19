import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import mquantiles
from scipy.stats import percentileofscore
from matplotlib.ticker import StrMethodFormatter
from collections import Counter
from numpy.linalg import eig
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve

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
                'ratingWorkLifeBalance',
                'employeesTotalNum',
                #'yearFounded',
                'gdSectorName',
                'metro',
                'jobTitle']

    tmpdf = df[keepcols1]
    
    for col in tmpdf.columns[1:12]:
        tmpdf[col] = [percentileofscore(tmpdf[col],val) for val in tmpdf[col].values] 

    le = preprocessing.LabelEncoder()
    le.fit(tmpdf['gdSectorName'])
    tmpdf['gdSectorName'] = le.transform(tmpdf['gdSectorName'])

    le = preprocessing.LabelEncoder()
    le.fit(tmpdf['metro'])
    tmpdf['metro'] = le.transform(tmpdf['metro'])

    le = preprocessing.LabelEncoder()
    le.fit(tmpdf['jobTitle'])
    tmpdf['jobTitle'] = le.transform(tmpdf['jobTitle'])

    return tmpdf

if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    pdf = normdata_construct(df)
    
    X = pdf.iloc[:,1:].values
    y = pdf['leftEmployer'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    predvls = reg.predict(X_test)

    predvls = np.array([min(val,1.) for val in predvls])
    predvls = np.array([max(val,0.) for val in predvls])

    fpr, tpr, thresh = roc_curve(y_test,predvls)

    # model 2
    # WORK ON THIS MORE ... UNDERSTAND WHY UNDER PREF.
    model = LogisticRegression().fit(X_train,y_train)
    y_pred_logistic = model.predict(X_test)

    fpr1, tpr1, thresh1 = roc_curve(y_test,y_pred_logistic)

    plt.figure()
    
    plt.plot([0, 1], [0, 1],'r--')
    plt.plot(fpr,tpr)
    plt.plot(fpr1,tpr1)
    plt.axis('equal')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.show()

