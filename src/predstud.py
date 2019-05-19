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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

plt.rc('text', usetex=True)

font = {'size'   : 22}

plt.rc('font', **font)

def _qtilenorm(arr):
    return np.array([percentileofscore(arr,val) for val in arr]) 

RATFLD = ['ratingBusinessOutlook',
          'ratingCareerOpportunities', 
          'ratingCeo',
          'ratingCompensationAndBenefits',
          'ratingCultureAndValues', 
          'ratingOverall',
          'ratingRecommendToFriend',
          'ratingSeniorManagement', 
          'ratingWorkLifeBalance',
        ]

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

def feature_add(df): 
    '''
    add features to this from PCA, etc. 
    '''  
    pcawgts = np.array([0.27362386, 0.34379945, 0.29999705, 0.2817328 , 0.35799411, 0.38224064, 0.36879486, 0.36867397, 0.30226229])

    orgdf = pd.read_csv('data.csv')

    df['ratepca_f'] = df[RATFLD].dot(pcawgts) 
    #df['rate_f'] = _qtilenorm(df['rate_f'].values) 

    # salary and ratings percentage increase old to new 

    newflds = RATFLD + ['avgSalary']
    #newflds = RATFLD + ['avgSalary','employeesTotalNum']
    newnewflds = [val + '_new' for val in newflds]

    for fld1,fld2 in zip(newflds,newnewflds): 
        df[fld1+'_f'] = (orgdf[fld2] - orgdf[fld1])/orgdf[fld1]  

    df['job_f'] = (orgdf['jobLength_new'] + orgdf['jobLength_old'])/orgdf['jobLength_old']
    
    return df 

if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    pdf = normdata_construct(df)
    dropcols = ['gdSectorName','metro','jobTitle'] 
    pdf = pdf.drop(dropcols,axis=1)

    import ipdb 
    ipdb.set_trace()

    ## REMOVE COLS THAT SHOULD NOT BE HERE 

    # add features to df 
    pdf =feature_add(pdf) 

    ## QUANTILE NORMALIZE HERE ... TRY WITHOUT FIRST 

    keepcols = ['leftEmployer', 
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

    #pdf = pdf[keepcols]

    pdf = pdf.fillna(0)
    pdf[np.abs(pdf) > 1e8] = 1e8

    
    X = pdf.iloc[:,1:].values
    y = pdf['leftEmployer'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=43)

    reg = LinearRegression().fit(X_train, y_train)
    predvls = reg.predict(X_test)

    predvls = np.array([min(val,1.) for val in predvls])
    predvls = np.array([max(val,0.) for val in predvls])

    fpr, tpr, thresh = roc_curve(y_test,predvls)

    # model 2
    # WORK ON THIS MORE ... UNDERSTAND WHY UNDER PREF.
    model = LogisticRegression().fit(X_train,y_train)
    y_pred_logistic = model.decision_function(X_test)

    fpr1, tpr1, thresh1 = roc_curve(y_test,y_pred_logistic)

    # model 3
    #model = SVC(kernel='rbf',gamma='auto').fit(X_train,y_train)
    #y_pred_svc = model.decision_function(X_test)

    #fpr2, tpr2, thresh2 = roc_curve(y_test,y_pred_svc)

    #model 4 naieve bayes
    #gnb = GaussianNB()
    #y_pred = gnb.fit(X_train, y_train)
    #y_pred_nb = gnb.decision_function(X_test)
    #fpr3, tpr3, thresh3 = roc_curve(X_test,y_pred_nb)

    #model 5 quad dis
    #tp = MLPClassifier(alpha=1, max_iter=1000)
    #y_pred = tp.fit(X_train, y_train)
    #y_pred_nb = tp.decision_function(X_test)
    #fpr4, tpr4, thresh4 = roc_curve(X_test,y_pred_nb)


    model = DecisionTreeClassifier(max_depth=5).fit(X_train,y_train)
    m1pred = model.predict_proba(X_test)

    fpr4, tpr4, thresh4 = roc_curve(y_test,m1pred[:,1])

    #model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train,y_train)
    #m2pred = model.predict_proba(X_test)

    #fpr5, tpr5, thresh5 = roc_curve(y_test,m2pred[:,1])

    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #for i in range(2):
    #    fpr[i], tpr[i], _ = roc_curve(y_test[i], m2pred[i])

    #fpr5, tpr5, thresh5 = roc_curve(y_test,m2pred)
    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), m2pred.ravel())

    plt.figure()
    
    plt.plot([0, 1], [0, 1],'k--')
    plt.plot(fpr,tpr)
    plt.plot(fpr1,tpr1)
    #plt.plot(fpr2,tpr2)
    #plt.plot(fpr3,tpr3)
    plt.plot(fpr4,tpr4)
    #plt.plot(fpr5,tpr5)
    plt.axis('equal')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.show()

    ## Display Dec. Tree
    #export_graphviz(model, out_file=dot_data,  
    #                filled=True, rounded=True,
    #                special_characters=True)    
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    #Image(graph.create_png())

    dot_data = tree.export_graphviz(model,out_file='tree.dot',feature_names=pdf.columns[1:],class_names=pdf.columns[0])
    #graph = pydotplus.graphviz.graph_from_dot_data(dot_data)

    #tex = tree.export_graphviz(model, out_file='treepic.dot')
    #graph = pydotplus.graph_from_dot_data(tex)
    #graph.write_png('treetst.png')
