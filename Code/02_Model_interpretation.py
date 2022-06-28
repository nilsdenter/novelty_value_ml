"""
            PERMUTATION IMPORTANCE OF TECHNOLOGICAL IMPORTANCE
"""   

def permut_tech(iterations = 500, train = False, years = "7_years", top_x = "top_10"):
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from time import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import matplotlib as mpl
    import os
    import numpy as np
    np.random.seed(seed=0)
    import seaborn as sns
    sns.set_palette(['#000000'], n_colors=100)
    model_name = "MLP"
    cm = 1/2.54
    mpl.rcParams['font.size'] = 10
    mpl.rcParams["font.family"] = "arial"
    iterations = 500
    model_name = "MLP"
   
    t0 =time()
    data = pd.read_csv("Input_data_scaled_citations_further_controls.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 27 #87 oder 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    X = data.iloc[:,columns_iv]
    y = data["{0}_CIT_{1}".format("".join(top_x.upper().split("_")), "".join(years.upper().split("_")))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("citations_{0}_{1}".format(years, top_x))
    
    model =  joblib.load("{0}.joblib".format(model_name))
    
    result_1 = permutation_importance(model, X_test, y_test, n_repeats=iterations,
                            random_state=0, n_jobs=-1, scoring="roc_auc")
    sorted_idx = result_1.importances_mean.argsort()
    
    novelty_variables = [i for i in range(0,12)]
    novelty_sorted_idx = []
    for entry in sorted_idx:
        if entry in novelty_variables:
            novelty_sorted_idx.append(entry)
    
    fig, ax = plt.subplots(figsize=[12*cm,7.5*cm])
    ax.boxplot(result_1.importances[novelty_sorted_idx].T,
               vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
    ax.grid()
    ax.set(xlabel='\u0394 decrease in ROC AUC score')
    ax.locator_params(tight=True, nbins=6, axis="x")
    
    fig.tight_layout()
    #fig.savefig("{0}_Permutation_Technological_Importance.pdf".format(model_name), dpi=1600)
    #fig.savefig("{0}_Permutation_Technological_Importance.png".format(model_name), dpi=1600)
    #fig.savefig("{0}_Permutation_Technological_Importance.eps".format(model_name), format="eps", dpi=1600)
    fig.savefig("{0}_Permutation_Technological_Importance.svg".format(model_name), format="svg", dpi=1600)
    if train:
        #test and train beside each other
        result_2 = permutation_importance(model, X_train, y_train, n_repeats=iterations,
                                random_state=0, n_jobs=-1, scoring="roc_auc")
        
        fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=[6.4,4], sharey=True)
        mpl.rcParams['font.size'] = 10
        ax1.boxplot(result_1.importances[novelty_sorted_idx].T,
                   vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
        ax1.grid()
        ax1.set(xlabel='\u0394 decrease in ROC AUC score')
        ax1.set(title='Test Data')
        ax1.locator_params(tight=True, nbins=3, axis="x")
        
        ax2.boxplot(result_2.importances[novelty_sorted_idx].T,
                   vert=False, labels=[i.lower() for i in X_train.columns[novelty_sorted_idx]])
        ax2.grid()
        ax2.set(xlabel='\u0394 decrease in ROC AUC score')
        ax2.set(title='Training Data')
        ax2.locator_params(tight=True, nbins=3, axis="x")
        
        
        mpl.rcParams['font.size'] = 12
        fig.tight_layout()
        #fig.savefig("{0}_Permutation_Technological_Importance_test_train.pdf".format(model_name), dpi=1600)
        #fig.savefig("{0}_Permutation_Technological_Importance_test_train.png".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Technological_Importance_test_train.eps".format(model_name), format="eps", dpi=1600)
    print("Technological Importance finished")
    plt.show()
    os.chdir("../")
    from IPython import get_ipython
    get_ipython().magic('reset -sf')

"""
PERMUTATION IMPORTANCE OF ECONOMIC IMPORTANCE             
"""
def permut_econ(iterations = 500, train = False, top_x = "top_10"):
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from time import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import matplotlib as mpl
    import os
    import numpy as np
    np.random.seed(seed=0)
    import seaborn as sns
    sns.set_palette(['#000000'], n_colors=100)
    model_name = "MLP"
    cm = 1/2.54
    mpl.rcParams['font.size'] = 10
    mpl.rcParams["font.family"] = "arial"
    iterations = 500
    model_name = "MLP"  
    
    t0 =time()
    data = pd.read_csv("Input_data_scaled_KPSS_further_controls.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 27
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    X = data.iloc[:,columns_iv]
    y = data["{0}_KPSS".format("".join(top_x.upper().split("_")))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("kpss_{0}".format(top_x))
    
    model =  joblib.load("{0}.joblib".format(model_name))
    
    result_1 = permutation_importance(model, X_test, y_test, n_repeats=iterations,
                            random_state=0, n_jobs=-1, scoring="roc_auc")
    sorted_idx = result_1.importances_mean.argsort()
    
    novelty_variables = [i for i in range(0,12)]
    novelty_sorted_idx = []
    for entry in sorted_idx:
        if entry in novelty_variables:
            novelty_sorted_idx.append(entry)
    
    fig, ax = plt.subplots(figsize=[12*cm,7.5*cm])
    ax.boxplot(result_1.importances[novelty_sorted_idx].T,
               vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
    ax.grid()
    ax.set(xlabel='\u0394 decrease in ROC AUC score')
    ax.locator_params(tight=True, nbins=6, axis="x")
    
    fig.tight_layout()
    #fig.savefig("{0}_Permutation_Economic_Importance.pdf".format(model_name), dpi=1600)
    #fig.savefig("{0}_Permutation_Economic_Importance.png".format(model_name), dpi=1600)
    #fig.savefig("{0}_Permutation_Economic_Importance.eps".format(model_name), format="eps", dpi=1600)
    fig.savefig("{0}_Permutation_Economic_Importance.svg".format(model_name), format="svg", dpi=1600)
    if train:
        #test and train beside each other
        result_2 = permutation_importance(model, X_train, y_train, n_repeats=iterations,
                                random_state=0, n_jobs=-1, scoring="roc_auc")
        
        fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=[6.4,4], sharey=True)
        mpl.rcParams['font.size'] = 10
        ax1.boxplot(result_1.importances[novelty_sorted_idx].T,
                   vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
        ax1.grid()
        ax1.set(xlabel='\u0394 decrease in ROC AUC score')
        ax1.set(title='Test Data')
        ax1.locator_params(tight=True, nbins=3, axis="x")
        
        ax2.boxplot(result_2.importances[novelty_sorted_idx].T,
                   vert=False, labels=[i.lower() for i in X_train.columns[novelty_sorted_idx]])
        ax2.grid()
        ax2.set(xlabel='\u0394 decrease in ROC AUC score')
        ax2.set(title='Training Data')
        ax2.locator_params(tight=True, nbins=3, axis="x")
        
        
        mpl.rcParams['font.size'] = 12
        fig.tight_layout()
        #fig.savefig("{0}_Permutation_Economic_Importance_test_train.pdf".format(model_name), dpi=1600)
        #fig.savefig("{0}_Permutation_Economic_Importance_test_train.png".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Economic_Importance_test_train.eps".format(model_name), format="eps", dpi=1600)
    print("Economic Importance finished")
    os.chdir("../")
    plt.show()
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
 
permut_tech(iterations = 500, train = False, years = "7_years", top_x = "top_10")
permut_econ(iterations = 500, train = False, top_x = "top_10")

permut_tech(iterations = 500, train = False, years = "5_years", top_x = "top_10")
permut_tech(iterations = 500, train = False, years = "10_years", top_x = "top_10")
permut_tech(iterations = 500, train = False, years = "7_years", top_x = "top_1")
permut_tech(iterations = 500, train = False, years = "5_years", top_x = "top_1")
permut_tech(iterations = 500, train = False, years = "10_years", top_x = "top_1")
permut_econ(iterations = 500, train = False, top_x = "top_1")
