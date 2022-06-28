def pdp_tech(iterations = 500, years = "7_years", top_x = "top_10", var_name1 = "MEAN_PAT_AGE", var_name2 = "RADICALNESS", var_name3 = "NEW_UNIGRAMS"):
    """
                1-WAY PARTIAL DEPENDENCE PLOTS OF TECHNOLOGICAL IMPORTANCE      
    """    
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from time import time
    import pandas as pd
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os
    import copy
    import numpy as np
    import statistics
    np.random.seed(seed=0)
    import seaborn as sns
    sns.set_palette(['#000000'], n_colors=100)
    model_name = "MLP"
    cm = 1/2.54
    mpl.rcParams['font.size'] = 10
    mpl.rcParams["font.family"] = "arial"
    fig, axes = plt.subplots(figsize=[12*cm,7.5*cm])
    np.random.seed(seed=0)
    t0 =time()
    data = pd.read_csv("Input_data_scaled_citations_further_controls.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 27 #87 oder 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["{0}_CIT_{1}".format("".join(top_x.upper().split("_")), "".join(years.upper().split("_")))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("citations_{0}_{1}".format(years, top_x))
    
    model =  joblib.load("{0}.joblib".format(model_name))
    categorical_var = False
    X_copy = copy.deepcopy(X)
    which_class=1
    
    var_name1 = var_name1
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name1].min(), X_copy[var_name1].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name1]))
    
    samples=np.random.choice(len(X_copy), size=iterations, replace=False)
    
    predictions1 = pd.DataFrame()
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name1]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions1=predictions1.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name1))
        counter += 1 
     
    predictions1_copy = copy.deepcopy(predictions1)
    X_copy = copy.deepcopy(X)
    var_name2 = var_name2
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name2].min(), X_copy[var_name2].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name2]))
    
    predictions2 = pd.DataFrame()
      
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name2]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions2=predictions2.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name2))
        counter += 1
        
    X_copy = copy.deepcopy(X)
    var_name3 = var_name3
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name3].min(), X_copy[var_name3].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name3]))
    
    predictions3 = pd.DataFrame()
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name3]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions3=predictions3.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name3))
        counter += 1        
        
    
    predictions1.rename(columns={"pred": "Log Odds"}, inplace = True)
    predictions2.rename(columns={"pred": "Log Odds"}, inplace = True)
    predictions3.rename(columns={"pred": "Log Odds"}, inplace = True)
    
    predictions1["Novelty Measure"] = [var_name1.lower() for i in range(len(predictions1))]
    predictions2["Novelty Measure"] = [var_name2.lower() for i in range(len(predictions2))]
    predictions3["Novelty Measure"] = [var_name3.lower() for i in range(len(predictions3))]
    
    predictions1 = predictions1.append(predictions2)
    predictions1 = predictions1.append(predictions3)
    
    #plotting
    sns.lineplot(ax = axes, data=predictions1, x="x_val", y="Log Odds", style="Novelty Measure")
    if top_x == "top_10":
        axes.set(ylim=(-2, 0))
    if top_x == "top_1":
        axes.set(ylim=(-4, 0))
    #ax.set(ylabel=r'Log10(Odds(P))', xlabel=None)
    axes.set(ylabel=r'$Log_{10}$ Probability', xlabel=None)
    
    axes.locator_params(axis="y", nbins=4)
    axes.locator_params(axis="x", nbins=5)
    #ax.set(title='Partial Dependence Plots (Technological Value)')
    axes.grid()
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    #fig.savefig("{0}_PDP_Technological_Importance.pdf".format(model_name), dpi=1600)
    #plt.savefig("{0}_PDP_Technological_Importance_confidence_interval_log10_probability.png".format(model_name), dpi=1600)
    plt.savefig("{0}_PDP_Technological_Importance_confidence_interval_log10_probability.svg".format(model_name), format="svg", dpi=1600)
    #fig.savefig("{0}_PDP_Technological_Importance.eps".format(model_name), format="eps", dpi=1600)
    os.chdir("../")
    from IPython import get_ipython
    get_ipython().magic('reset -sf')

def pdp_econ(iterations = 500, top_x = "top_10", var_name1 = "MEAN_PAT_AGE", var_name2 = "RADICALNESS", var_name3 = "NEW_UNIGRAMS"):
    
    """
    
                1-WAY PARTIAL DEPENDENCE PLOTS OF ECONOMICAL IMPORTANCE
                
    """
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from time import time
    import pandas as pd
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os
    import copy
    import numpy as np
    import statistics
    np.random.seed(seed=0)
    import seaborn as sns
    sns.set_palette(['#000000'], n_colors=100)
    model_name = "MLP"
    cm = 1/2.54
    mpl.rcParams['font.size'] = 10
    mpl.rcParams["font.family"] = "arial"
    fig, axes = plt.subplots(figsize=[12*cm,7.5*cm])
    np.random.seed(seed=0)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import copy
    import numpy as np
    
    t0 =time()
    data = pd.read_csv("Input_data_scaled_KPSS_further_controls.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 27
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["{0}_KPSS".format("".join(top_x.upper().split("_")))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("kpss_{0}".format(top_x))
    
    model =  joblib.load("{0}.joblib".format(model_name))
    categorical_var = False
    X_copy = copy.deepcopy(X)
    which_class=1
    
    var_name1 = var_name1
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name1].min(), X_copy[var_name1].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name1]))
    
    samples=np.random.choice(len(X_copy), size=iterations, replace=False)
    
    predictions1 = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name1]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions1=predictions1.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name1))
        counter += 1
        
    predictions1_copy = copy.deepcopy(predictions1)
    
    X_copy = copy.deepcopy(X)
    var_name2 = var_name2
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name2].min(), X_copy[var_name2].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name2]))
    
    predictions2 = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name2]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions2=predictions2.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name2))
        counter += 1
        
    X_copy = copy.deepcopy(X)
    var_name3 = var_name3
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name3].min(), X_copy[var_name3].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name3]))
    
    predictions3 = pd.DataFrame()
      
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name3]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions3=predictions3.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        #sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name3))
        counter += 1
    
    
    predictions1.rename(columns={"pred": "Log Odds"}, inplace = True)
    predictions2.rename(columns={"pred": "Log Odds"}, inplace = True)
    predictions3.rename(columns={"pred": "Log Odds"}, inplace = True)
    
    
    predictions1["Novelty Measure"] = [var_name1.lower() for i in range(len(predictions1))]
    predictions2["Novelty Measure"] = [var_name2.lower() for i in range(len(predictions2))]
    predictions3["Novelty Measure"] = [var_name3.lower() for i in range(len(predictions3))]
    
    
    
    predictions1 = predictions1.append(predictions2)
    predictions1 = predictions1.append(predictions3)
    
    
    #plotting
    sns.lineplot(ax = axes,data=predictions1, x="x_val", y="Log Odds", style="Novelty Measure")
    if top_x == "top_10":
        axes.set(ylim=(-2, 0))
    if top_x == "top_1":
        axes.set(ylim=(-4, 0))
    
    axes.locator_params(axis="y", nbins=4)
    axes.locator_params(axis="x", nbins=5)
    #ax.set(ylabel=r'Log10(Odds(P))', xlabel=None)
    axes.set(ylabel=r'$Log_{10}$ Probability', xlabel=None)
    #ax.set(title='Partial Dependence Plots (Economical Value)')
    axes.grid()
    #ax.legend(loc="upper right")
    plt.legend()
    plt.tight_layout()
    
    #fig.savefig("{0}_PDP_Technological_Importance.pdf".format(model_name), dpi=1600)
    #plt.savefig("{0}_PDP_Economical_Importance_confidence_interval_log10_probability.png".format(model_name), dpi=1600)
    plt.savefig("{0}_PDP_Economical_Importance_confidence_interval_log10_probability.svg".format(model_name), format="svg", dpi=1600)
    #fig.savefig("{0}_PDP_Technological_Importance.eps".format(model_name), format="eps", dpi=1600)
    os.chdir("../")
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
#pdp_tech(years = "7_years", top_x = "top_10", var_name1 = "MEAN_PAT_AGE", var_name2 = "RADICALNESS", var_name3 = "NEW_TRIGRAMS")
pdp_econ(top_x = "top_10", var_name1 = "MEAN_PAT_AGE", var_name2 = "RADICALNESS", var_name3 = "NEW_TRIGRAMS")