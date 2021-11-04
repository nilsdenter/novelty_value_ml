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
import copy
import numpy as np
np.random.seed(seed=0)
model_name = "MLP"

def permut_tech(iterations = 500, train = False):
    """
                PERMUTATION IMPORTANCE OF TECHNOLOGICAL IMPORTANCE
    """
    
    mpl.rcParams['font.size'] = 12
    mpl.rcParams["font.family"] = "calibri"

    t0 =time()
    data = pd.read_csv("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_citations.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 20 #87 oder 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["TOP10_CIT_7YEARS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\7_years")
    
    model =  joblib.load("{0}.joblib".format(model_name))
    
    root = "N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\7_years\\"
    folder = root+"{0}".format(model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    
    result_1 = permutation_importance(model, X_test, y_test, n_repeats=iterations,
                            random_state=0, n_jobs=-1, scoring="roc_auc")
    sorted_idx = result_1.importances_mean.argsort()
    
    novelty_variables = [i for i in range(0,12)]
    novelty_sorted_idx = []
    for entry in sorted_idx:
        if entry in novelty_variables:
            novelty_sorted_idx.append(entry)
    
    fig, ax = plt.subplots(figsize=[6.4,4])
    ax.boxplot(result_1.importances[novelty_sorted_idx].T,
               vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
    ax.grid()
    ax.set(xlabel='\u0394 decrease in ROC AUC score', title='Permutation Importance (Technological Value)')
    ax.locator_params(tight=True, nbins=6, axis="x")
    
    fig.tight_layout()
    #fig.savefig("{0}_Permutation_Technological_Importance.pdf".format(model_name), dpi=1600)
    fig.savefig("{0}_Permutation_Technological_Importance.png".format(model_name), dpi=1600)
    fig.savefig("{0}_Permutation_Technological_Importance.eps".format(model_name), format="eps", dpi=1600)
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
        fig.suptitle('Permutation Importance (Technological Value)')
        fig.tight_layout()
        #fig.savefig("{0}_Permutation_Technological_Importance_test_train.pdf".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Technological_Importance_test_train.png".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Technological_Importance_test_train.eps".format(model_name), format="eps", dpi=1600)
    print("Technological Importance finished")
    plt.show()

def permut_econ(iterations = 500, train = False):
    
    """
                PERMUTATION IMPORTANCE OF ECONOMICAL IMPORTANCE          
    """

    t0 =time()
    data = pd.read_csv("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_KPSS.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["TOP10_KPSS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_10")
    
    model =  joblib.load("{0}.joblib".format(model_name))
    
    root = "N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_10\\"
    folder = root+"{0}".format(model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    
    result_1 = permutation_importance(model, X_test, y_test, n_repeats=iterations,
                            random_state=0, n_jobs=-1, scoring="roc_auc")
    sorted_idx = result_1.importances_mean.argsort()
    
    novelty_variables = [i for i in range(0,12)]
    novelty_sorted_idx = []
    for entry in sorted_idx:
        if entry in novelty_variables:
            novelty_sorted_idx.append(entry)
    
    fig, ax = plt.subplots(figsize=[6.4,4])
    ax.boxplot(result_1.importances[novelty_sorted_idx].T,
               vert=False, labels=[i.lower() for i in X_test.columns[novelty_sorted_idx]])
    ax.grid()
    ax.set(xlabel='\u0394 decrease in ROC AUC score', title='Permutation Importance (Economical Value)')
    ax.locator_params(tight=True, nbins=6, axis="x")
    
    fig.tight_layout()
    #fig.savefig("{0}_Permutation_Economical_Importance.pdf".format(model_name), dpi=1600)
    fig.savefig("{0}_Permutation_Economical_Importance.png".format(model_name), dpi=1600)
    fig.savefig("{0}_Permutation_Economical_Importance.eps".format(model_name), format="eps", dpi=1600)
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
        fig.suptitle('Permutation Importance (Economical Value)')
        fig.tight_layout()
        #fig.savefig("{0}_Permutation_Economical_Importance_test_train.pdf".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Economical_Importance_test_train.png".format(model_name), dpi=1600)
        fig.savefig("{0}_Permutation_Economical_Importance_test_train.eps".format(model_name), format="eps", dpi=1600)
    print("Economical Importance finished")
    plt.show()

def pdp_tech(iterations = 500, with_figure=True):
    
    """
            1-WAY PARTIAL DEPENDENCE PLOTS OF TECHNOLOGICAL IMPORTANCE
    """
    
    
    t0 =time()
    data = pd.read_csv("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_citations.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["TOP10_CIT_7YEARS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    
    os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\7_years")
    
    model =  joblib.load("{0}.joblib".format(model_name))
    categorical_var = False
    X_copy = copy.deepcopy(X)
    which_class=1
    
    root = "N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\7_years\\"
    folder = root+"{0}".format(model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    
    var_name1 = "MEAN_PAT_AGE"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name1].min(), X_copy[var_name1].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name1]))
    
    samples=np.random.choice(len(X_copy), size=iterations, replace=False)
    
    predictions = pd.DataFrame()
      
    counter = 1
    for sample in samples:
        for i in var_grid_vals:
            X_copy[var_name1]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
    if (counter%int(iterations/10))==0:
        print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name1))
    counter += 1
    
    preds_grouped_1 = predictions.groupby(['x_val']).mean().reset_index()
    
    #preds_grouped_1.to_excel("{0}_1wayPDP_Technological_Importance.xlsx".format(var_name1))
    
    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    from statistics import mean, stdev
    var_name_mean = mean(X_copy[var_name1])
    var_name_sd = stdev(X_copy[var_name1])
    
    mean_translation = []
    
    for index, row in preds_grouped_1.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_1["Mean"] = mean_translation
    preds_grouped_1.to_excel("{0}_1wayPDP_Technological_Importance.xlsx".format(var_name1))
    
    X_copy = copy.deepcopy(X)
    var_name2 = "RADICALNESS"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name2].min(), X_copy[var_name2].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name2]))
    
    predictions = pd.DataFrame()   
    
    counter = 1
    for sample in samples:
        #x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name2]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
            #sample_preds = predictions[predictions['sample']==sample]
    #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
    if (counter%int(iterations/10))==0:
        print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name2))
    counter += 1
    
    preds_grouped_2 = predictions.groupby(['x_val']).mean().reset_index()

    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    var_name_mean = mean(X_copy[var_name2])
    var_name_sd = stdev(X_copy[var_name2])
    predictions = pd.DataFrame()
    
    
    mean_translation = []
    
    for index, row in preds_grouped_2.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_2["Mean"] = mean_translation
    preds_grouped_2.to_excel("{0}_1wayPDP_Technological_Importance.xlsx".format(var_name2))

    X_copy = copy.deepcopy(X)
    var_name3 = "MEAN_SEM_DIST"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name3].min(), X_copy[var_name3].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name3]))
    
    predictions = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        #x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name3]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
            #sample_preds = predictions[predictions['sample']==sample]
    #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
    if (counter%int(iterations/10))==0:
        print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name3))
    counter += 1
    
    preds_grouped_3 = predictions.groupby(['x_val']).mean().reset_index()
    
    
    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    var_name_mean = mean(X_copy[var_name3])
    var_name_sd = stdev(X_copy[var_name3])
    
    
    mean_translation = []
    
    for index, row in preds_grouped_3.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_3["Mean"] = mean_translation
    preds_grouped_3.to_excel("{0}_1wayPDP_Technological_Importance.xlsx".format(var_name3))   
    
    if with_figure:
        
        X_copy = copy.deepcopy(X)
        #plotting
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from cycler import cycler
        mpl.rcParams['font.size'] = 12
        mpl.rcParams["font.family"] = "calibri"
        #custom_cycler = (cycler(color=sns.color_palette("Greys",3)) + cycler(linestyle=["-", "-", "--"]))
        
        fig, ax = plt.subplots(figsize=[6.4,4])
        
        ax.plot(preds_grouped_1.x_val, preds_grouped_1.pred, label=var_name1.lower(), color="black", linestyle="-")
        ax.plot(preds_grouped_2.x_val, preds_grouped_2.pred, label=var_name2.lower(), color="black", linestyle="--")
        ax.plot(preds_grouped_3.x_val, preds_grouped_3.pred, label=var_name3.lower(), color="black", linestyle="-.")
        ax.set_ylim(-2,0)
        ax.locator_params(axis="y", nbins=4)
        ax.locator_params(axis="x", nbins=5)
        
        #ax.set(ylabel=r'Log Odds (log $\frac{h_\theta(x)}{1-h_\theta(x)}$)')
        ax.set(ylabel=r'$Log_{10}$ Probability')
        ax.set(title='Partial Dependence Plots (Technological Value)')
        
        ax.grid()
        ax.legend(loc="upper right")
        
        
        fig.tight_layout()
        #fig.savefig("{0}_PDP_Technological_Importance.pdf".format(model_name), dpi=1600)
        fig.savefig("{0}_PDP_Technological_Importance.png".format(model_name), dpi=1600)
        fig.savefig("{0}_PDP_Technological_Importance.eps".format(model_name), format="eps", dpi=1600)
        plt.show()

def pdp_econ(iterations = 500, with_figure=True):
    
    """
                1-WAY PARTIAL DEPENDENCE PLOTS OF ECONOMICAL IMPORTANCE          
    """
    
    import copy
    import numpy as np
    
    
    np.random.seed(seed=0)
    t0 =time()
    data = pd.read_csv("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_KPSS.csv" , index_col=0)
    print("Data loaded in {0} seconds.".format(int(time()-t0)))
    number_columns = len(data.columns)
    number_iv = 20
    columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
    columns = data.columns[columns_iv]
    X = data.iloc[:,columns_iv]
    y = data["TOP10_KPSS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    
    
    os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_10")
    
    model =  joblib.load("{0}.joblib".format(model_name))
    categorical_var = False
    X_copy = copy.deepcopy(X)
    which_class=1
    
    root = "N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_10\\"
    folder = root+"{0}".format(model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    
    var_name1 = "MEAN_PAT_AGE"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name1].min(), X_copy[var_name1].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name1]))
    
    samples=np.random.choice(len(X_copy), size=iterations, replace=False)
    
    predictions = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name1]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name1))
        counter += 1
        
    preds_grouped_1 = predictions.groupby(['x_val']).mean().reset_index()
   
    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    from statistics import mean, stdev
    var_name_mean = mean(X_copy[var_name1])
    var_name_sd = stdev(X_copy[var_name1])
    
    mean_translation = []
    
    for index, row in preds_grouped_1.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_1["Mean"] = mean_translation
    preds_grouped_1.to_excel("{0}_1wayPDP_Economical_Importance.xlsx".format(var_name1))  
    
    X_copy = copy.deepcopy(X)
    var_name2 = "RADICALNESS"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name2].min(), X_copy[var_name2].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name2]))
    
    predictions = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name2]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name2))
        counter += 1
        
    preds_grouped_2 = predictions.groupby(['x_val']).mean().reset_index()

    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    from statistics import mean, stdev
    var_name_mean = mean(X_copy[var_name2])
    var_name_sd = stdev(X_copy[var_name2])
    
    mean_translation = []
    
    for index, row in preds_grouped_2.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_2["Mean"] = mean_translation
    preds_grouped_2.to_excel("{0}_1wayPDP_Economical_Importance.xlsx".format(var_name2))    

    X_copy = copy.deepcopy(X)
    var_name3 = "NEW_TRIGRAMS"
    #For the continuous variables that will be plotted, create 40-interval arrays.
    if categorical_var == False:
        var_grid_vals = np.linspace(X_copy[var_name3].min(), X_copy[var_name3].max(), num=200)
    #For the categorical variables that will be plotted, create array of the unique values
    if categorical_var == True:
        var_grid_vals = list(set(X_copy[var_name3]))
    
    predictions = pd.DataFrame()
      
    
    counter = 1
    for sample in samples:
        x_vals = list()
        for i in var_grid_vals:
            X_copy[var_name3]=i
            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]
            y_hat_log_odds = np.log10(y_hat)
            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)
        sample_preds = predictions[predictions['sample']==sample]
        #plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)
        if (counter%int(iterations/10))==0:
            print("Processed case {0} from {1} for {2}.".format(counter ,len(samples), var_name3))
        counter += 1
        
    preds_grouped_3 = predictions.groupby(['x_val']).mean().reset_index()
    
    X_copy = copy.deepcopy(X)
    #calculate probabilities for specific x values
    from statistics import mean, stdev
    var_name_mean = mean(X_copy[var_name3])
    var_name_sd = stdev(X_copy[var_name3])
    
    mean_translation = []
    
    for index, row in preds_grouped_3.iterrows():
        s = "Mean"
        x_value = row[0]
        sd_times = round((x_value-var_name_mean)/var_name_sd,3)
        if sd_times > 0:
            s += " +{0}".format(sd_times)
        elif sd_times < 0:
            s += " {0}".format(sd_times)
            
        mean_translation.append(s)
    
    preds_grouped_3["Mean"] = mean_translation
    preds_grouped_3.to_excel("{0}_1wayPDP_Economical_Importance.xlsx".format(var_name3))   

    if with_figure:
        
        X_copy = copy.deepcopy(X)
        #plotting
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from cycler import cycler
        mpl.rcParams['font.size'] = 12
        mpl.rcParams["font.family"] = "calibri"
        #custom_cycler = (cycler(color=sns.color_palette("Greys",3)) + cycler(linestyle=["-", "-", "--"]))
        
        fig, ax = plt.subplots(figsize=[6.4,4])
        
        ax.plot(preds_grouped_1.x_val, preds_grouped_1.pred, label=var_name1.lower(), color="black", linestyle="-")
        ax.plot(preds_grouped_2.x_val, preds_grouped_2.pred, label=var_name2.lower(), color="black", linestyle="--")
        ax.plot(preds_grouped_3.x_val, preds_grouped_3.pred, label=var_name3.lower(), color="black", linestyle="-.")
        ax.set_ylim(-2,0)
        ax.locator_params(axis="y", nbins=4)
        ax.locator_params(axis="x", nbins=5)
        
        
        
        #ax.set(ylabel=r'Log Odds (log $\frac{h_\theta(x)}{1-h_\theta(x)}$)')
        ax.set(ylabel=r'$Log_{10}$ Probability')
        ax.set(title='Partial Dependence Plots (Economical Value)')
        
        ax.grid()
        #ax.legend(loc="upper right")
        ax.legend()
        
        
        fig.tight_layout()
        #fig.savefig("{0}_PDP_Economical_Importance.pdf".format(model_name), dpi=1600)
        fig.savefig("{0}_PDP_Economical_Importance.png".format(model_name), dpi=1600)
        fig.savefig("{0}_PDP_Economical_Importance.eps".format(model_name), format="eps", dpi=1600)
        plt.show()
 
permut_tech(train=True)
permut_econ(train=True)    
pdp_tech(iterations = 500, with_figure=False)
pdp_econ(iterations = 500, with_figure=True)
