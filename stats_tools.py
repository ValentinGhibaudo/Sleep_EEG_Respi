import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt

def normality(df, predictor, outcome, show = False):
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)

    normalities = pg.normality(data = df , dv = outcome, group = predictor)['normal']
    
    if sum(normalities) == normalities.size:
        normal = True
    else:
        normal = False
    
    if show : 
        group1 = df[df[predictor] == groups[0]][outcome]
        group2 = df[df[predictor] == groups[1]][outcome]
        fig , axs = plt.subplots(ncols = 2)
        
        ax = axs[0]
        group1.plot(kind='density', ax = ax)
        ax.set_title('Group 1')
        ax.vlines(x = mean, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'r')
        ax.vlines(x = low, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'y')
        ax.vlines(x = high, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'y')

        ax = axs[1]
        group2.plot(kind='density', ax = ax)
        ax.set_title('Group 1')
        ax.vlines(x = mean, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'r')
        ax.vlines(x = low, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'y')
        ax.vlines(x = high, ymin = 0, ymax = 1, linestyles = 'dotted', color = 'y')

        plt.show()
    return normal

def homoscedasticity(df, predictor, outcome):
    homoscedasticity = pg.homoscedasticity(data = df, dv = outcome, group = predictor)['equal_var'].values[0]
    return homoscedasticity

def parametric(df, predictor, outcome):
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)
    
    normal = normality(df, predictor, outcome)
    homoscedastic = homoscedasticity(df, predictor, outcome)
    
    if normal and homoscedastic:
        parametricity = True
    else:
        parametricity = False
        
    return parametricity


def guidelines(df, predictor, outcome, design):
    
    parametricity = parametric(df, predictor, outcome)
        
    n_groups = len(list(set(df[predictor])))
    
    if parametricity:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'t-test_ind','post':None}
            elif design == 'within':
                tests = {'pre':'t-test_paired','post':None}
        else:
            if design == 'between':
                tests = {'pre':'anova','post':'pairwise_tukey'}
            elif design == 'within':
                tests = {'pre':'rm_anova','post':'pairwise_ttests_paired_paramTrue'}
    else:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'Mann-Whitney','post':None}
            elif design == 'within':
                tests = {'pre':'Wilcoxon','post':None}
        else:
            if design == 'between':
                tests = {'pre':'Kruskal','post':'pairwise_ttests_ind_paramFalse'}
            elif design == 'within':
                tests = {'pre':'friedman','post':'pairwise_ttests_paired_paramFalse'}
                
    return tests

def pg_compute_pre(df, predictor, outcome, test, subject=None, show = False):
    
    pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
    esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}
    
    if test == 't-test_ind':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=False)
        
    elif test == 't-test_paired':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=True)
        
    elif test == 'anova':
        res = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')
    
    elif test == 'rm_anova':
        res = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)
        
    elif test == 'Mann-Whitney':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.mwu(x, y)
        
    elif test == 'Wilcoxon':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.wilcoxon(x, y)
        
    elif test == 'Kruskal':
        res = pg.kruskal(data=df, dv=outcome, between=predictor)
        
    elif test == 'friedman':
        res = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)
    
    pval = res[pval_labels[test]].values[0]
    es_label = esize_labels[test]
    if es_label is None:
        es = None
    else:
        es = res[es_label].values[0]
    
    results = {'p':pval, 'es':es, 'es_label':es_label}
      
    return results

def get_stats_tests():
    
    ttest_ind = ['parametric', 'indep', 2, 't-test_ind' , 'NA']
    ttest_paired = ['parametric', 'paired', 2, 't-test_paired', 'NA']
    anova = ['parametric', 'indep', '3 ou +', 'anova', 'pairwise_tukey']
    rm_anova = ['parametric', 'paired', '3 ou +', 'rm_anova', 'pairwise_ttests_paired_paramTrue']
    mwu = ['non parametric', 'indep', 2, 'Mann-Whitney',  'NA']
    wilcox = ['non parametric', 'paired', 2, 'Wilcoxon', 'NA']
    kruskal = ['non parametric', 'indep', '3 ou +', 'Kruskal','pairwise_ttests_ind_paramFalse']
    friedman = ['non parametric', 'paired', '3 ou +', 'friedman', 'pairwise_ttests_paired_paramFalse']
    
    rows = [ttest_ind, ttest_paired, anova, rm_anova, mwu , wilcox, kruskal, friedman ]
    
    df=pd.DataFrame(rows , columns = ['parametricity','paired','samples','test','post_hoc'])
    df = df.set_index(['parametricity','paired','samples'])
    return df


        
def pg_compute_post_hoc(df, predictor, outcome, test, subject=None):
    
    if test == 'pairwise_tukey':
        res = pg.pairwise_tukey(data = df, dv=outcome, between=predictor)
        
    elif test == 'pairwise_ttests_paired_paramTrue':
        res = pg.pairwise_ttests(data = df, dv=outcome, within=predictor, subject=subject, parametric=True)
        
    elif test == 'pairwise_ttests_ind_paramFalse':
        res = pg.pairwise_ttests(data = df, dv=outcome, between=predictor, parametric=True)

    elif test == 'pairwise_ttests_paired_paramFalse':
        res = pg.pairwise_ttests(data = df, dv=outcome, within=predictor, subject=subject, parametric=False)
        
    return res


def auto_annotated_stats(df, predictor, outcome, test):
    
    x = predictor
    y = outcome

    order = list(set(df[predictor]))

    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(test=test, text_format='star', loc='inside')
    annotator.apply_and_annotate()
    # plt.show()

def custom_annotated_two(df, predictor, outcome, order, pval, ax=None, plot_mode = 'box'):
        
    stars = pval_stars(pval)
    
    x = predictor
    y = outcome

    order = order
    formatted_pvalues = [f"{stars}"]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw = 0.08)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

def custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=None, plot_mode = 'box'):
        
    pvalues = list(post_hoc['p-unc'])

    x = predictor
    y = outcome

    order = order
    pairs = [tuple(post_hoc.loc[i,['A','B']]) for i in range(post_hoc.shape[0])]
    formatted_pvalues = [f"{pval_stars(pval)}" for pval in pvalues]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw= 0.08)
    
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

             
def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001:
        stars = '***'
    elif pval < 0.0001:
        stars = '****'
    else:
        stars = 'ns'
    return stars
      

def auto_stats(df, predictor, outcome, ax=None, subject=None, design='within', mode = 'box'):

    if ax is None:
        fig, ax = plt.subplots()
        
    N = df[predictor].value_counts()[0]
    ngroups = len(list(df[predictor].unique()))
    
    tests = guidelines(df, predictor, outcome, design)
    pre_test = tests['pre']
    post_test = tests['post']
    results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
    pval = round(results['p'], 3)
    
    if not results['es'] is None:
        es = round(results['es'], 3)
    else:
        es = results['es']
    es_label = results['es_label']
    
    order = list(df[predictor].unique())
    
    if mode == 'box':
        if not post_test is None:
            post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)
            ax = custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=ax)
        else:
            ax = custom_annotated_two(df, predictor, outcome, order, pval, ax=ax)
     
    elif mode == 'distribution':
        ax = sns.histplot(df, x=outcome, hue = predictor, kde = True, ax=ax)
        
    ax.set_title(f'Effect of {predictor} on {outcome} \n N = {N} * {ngroups} \n {pre_test} : p-{pval}, {es_label} : {es}')
        
    return ax