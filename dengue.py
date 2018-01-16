import math

import wsnh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from scipy.stats import boxcox

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression


@wsnh.cmd()
def preprocess():
    print 'Processing data ...'
    train_data = pd.read_csv('data/dengue_features_train.csv')
    label_data = pd.read_csv('data/dengue_labels_train.csv')
    test_data = pd.read_csv('data/dengue_features_test.csv')

    train_data['total_cases'] = label_data['total_cases']
    all_data = pd.concat([train_data, test_data])
    
    # add month feature
    all_data['month'] = all_data['week_start_date'].apply(lambda x: int(x.split('-')[1]))
    
    # drop week_start_date and precip, which is same as reanalysis_sat_precip_amt_mm
    all_data = all_data.drop(['precipitation_amt_mm', 'week_start_date'], axis = 1)
    
    for city in ('sj', 'iq'):
        city_all = all_data[all_data.city == city].copy()
        
        train = city_all.head(len(train_data[train_data.city == city])).drop('city', axis = 1).reset_index(drop = True)
        test = city_all.tail(len(test_data[test_data.city == city])).drop(['total_cases', 'city'], axis = 1).reset_index(drop = True)
        
        ep = epidemic(train[['weekofyear', 'total_cases']])
        train['epidemic'] = ep
        train['epidemic_duration'] = ep_convert(ep)
        
        train = train.ffill()
        test = test.ffill()
        
        print 'Saving', city, 'data ...'
        train.to_csv('data/' + city + '_train.csv', index = False)
        test.to_csv('data/' + city + '_test.csv', index = False)
    
@wsnh.cmd()
def adjust_data(city = 'sj'):
    print 'Adjusting', city, '...'
    train, test = get_data(city)
    
    train['bc_total_cases'] = boxcox(train['total_cases'] + .00001)[0]
    
    train_keep = ['year', 'month', 'weekofyear', 'epidemic', 'total_cases', 'epidemic_duration', 'bc_total_cases']
    test_keep = ['year', 'month', 'weekofyear']

    train_ep_adj = train[train_keep].copy()
    test_ep_adj = test[test_keep].copy()
    
    diff = 0
    window = 0
    for col in train.drop(train_keep, axis = 1).columns:
        best = best_corr(train[train.year != 1994][col], train[train.year != 1994]['epidemic'])
        train_ep_adj[col] = best['method'](train[col])
        test_ep_adj[col] = best['method'](pd.concat([train[col], test[col]])).tail(len(test))
        
        if best['diff'] > diff:
            diff = best['diff']
            
        if best['window'] > window:
            window = best['window']
        
        print col
        print best
        print
        
    # omit beginning if diff is greater than 0
    train_ep_adj = train_ep_adj.tail(len(train_ep_adj) - (diff + window))
        
    print 'Saving epidemic data ...'
    train_ep_adj.to_csv('data/' + city + '_train_ep_adj.csv', index = False)
    test_ep_adj.to_csv('data/' + city + '_test_ep_adj.csv', index = False)
        
    train_tc_adj = train[train_keep].copy()
    test_tc_adj = test[test_keep].copy()
    
    diff = 0
    window = 0
    for col in train.drop(train_keep, axis = 1).columns:
        best = best_corr(train[train.year != 1994][col], train[train.year != 1994]['bc_total_cases'], max_window = 26)
        train_tc_adj[col] = best['method'](train[col])
        test_tc_adj[col] = best['method'](pd.concat([train[col], test[col]])).tail(len(test))
        
        if best['diff'] > diff:
            diff = best['diff']
            
        if best['window'] > window:
            window = best['window']
        
        print col
        print best
        print
        
    train_tc_adj = train_tc_adj.tail(len(train_tc_adj) - (diff + window))
        
    print 'Saving cases data ...'
    train_tc_adj.to_csv('data/' + city + '_train_tc_adj.csv', index = False)
    test_tc_adj.to_csv('data/' + city + '_test_tc_adj.csv', index = False)
        
    
@wsnh.cmd()
def score_class(city = 'sj'):
    df_train, _ = get_data(city, '_ep_adj')
    
    models = get_models('class', city)
    fields = get_fields('class', city)
    
    for name, model in models.items():
        train = df_train[fields].copy()
        
        if(name == 'knn' or name == 'svm'):
            train = scale(train)

        scores = np.mean(cross_val_score(model, 
            train,
            df_train['epidemic'],
            cv = 3, 
            scoring = 'f1',
            n_jobs = -1))
        
        print name
        print scores
        
@wsnh.cmd()
def plot(city = 'sj'):
    train, test = get_data(city, suff = '_ep_adj')
    
    df = pd.concat([train, test])
    
    for feature in df.drop(['year', 'month', 'weekofyear', 'total_cases', 'epidemic'], axis = 1).columns:
        df[feature].plot(title = feature, use_index = False)

        df['total_cases'].plot(secondary_y = True)
        train['epidemic'].plot(use_index = False, secondary_y = True)
        for j in [i * 52 for i in range(23)]:
            plt.axvline(52 - df['weekofyear'].iloc[0] + j)

        plt.show()
   
@wsnh.cmd()
def ep_predict(city = 'sj'):
    df_train, df_test = get_data(city, '_ep_adj')
    
    print 'Predicting epidemics for', city, '...'
    
    models = get_models('class', city)
    fields = get_fields('class', city)
    
    preds = []
    for name, model in models.items():
        print name
        train = df_train[fields].copy()
        test = df_test[fields].copy()
        
        if(name in ['knn', 'svm']):
            train, test = scale(train, test)
        
        fit = model.fit(train, df_train['epidemic'])
        pred = fit.predict(test)
        
        preds.append(pred)
        
    # simple majority
    ep = [1 if sum(pred) >= math.floor(len(models) / 2) + 1 else 0 for pred in zip(*preds)]
    
    print 'Writing csv ...'
    ep_df = pd.DataFrame()
    ep_df['epidemic'] = ep 
    ep_df['epidemic_duration'] = ep_convert(ep)
    
    ep_df.to_csv('data/' + city + '_ep_predict.csv', index = False)
        
def get_data(city = 'sj', suff = '', ep = False, tc = False):
    train = pd.read_csv('data/' + city + '_train' + suff + '.csv')
    test = pd.read_csv('data/' + city + '_test' + suff + '.csv')
    
    if ep:
        test_ep = pd.read_csv('data/' + city + '_ep_predict.csv')
        test = test.join(test_ep)
        
    if tc:
        test_tc = pd.read_csv('data/' + city + '_tc_predict.csv')
        test = test.join(test_tc)
    
    return train, test
    
def get_models(for_what, city):
    if for_what == 'class':
        if city == 'sj':
            knn = KNeighborsClassifier(10, weights = 'distance')
            svm = SVC(C = 1000, gamma = .05)
            rf = RandomForestClassifier(500, max_features = None, n_jobs = -1)
            gb = GradientBoostingClassifier(max_depth = 10, max_features = None)
            et = ExtraTreesClassifier(1000, max_features = None, n_jobs = -1)

            models = {
                'knn': knn,
                'svm': svm,
                'random forest': rf,
                'gradient boost': gb,
                'extra trees': et
            }
            
        elif city == 'iq':
            knn = KNeighborsClassifier(5, weights = 'distance')
            svm = SVC(C = 1000, gamma = .01)
            rf = RandomForestClassifier(1000, max_features = None, n_jobs = -1)
            ada = AdaBoostClassifier()
            gb = GradientBoostingClassifier(max_depth = 10, max_features = None)
            et = ExtraTreesClassifier(1000, max_features = None, n_jobs = -1)

            models = {
                'knn': knn,
                #'svm': svm,
                'ada': ada,
                'random forest': rf,
                #'gradient boost': gb,
                'extra trees': et
            }
        
    elif for_what == 'regression':
        if city == 'sj':
            svm = SVR(C = 1000, gamma = .01)
            rf = RandomForestRegressor(n_estimators = 500, n_jobs = -1)
            ada = AdaBoostRegressor()
            kr = KernelRidge()
            r = Ridge()
            dt = DecisionTreeRegressor()
            l = Lasso()
            knn = KNeighborsRegressor(7, weights = 'distance')
            gb = GradientBoostingRegressor(n_estimators = 1000, max_depth = 10)
            en = ElasticNet(.01)
            lr = LinearRegression()

            models = {
                #'knn': knn,
                #'svm': svm,
                'linear': lr,
                'ridge': r,
                #'kernel ridge': kr,
                #'random forest': rf,
                #'lasso': l,
                'elastic': en,
                #'gradient boost': gb,
                #'ada': ada
            }
            
        elif city == 'iq':
            svm = SVR(C = 1000, gamma = .001)
            rf = RandomForestRegressor(1000, max_features = None, n_jobs = -1)
            ada = AdaBoostRegressor(n_estimators = 500)
            kr = KernelRidge()
            r = Ridge()
            dt = DecisionTreeRegressor()
            l = Lasso(alpha = .0005)
            knn = KNeighborsRegressor(7, weights = 'distance')
            gb = GradientBoostingRegressor(n_estimators = 1000, max_depth = 10)
            en = ElasticNet(alpha = .0005)
            lr = LinearRegression()

            models = {
                'svm': svm,
                'linear': lr,
                'ridge': r,
                #'ada': ada,
                'elastic': en
            }
        
    return models
    
def get_fields(for_what, city = 'sj'):
    if city == 'sj':
        if for_what == 'class':
            fields = ['station_min_temp_c', 'station_max_temp_c']
            
        elif for_what == 'regression':
           fields = ['weekofyear', 'month', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'station_min_temp_c', 'station_max_temp_c', 'epidemic', 'epidemic_duration']
    
    elif city == 'iq':
        if for_what == 'class':
            fields = ['ndvi_ne', 'ndvi_sw', 'ndvi_se', 'station_max_temp_c']
            
        elif for_what == 'regression':
            fields = ['weekofyear', 'month', 'ndvi_sw', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_diur_temp_rng_c', 'epidemic', 'epidemic_duration']

    return fields

def best_corr(x, y, max_window = 52, method = 'pearson', diffs = (0, 1, 52)):
    windows = range(1, max_window, 1)
    quants = np.arange(.05, 1, .05)

    metrics = {
        'mean': lambda z: z.mean(),
        'min': lambda z: z.min(), 
        'max': lambda z: z.max(), 
        'sum': lambda z: z.sum(), 
        'var': lambda z: z.var(),
        'range': lambda z: z.apply(lambda a: a.max() - a.min()), 
        'prod': lambda z: z.apply(lambda a: a.prod()), 
        'first': lambda z: z.apply(lambda a: a[0]),
        'quantile': lambda z, q: z.quantile(q),
        'increase': lambda z: z.apply(lambda a: 1 if pd.Series(a).is_monotonic_increasing else 0),
        'decrease': lambda z: z.apply(lambda a: 1 if pd.Series(a).is_monotonic_decreasing else 0),
        'change': lambda z: z.apply(lambda a: a[len(a) - 1] - a[0])
    }
    
    best = {}
    best['corr'] = 0
    best['diff'] = 0
    
    for diff in diffs:
        if diff:
            series = x.diff(diff).copy()
        else:
            series = x.copy()
            
        for w in windows:
            for name, lamb in metrics.iteritems():
                if name == 'quantile':
                    best_quant = 0
                    for quant in quants:
                        cur_corr = lamb(series.rolling(w, min_periods = 1), quant).corr(y.tail(len(x) - w), method = method)
                        
                        if abs(cur_corr) > abs(new_corr):
                            new_corr = cur_corr
                            best_quant = quant
                else:
                    new_corr = lamb(series.rolling(w, min_periods = 1)).tail(len(x) - w).corr(y.tail(len(x) - w), method = method)
                    
                if abs(new_corr) > abs(best['corr']):
                    best['corr'] = new_corr
                    best['metric'] = name
                    best['window'] = w
                    best['diff'] = diff
                    
                    if name == 'quantile':
                        best['quant'] = best_quant
                    else:
                        best['quant'] = None
                    
    if best['diff']:
        if best['metric'] == 'quantile':
            best['method'] = lambda z: metrics[best['metric']](z.diff(best['diff']).rolling(best['window'], min_periods = 1), best['quant'])
        else:
            best['method'] = lambda z: metrics[best['metric']](z.diff(best['diff']).rolling(best['window'], min_periods = 1))
    else:
        if best['metric'] == 'quantile':
            best['method'] = lambda z: metrics[best['metric']](z.rolling(best['window'], min_periods = 1), best['quant'])
        else:
            best['method'] = lambda z: metrics[best['metric']](z.rolling(best['window'], min_periods = 1))
                
    return best
    
def epidemic(df):
    weekly_thresholds = df.groupby('weekofyear')['total_cases'].quantile(.75).to_dict()
    ep = [1 if r['total_cases'] > weekly_thresholds[r['weekofyear']] else 0 for i, r in df.iterrows()]
    
    return ep
    
def ep_convert(ep):
    counts = []
    counter = 0
    for i in ep:
        if i:
            counter += 1
        else:
            counts.append(counter)
            counter = 0
        
    counts = [c for c in counts if c]
    
    count_ep = []
    cur_value = None
    for i in ep:
        if i:
            if not cur_value:
                cur_value = counts.pop(0)

            count_ep.append(cur_value)
        else:
            if cur_value:
                cur_value = None
            count_ep.append(0)
        
    return count_ep    
    
def scale(df, df2 = None):
    if df2 is not None:
        fit = StandardScaler().fit(df)
        scaled = pd.DataFrame(fit.transform(df), columns = df.columns)
        scaled2 = pd.DataFrame(fit.transform(df2), columns = df2.columns)
        return scaled, scaled2
    else:
        scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns = df.columns)
        return scaled
        
def reverse_boxcox(value, series):
    _, l = boxcox(series + .00001)
    return np.exp(np.log(l * value + 1) / l) 
 
@wsnh.cmd()
def score_regression(city = 'sj', response = 'bc_total_cases', lags = 6, index_per = .66):
    df, _ = get_data(city, '_tc_adj', ep = True)
    
    models = get_models('regression', city)
    fields = get_fields('regression', city)

    idx = int(math.floor(len(df) * index_per))

    preds = []
    for name, model in models.items():
        print 'Scoring', name, '...'
        
        train = df[fields][:idx]
        test = df[fields][idx:].reset_index(drop = True)
        
        if(name in ['knn', 'svm', 'lasso', 'elastic', 'ridge', 'kernel ridge']):
            train, test = scale(train, test)
            
        train[response] = df[response][:idx]
        
        for i in range(len(test)):
            train = train.append(test.iloc[i])
            
            for j in range(1, lags + 1):
                if(not name in ['knn', 'svm', 'lasso', 'elastic', 'ridge', 'kernel ridge']):
                    train[response + str(j)] = train[response].shift(j)
                else:
                    train[response + str(j)] = np.append(np.repeat(np.nan, j), StandardScaler().fit_transform(train[response].shift(j)[j:].values.reshape(-1, 1)))
            
            fit = model.fit(train[:-1].drop(response, axis = 1).iloc[lags:], train[:-1][response].iloc[lags:])
            pred = fit.predict(train[-1:].drop(response, axis = 1))

            train.iloc[-1, train.columns.get_loc(response)] = pred
            
        train[response] = [reverse_boxcox(c, df['total_cases']) for c in train[response]]
            
        print 'Score:', np.mean(map(abs, np.subtract(train[response][idx:], df['total_cases'][idx:])))
            
        preds.append(map(int, map(round, train[response][idx:])))
        
    print 'Averaging ...'
    avg = map(int, map(round, [np.mean(pred) for pred in zip(*preds)]))
    print 'Score:', np.mean(map(int, map(abs, np.subtract(avg, df['total_cases'][idx:]))))
    
    '''for pred in zip(*preds):
        print pred'''
        
    plt.xticks([i for i, w in enumerate(df['weekofyear'][idx:]) if w == 1], range(df['year'][idx], df['year'][len(df) - 1]))
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    tc = plt.plot(df['total_cases'][idx:].values, linewidth = 1.2)
    pr = plt.plot(avg, linewidth = 1.2, color = 'r')
    plt.legend(['Actual', 'Predicted'])
    plt.show()
    
    #train['total_cases'].plot(use_index = False)
    #plt.show()

@wsnh.cmd()
def tc_predict(city = 'sj', lags = 6, response = 'bc_total_cases'):
    print 'Predicting total cases for', city, '...'

    df_train, df_test = get_data(city, '_tc_adj', ep = True)
    
    models = get_models('regression', city)
    fields = get_fields('regression', city)

    preds = []
    for name, model in models.items():
        print 'Predicting', name, '...'
        
        train = df_train[fields].copy()
        test = df_test[fields].copy()
        
        if(name in ['knn', 'svm', 'lasso', 'elastic', 'ridge', 'kernel ridge']):
            train, test = scale(train, test)
            
        train[response] = df_train[response]
        
        for i in range(len(test)):
            train = train.append(test.iloc[i])
            
            for j in range(1, lags + 1):
                if(not name in ['knn', 'svm', 'lasso', 'elastic', 'ridge', 'kernel ridge']):
                    train[response + str(j)] = train[response].shift(j)
                else:
                    train[response + str(j)] = np.append(np.repeat(np.nan, j), StandardScaler().fit_transform(train[response].shift(j)[j:].values.reshape(-1, 1)))
            
            fit = model.fit(train[:-1].drop(response, axis = 1).iloc[lags:], train[:-1][response].iloc[lags:])
            pred = fit.predict(train[-1:].drop(response, axis = 1))
            
            train.iloc[-1, train.columns.get_loc(response)] = pred
            
        preds.append(map(int, map(round, [reverse_boxcox(p, df_train['total_cases']) for p in train[response].tail(len(df_test))])))
        
    for pred in zip(*preds):
        print pred
        
    print 'Averaging ...'
    avg = map(int, map(round, [np.mean(pred) for pred in zip(*preds)]))
    
    print 'Saving csv ...'
    pd.DataFrame({'total_cases': avg}).to_csv('data/' + city + '_tc_predict.csv', index = False)
        
@wsnh.cmd()
def plot_forecast(city = 'sj'):
    train, test = get_data(city, '_tc_adj', tc = True)
    
    all_data = pd.concat([train, test]).reset_index(drop = True)
    
    all_data['total_cases'].plot()
    all_data['total_cases'].tail(len(test)).plot()
    
    for j in [i * 52 for i in range(23)]:
        plt.axvline(52 - all_data['weekofyear'].iloc[0] + j)
        
    plt.show()
    
@wsnh.cmd()
def corr_map(city = 'sj', adj = False):
    sj_train, sj_test, iq_train, iq_test = get_data(adj = adj)

    if city == 'sj':
        x = sj_train.drop(['year', 'epidemic', 'epidemic_duration'], axis = 1)
    elif city == 'iq':
        x = iq_train.drop(['year', 'epidemic', 'epidemic_duration'], axis = 1)
        
    corr = x.corr()
    
    labels = [' '.join([l[:4] for l in label.split('_')][:3]) for label in x.columns]

    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.subplots(figsize = (10, 5))

    cmap = sns.diverging_palette(220, 75, 75, as_cmap = True)

    heatmap = sns.heatmap(corr, mask = mask, cmap = cmap, center = 0, square = True, vmax = 1, vmin = -1, linewidths = .5, xticklabels = labels, yticklabels = labels)
    heatmap.get_figure().savefig('sj_corr.png', bbox_inches = 'tight', pad_inches = .2)
    
@wsnh.cmd()
def scatter(var, var2 = None, city = 'sj', adj = False):
    train, test = get_data(city, '_tc_adj', tc = True)
    
    from pandas.plotting import autocorrelation_plot
    
    #autocorrelation_plot(iq_train['total_cases'])
    plt.plot(iq_train['total_cases'])
    plt.show()
    exit()
    
    if var2 is not None:
        if(city == 'sj'):
            plt.plot(sj_train[var], sj_train[var2], 'bo', alpha = .5)
        elif(city == 'iq'):
            plt.plot(iq_train[var], iq_train[var2], 'bo', alpha = .5)
            
    else:
        if(city == 'sj'):
            plt.xlabel('Year')
            plt.ylabel('Total number of cases')
            plt.xticks([i for i, w in enumerate(sj_train['weekofyear']) if w == 1], [str(i)[2:] for i in range(1991, 2009)])
            plt.plot(sj_train[var])
        
    plt.savefig(var + '.png')
    
@wsnh.cmd()
def submit():
    _, sj_test = get_data('sj', tc = True)
    _, iq_test = get_data('iq', tc = True)
    
    sj_test['city'] = ['sj'] * len(sj_test)
    iq_test['city'] = ['iq'] * len(iq_test)
    
    all_data = pd.concat([sj_test, iq_test])
    
    all_data = all_data[['city', 'year', 'weekofyear', 'total_cases']]
    
    all_data.to_csv('submit/submission.csv', index = False)
    
@wsnh.cmd()
def test():
    train, test = get_data('sj')

    '''for col in train.drop(['year', 'month', 'weekofyear', 'epidemic', 'epidemic_duration', 'total_cases', 'ndvi_ne', 'ndvi_nw', 'ndvi_sw', 'ndvi_se'], axis = 1).columns:
        best = best_corr(train[train.epidemic > 0][col], pd.Series(boxcox(train[train.epidemic > 0]['total_cases'] + .00001)[0]))
 
        print col
        print best
        print
        
    exit()'''

    
    
    all_data = pd.concat([train, test]).reset_index(drop = True)
    
    
    all_data['station_max_temp_c'].diff(52).rolling(50).quantile(.15).plot()
    pd.Series(boxcox(train['total_cases'] + .00001)[0]).plot(secondary_y = True)
    
    for j in [i * 52 for i in range(23)]:
        plt.axvline(52 - all_data['weekofyear'].iloc[0] + j)
    
    plt.show()
    
    exit()
    
    
    '''all_data = pd.concat([train, test]).reset_index(drop = True)
    
    
    ### spearman
    #all_data['station_min_temp_c'].diff(52).rolling(49, min_periods = 1).quantile(.1).plot()
    #all_data['station_max_temp_c'].diff(1).rolling(51, min_periods = 1).quantile(.35).plot()
    #all_data['reanalysis_tdtr_k'].diff(1).rolling(51, min_periods = 1).quantile(.7).plot()
    #all_data['station_precip_mm'].diff(1).rolling(51, min_periods = 1).quantile(.6).plot()
    
    ### pearson
    #all_data['station_min_temp_c'].diff(52).rolling(42, min_periods = 1).sum().plot()
    #all_data['station_max_temp_c'].diff(1).rolling(51, min_periods = 1).quantile(.35).plot()
    #all_data['reanalysis_sat_precip_amt_mm'].rolling(51, min_periods = 1).sum().plot()
    
    ### pearson, epidemic as 1
    #all_data['station_max_temp_c'].diff(1).rolling(51, min_periods = 1).quantile(.35).plot()
    #all_data['station_min_temp_c'].diff(52).rolling(45, min_periods = 1).mean().plot()
    #all_data['station_avg_temp_c'].diff(52).rolling(40, min_periods = 1).quantile(.2).plot()
    #all_data['reanalysis_specific_humidity_g_per_kg'].diff(52).rolling(36, min_periods = 1).mean().plot()
    
    pd.Series(np.divide(1, all_data['reanalysis_sat_precip_amt_mm'] + .00001)).rolling(51).apply(lambda x: x.max() - x.min()).plot()
    
    all_data['total_cases'].plot(secondary_y = True)
    
    for j in [i * 52 for i in range(23)]:
        plt.axvline(52 - all_data['weekofyear'].iloc[0] + j)
        
    plt.show()
    
    exit()'''
    
    for col in train.drop(['year', 'month', 'weekofyear', 'total_cases', 'epidemic', 'epidemic_duration'], axis = 1).columns:
        best = best_corr(train[train.year != 1994][col], train[train.year != 1994]['epidemic'], method = 'spearman')
        
        print col
        print best
        print

wsnh.run()

