import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Smooting_methods import plot_co2

warnings.filterwarnings('ignore')
data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y.isnull().sum()
y = y.fillna(y.bfill())  ##sonr aki değeri aldık burada

y.plot(figsize=(15, 6))
plt.show()
train = y[:'1997-12-01']
len(train)  #478 ay

test = y['1998-01-01':]
len(test)  #48 ay

###############
#ARIMA(p,d,q)
############

arima_model = ARIMA(train, order=(1, 1, 1)).fit()
arima_model.summary()

y_pred = arima_model.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")

##############
#hiper oarametre opmizisayonu
##########################

##AIC BIC##

p = d = q =range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model_result = ARIMA(train, order).fit()
            aic = arima_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


best_params_aic = arima_optimizer_aic(train, pdq)

arima_model = ARIMA(train, best_params_aic).fit()
y_pred = arima_model.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")


##############
##SARIMA
##################

model=SARIMAX(train,order=(1,0,1),seasonal_order=(0,0,0,12))

sarima_model=model.fit()
y_pred_test=sarima_model.get_forecast(48)
y_pred=y_pred_test.predicted_mean

y_pred=pd.Series(y_pred, index=test.index)

plot_co2(train,test, y_pred, "SARIMA")


#############
#HİPERPARAMETRE OPTİMİZSAYONU
##################################
p=d=q=range(0, 4)
pdq=list(itertools.product(p, d, q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in list (itertools.product(p, d, q))]


def sarima_optimizer_aic(train,pdq,seasonal_pdq):
    best_aic,best_order,best_params = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model=SARIMAX(train,order=param,seasonal_order=param_seasonal)
                results=sarimax_model.fit()
                aic=results.aic
                if aic < best_aic:
                    best_aic,best_order,best_seasonal_order=aic,param,param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order,best_seasonal_order

best_order, best_seasonal_order=sarima_optimizer_aic(train, pdq, seasonal_pdq)

##############
#final model
############

model=SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order)
sarima_final_model=model.fit()
y_pred_test=sarima_final_model.get_forecast(48)
y_pred=pd.Series(y_pred, index=test.index)
plot_co2(train,test,y_pred,"SARIMA")


######################333
#MAE'ye göre hiperparatme optimizsayonu

p=d=q=range(0, 2)
pdq=list(itertools.product(p, d, q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in list (itertools.product(p, d, q))]

def sarima_optimizer_mae(train,pdq,seasonal_pdq):
    best_mae,best_order,best_seasonal_order=float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model=SARIMAX(train,order=param,seasonal_order=param_seasonal)
                sarima_model=model.fit()
                y_pred_test=sarima_model.get_forecast(48)
                y_pred=y_pred_test.predicted_mean
                mae=mean_absolute_error(y_pred, test)
                if mae < best_mae:
                    best_mae,best_order,best_seasonal_order=mae,param,param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param,param_seasonal,mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE: {}'.format(best_order, best_seasonal_order, best_mae))
    return best_order,best_seasonal_order


best_order,best_seasonal_order=sarima_optimizer_mae(train,pdq,seasonal_pdq)
model=SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order)
sarima_final_model=model.fit()
y_pred_test=sarima_final_model.get_forecast(48)
y_pred=y_pred_test.predicted_mean
y_pred=pd.Series(y_pred, index=test.index)

plot_co2(train,test,y_pred,"SARIMA With MAE")




########################3
##FINAL MODEL
###################33
###işlemler tamamlandıktan sonra tüm veriyle final model kurulur
##Başarısını nasıl değenldireceğiz (canlı testler yapılır)
## Yani mesela 6 aylık veri tahmin ettim ilk ay ile canlı testle beklerim sonuuc
##zamansal bağımlıklıklarda nhai modle kurulurb


model=SARIMAX(y,order=best_order,seasonal_order=best_seasonal_order)
sarima_final_model=model.fit()
feature_predict=sarima_final_model.get_forecast(steps=6)
feature_predict=feature_predict.predicted_mean
