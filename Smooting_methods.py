##Smooting Methods

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.tsa.api as smt
warnings.filterwarnings('ignore')


##Veri seti CO2

data=sm.datasets.co2.load_pandas()
y=data.data
##Bu veri seti haftalıktır
##Haftalık formattan aylığa çevireceğiz ayarlara göre gruop by alıp ortalama yapacağız

y=y['co2'].resample('MS').mean()

#6. ayda eksik bir değer var timeserieslerde eksik değerler ortalama  ilie doldurulmaz yerine
# kendinden önceki veya kkendinde sonr aki değerler veya o değerlerin prtalaması ile doldurulur

y.isnull().sum()
y=y.fillna(y.bfill()) ##sonr aki değeri aldık burada

y.plot(figsize=(15,6))
plt.show()


##############3
#### holdout
#########
##Veri setini bölme
train=y[:'1997-12-01']
len(train) #478 ay


test=y['1998-01-01':]
len(test) #48 ay

##Zaman serisi ile çaprazz doğrulamalar pek mantıklı değildir aşırı öğrenmeinin önüne geçmek için
#train ve test olarak ayırmak en iyi yöntemdir


##Zaman serisi  yapısal analizi

##durağanlık
##mevsimsellik
##trend


#Durağanlık Testi( Dickey-Fuller Test)

def is_stationary(y):
    # "HO: Non-stationary"
    # "H1: Stationary
    p_value=sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p_value: {round(p_value,3)})")
    else:
        print(F"Result:Non-Stationary (H0: non-stationary, p_value: {round(p_value,3)})")

is_stationary(y)

##Zaman Serisi Bileşenleri ve durağanlık Testi
def ts_decompse(y,model="additive",stationary=False):
    result=seasonal_decompose(y,model=model)
    fig,axes=plt.subplots(4,1,sharex=True,sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for "+ model+ " model")
    axes[0].plot(y,'k',label='Orijinal'+model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend,label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal,'g',label='Seasonality & Mean: '+str(round(result.seasonal.mean(),4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid,'r',label='Residuals & Mean: '+ str(round(result.resid.mean(),4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompse(y)

#################################
##Single Exponantional Smoothing
###############################

ses_model=SimpleExpSmoothing(train).fit(smoothing_level=0.5)
y_pred=ses_model.forecast(48) ##predict yerine forcasting kullanılır
mean_absolute_error(test,y_pred)

train["1985":].plot(title="Single Exponantial Smooting")
test.plot()
y_pred.plot()
plt.show()

def plot_co2(train,test,y_pred,title):
    mae=mean_absolute_error(test,y_pred)
    train["1985":].plot(legend=True,label="TRAIN",title=f"{title}, MAE: {round(mae,4)}")
    test.plot(legend=True,label="TEST",figsize=(6,4))
    y_pred.plot(legend=True,label="PREDICTION")
    plt.show()


plot_co2(train,test,y_pred,"Single Exponential Smooting")


ses_model.params

####
##SES Hiperparametre Optimizasyonu
################################33


def ses_optimizer(train,aplhas,step=48):
    best_alpha,best_mae=None,float("inf")
    for alpha in aplhas:
        ses_model=SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred=ses_model.forecast(step)
        mae=mean_absolute_error(test,y_pred)
        if mae < best_mae:
            best_alpha,best_mae=alpha,mae
        print("alpha:", round(alpha,2), "mae: ", round(mae,4))
    print("best_alpha: ",round(best_alpha,2),"best_mae: ",round(best_mae,4))
    return best_alpha,best_mae
alphas=np.arange(0.8,1,0.01)
best_alpha,best_mae=ses_optimizer(train,alphas)

ses__model=SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred=ses__model.forecast(48)

plot_co2(train,test,y_pred,"Single Exponential Smooting")







####DES Double Exponatinal Smooting
##################################
##DES
#y(t)=Level * Trend * Seasonality * Noise
#y(t)= Level + Trend + Seaonality + Noise


des_model=ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                     smoothing_trend=0.5)

y_pred=des_model.forecast(48)

plot_co2(train,test,y_pred,"Exponential Smooting")

########
#HİPERPARAMETRE OPTİMİZAASONU
#############################


def des_optimizer(train,aplhas,betas,step=48):
    best_alpha,best_beta,best_mae=None,None,float("inf")
    for alpha in aplhas:
        for beta in betas:
            des_model=ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha,
                                                                   smoothing_trend=beta)
            y_pred=des_model.forecast(step)
            mae=mean_absolute_error(test,y_pred)
            if mae < best_mae:
              best_alpha,best_beta,best_mae=alpha,beta,mae
            print("alpha:", round(alpha,2), "beta: ", round(beta,2),"mae: ", round(mae,4))
    print("best_alpha:", round(best_alpha,2),"best_beta:", round(best_beta,2),"best_mae: ", round(best_mae,4))
    return best_alpha,best_beta,best_mae


alphas=np.arange(0.01,1,0.10)
betas=np.arange(0.01,1,0.10)


best_alpha,best_beta,best_mae_=des_optimizer(train,alphas,betas)


final_des_model=ExponentialSmoothing(train,trend="mul").fit(smoothing_level=best_alpha,smoothing_trend=best_beta)

y_pred=final_des_model.forecast(48)

plot_co2(train,test,y_pred,"Exponential Smooting")



#########################

#TES=SES+DES+mEVSİMSELLİK
###########################

tes_model=ExponentialSmoothing(train,
                               trend="add",
                               seasonal="add",
                               seasonal_periods=12).fit(smoothing_level=0.5,
                                                        smoothing_slope=0.5,
                                                        smoothing_seasonal=0.5)
y_pred=tes_model.forecast(48)

plot_co2(train,test,y_pred,"Exponential Smooting for Holt-Winters")
alphas=betas=gammas=np.arange(0.20,1,0.10)

abg=list(itertools.product(alphas,betas,gammas))



def tes_opmizer(train,abg,step=48):
    best_alpha,best_beta,best_gamma,best_mae=None,None,None,float("inf")
    for comb in abg:
        tes_model=ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit(smoothing_level=comb[0],
                                                                                                 smoothing_slope=comb[1],
                                                                                                 smoothing_seasonal=comb[2])
        y_pred=tes_model.forecast(step)
        mae=mean_absolute_error(test,y_pred)
        if mae < best_mae:
            best_alpha,best_beta,best_gamma,best_mae=comb[0],comb[1],comb[2],mae
        print([round(comb[0],2),round(comb[1],2),round(comb[2],2), round(mae,4)])
    print("best_alpha:",round(best_alpha,2),"best_beta:",round(best_beta,2),"best_gamma:",round(best_gamma,2),"best_mae:",round(best_mae,4))
    return best_alpha,best_beta,best_gamma,best_mae

best_alpha,best_beta,best_gamma,best_mae=tes_opmizer(train,abg,step=48)

final_tes_model=ExponentialSmoothing(train,
                                     trend="add",
                                     seasonal="add",
                                     seasonal_periods=12).fit(smoothing_level=best_alpha,
                                                              smoothing_slope=best_beta,
                                                              smoothing_seasonal=best_gamma)

y_pred=final_tes_model.forecast(48)
plot_co2(train,test,y_pred,"Exponential Smooting for Holt-Winters")



##########################
##ARIMA(p,d,q)
#########################
