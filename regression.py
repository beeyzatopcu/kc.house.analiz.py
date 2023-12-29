# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:46:16 2023

@author: beyza
"""


import pandas as pd
import numpy as np


import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as mp
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("kc_house_data.csv")

# Verimize geniş bir sekilde bakıyoruz
data.head()

# Tanımlayıcı istatstiklere bakıyoruz
data.describe().T

# veride hangi kolonlar var gorelim
data.columns 

# Verimizde eksik gözlem bulunuyor mu diye bakmalıyız.
data.isnull().values.any() 
#eğer eksik veri olsaydı eksik verileri giderme yontemi kullanılırdı fakat False
#yazıyor eksik verimiz yok.

# veri setindeki degiskenlerin aralarındaki korelasyonlara bakıyoruz
#İki değişken arasındaki korelasyon -1 ve 1 arasındadır.
#Korelasyon katsayısı 1'e yaklaştıkça pozitif yönde doğrusal ilişki artar
#-1'e yaklaştıkça negatif yönde doğrusallık artar.
data.corr()
#bağımlı değiskendeki değisim için en yüksek değer 0.702035 o iki bağımsız değişken 
#arasında ilerleyeceğim

# Regresyon icin bagımsız degisken sqft_living  ve bagımlı
# degisken icin ise price degiskeni.

#verim biraz büyük olduğu için plotlar çizilmesi uzun sürüyor.
import seaborn as sbn
sbn.pairplot(data, kind="reg") 


sbn.jointplot(x = "sqft_living", y = "price",
             data = data, kind = "reg")
# sacılım grafigi ile beraber degiskenlerin dagılımını gosterir.

# Regresyon modeli kurmak statsmodels sklear kullanıyoruz ilk 
#statsmodelsi kullanalım.

y = data[["price"]] #bagımlı
y.describe() # sectigimizi veryi gorelim
x = data[["sqft_living"]] #bagımsız

import statsmodels.api as sm

# sabit parametre için 1'lerden olusan matris kolonu ekledim.
x = sm.add_constant(x)
x.head()

#ekk tahmini olusturulur
model = sm.OLS(y, x).fit()

# Cıktıları elde etmek icin
model.summary()
#modelimizin R-squared: 0.493, Adj. R-squared:0.493
#AIC:6.005e+05, BIC:6.006e+05

import statsmodels.formula.api as smf
model = smf.ols("price~sqft_living", data=data).fit()

# summaryde gördüğümüz değerleri extra görmek isersek;
model.params
model.conf_int()
model.f_pvalue
model.fvalue
model.tvalues
model.mse_model
model.rsquared
model.rsquared_adj

#fit edilen y degerlerine baktık
model.fittedvalues 

# Model ne kadar iyi fit edildiğine bakalım
import seaborn as sbn
sbn.regplot(y, model.fittedvalues)
#y değerlerine bakıyoruz
#grafiktede görüldüğü üzere alt ve sol kısımda yoğunluk var
#linenın devamında yoğunluk yok

mean_squared_error(y,model.fittedvalues) #68351286833.03982
r2_score(y, model.fittedvalues ) #0.4928532179037932

# grafik olusturursak
sbn.regplot(data["sqft_living"], data["price"], ci=None, scatter_kws={"color":"r",
                                        "s":13}).set_title("Regresyon Modeli")

#ilk olarak sklearn kutuphanesi ile yapalım.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
mod = model.fit(x,y)

# cıktıları elde etmek icin
mod.intercept_
mod.coef_

# R^2 
mod.score(x, y)
#r2 değeri 1e yaklasatıkça modelin gücü artar

# Fit degerlerini 
mod.predict(x)


from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as mp

# mse degerlerini hesaplamak icin
mse = mean_squared_error(y, mod.predict(x)) #68351286833.039825
#r^2 degerlerini hesaplamak icin
r2_score(y, mod.predict(x)) #0.4928532179037931

mp.scatter(data["price"], mod.predict(x), color = "blue")

#çoklu regresyon

y = data["price"] #bagımlı 
b = data.drop("id",axis=1)
c = b.drop("date",axis=1)
X = c.drop("price",axis=1) #bagımsız

# test_size 0.3 --> % 30 ise kalan % 70 train olur
# random state = 15, random olmasin secimi yapiyoruz
# icindeki ornekler ayni olarak devam ediyor
xegitim, xtest, yegitim, ytest = train_test_split(X, y, test_size=0.30, random_state=15)

xegitim.shape
xtest.shape

# stats models kutuphanesi
model = sm.OLS(yegitim,xegitim).fit();model.summary()

# model Performansı
tahmin = model.predict(xegitim)
model_mse_egitim = mean_squared_error(yegitim, tahmin) 
#40960456205.24542
model_R2_egitim = r2_score(yegitim, tahmin) 
#0.6952763692723454

# tahmin Performansı
tahmin = model.predict(xtest)
model_mse_test = mean_squared_error(ytest, tahmin) 
#39564195542.22568
model_R2_test = r2_score(ytest, tahmin)
#0.7080645823680265

# sklearn kutuphanesi
model = LinearRegression()
mod = model.fit(xegitim, yegitim)
mod.intercept_
mod.coef_
fit = mod.predict(xegitim)
mp.scatter(yegitim, fit, color = 'red')

# Test kumesindeki bagimli degisken degerlerini tahmin edebilmek icin test 
#kumesindeki bagimsiz degiskenleri kullanarak isleme aldık
tahminler = mod.predict(xtest)
mp.scatter(ytest, tahminler, color = "blue")

mean_squared_error(ytest, tahminler) #39553042446.83695

# mape degeri
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(ytest, tahminler) #0.25804661724263633

#PCA

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import model_selection

from sklearn.model_selection import KFold

# K-fold (K=5)
cv = KFold(n_splits=10,shuffle=True,random_state=42)

pca = PCA()
xegitim_scaled = pca.fit_transform(scale(xegitim))

regr = LinearRegression()
mse = []

# MSE değerini hesaplamarken sadece sabit parametrelerin oldugu modelden yararlanalım

for i in np.arange(1,10):
    score = -1*model_selection.cross_val_score(regr, xegitim_scaled[:,:i],
                                           yegitim, cv = cv, scoring="neg_mean_squared_error").mean()
    mse.append(score)

print(mse)

import matplotlib.pyplot as mp
#cross-validation sonuclarına gore bir plot olusturmak istersek
mp.plot(mse)
mp.xlabel("Temel Olan Bilesenlerin Sayısı")    
mp.ylabel("MSE değerleri")


model = LinearRegression()
pcreg = model.fit(xegitim_scaled, yegitim)

# Model parametre tahminleri
pcreg.intercept_ # array([543433.59510012])
pcreg.coef_

#performansı
yhat = pcreg.predict(xegitim_scaled)

pcreg_mse_egitim = mean_squared_error(yegitim, yhat)
 #40950975316.89153
pcreg_R2_egitim = r2_score(yegitim, yhat)
 #0.6953469019516507

import seaborn as sbn
sbn.regplot(yegitim, yhat)


# Tahminleme Asaması (test kümesi için)
pca = PCA()
xtest_scaled = pca.fit_transform(scale(xtest))

predicted_y = pcreg.predict(xtest_scaled)#tahmin

sbn.regplot(ytest, predicted_y)

r2_score(ytest, predicted_y)
#R2 değeri 0.6110412558499053 çıktı kötü bir değer değil

#PLS

from sklearn.cross_decomposition import PLSRegression

pls_model = PLSRegression().fit(xegitim, yegitim)
pls_model.coef_

y_hat_pls0= pls_model.predict(xegitim)

#pls bilesenini belirtterek model alma
pls_model1 = PLSRegression(n_components = 4).fit(xegitim,yegitim)
pls_model1.coef_

y_hat_pls1= pls_model1.predict(xegitim)

# n_components = 4 ile kurulan modelin tahmin degerleri
# ile pls0ı karsılastırdım
sbn.regplot(yegitim,y_hat_pls0)
sbn.regplot(yegitim,y_hat_pls1)

mean_squared_error(yegitim, y_hat_pls0)# 45621877794.88187
mean_squared_error(yegitim, y_hat_pls1)# 41200807076.134636

# optizasyon yapılması (PLS için) 

from sklearn.model_selection import cross_val_predict

def optimum_pls(X, y, ncomp):
    model = PLSRegression(n_components=ncomp)
    cv = cross_val_predict(model, x, y, cv=5)
    rsq = r2_score(y, cv)
    mese = mean_squared_error(y, cv)
    return (cv, rsq, mese)

def plot_metrics(değer, ylabel, fonksiyon):
    with mp.style.context('ggplot'):
        mp.plot(np.arange(1, 14),np.array(değer), '-v', color='blue', mfc='blue')
        if fonksiyon=='min':
            idx = np.argmin(değer)
        else:
            idx = np.argmax(değer)
        mp.plot(np.arange(1, 14),np.array(değer), 'P', ms=10, mfc='red')

        mp.xlabel('PLS sayısı')
        mp.xticks = xticks
        mp.ylabel(ylabel)
        mp.title('PLS')
        mp.show()
    
plot_metrics(mese, 'MSE', 'min')
plot_metrics(rsq, 'R2', 'max')   

# tahmin ve modelin performans karsılastırması:
tahmin1 = pls_model1.predict(xegitim)
tahmin1 = pls_model1.predict(xtest)

pls_mse_egitim = mean_squared_error(yegitim,tahmin1) 
pls_mse_test = mean_squared_error(ytest, tahmin1)

pls_R2_egitim = r2_score(yegitim,tahmin1) 
pls_r2_test = r2_score(ytest, tahmin) 



# Ridge Regresyon
from sklearn.linear_model import Ridge
ridge_mod = Ridge(alpha = 0.2).fit(xegitim, yegitim)
ridge_mod.coef_

ridge_mod2 = Ridge().fit(xegitim, yegitim)

yhat1 = ridge_mod.predict(xegitim)

yhat2 = ridge_mod2.predict(xegitim)

    
ridge_model3 = Ridge(alpha=0.01).fit(xegitim, yegitim)

y_hat_alpha3 = ridge_model3.predict(xegitim)

# Alpha optimizasyonu

r2 = []
mse_sonuc = []
alpha_cand = np.array([0,0.04,0.2,0.1,0.4,1,3,9])
for i in alpha_cand:
    model_i = Ridge(alpha=i).fit(xegitim, yegitim)
    yhat = model_i.predict(xegitim)
    mse_sonuc.append(mean_absolute_percentage_error(yegitim, yhat))
    r2.append(r2_score(yegitim, yhat))
    
print(mse_sonuc)
#alpha arttıkca mse de artmıs
print(np.sort(mse_sonuc))

ridge_model4 = Ridge(alpha=0.2).fit(xegitim, yegitim)

# Model Performansı ile tahmin performansı karsılastırması :
tahmin2 = ridge_model4.predict(xegitim)
tahmin3 = ridge_model4.predict(xtest)

mse_4_egitim = mean_squared_error(yegitim, tahmin2) #40951059966.86896
mse_4_test = mean_squared_error(ytest, tahmin3)  #39556045539.59643 

r2_4_egitim =  r2_score(yegitim, tahmin2)#0.6953462722016222
r2_4_test = r2_score(ytest, tahmin3) #0.7081247194285344

sbn.regplot(yegitim, y_hat_alpha3)  


# Lasso Regresyon
from sklearn.linear_model import Lasso
lasso_m = Lasso(alpha = 0.2).fit(xegitim, yegitim)
lasso_m.coef_

fitt = lasso_m.predict(xegitim)
np(mean_squared_error(yegitim,fitt))
r2_score(yegitim, fitt)


# Alpha (Lambda) Optimizasyonu 
from sklearn.linear_model import LassoCV
model = LassoCV(cv=5, random_state=42, max_iter=20000)
model.fit(xegitim, yegitim)
LassoCV(cv=5, max_iter=10000, random_state=42)
model.alpha_ # 1372215.453414303
opt_alpha = model.alpha_
# optim_alpha yı alpha parametresinde yerine 

mod2 = Lasso(alpha=model.alpha_)
mod2.fit(xegitim, yegitim)
fit = mod2.predict(xegitim)
r2_score(yegitim, fit)#0.5211691099951534
np.sqrt(mean_squared_error(yegitim,fit))
 
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as mp
from sklearn.neighbors import KNeighborsRegressor


#KN
KNN_model = KNeighborsRegressor().fit(xegitim, yegitim)

# K degerine bakıyoruz
KNN_model.n_neighbors
#k değerimiz 5miş

y_hat_knn5 = KNN_model.predict(xegitim) 
#ysapka, yani y tahminleri

mean_squared_error(yegitim, y_hat_knn5) #43299467452.74638
r2_score(yegitim, y_hat_knn5) #0.6778753912148785

#en iyi k değerini bulmak istiyoruz
mse = []
r2 = []
for k in range(13):
    k = k+1
    KNN_model = KNeighborsRegressor(n_neighbors=k).fit(xegitim, yegitim)
    y_hat_k = KNN_model.predict(xtest)
    mse.append(mean_squared_error(ytest, y_hat_k))
    r2.append(r2_score(ytest, y_hat_k))
    

mse_df = pd.DataFrame(mse)
mse_df.plot()

# K değerine Grid Search ile  bakalım 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV

KNN_arg = KNeighborsRegressor()
k_params = {"n_neighbors": np.arange(1,40)}
KNN_model = GridSearchCV(KNN_arg, k_params, cv = 15)
KNN_model.fit(xegitim, yegitim)

KNN_model.best_params_ #{'n_neighbors': 16}


KNN_model.best = KNeighborsRegressor(n_neighbors=2).fit(xegitim, yegitim)

# Model Performansı ile Tahmin Performansı :
tahmin_knn_best = KNN_model.best.predict(xegitim)
predict_knn_best = KNN_model.best.predict(xtest)

#mean_squared
knn_mse2_egitim = mean_squared_error(yegitim, tahmin_knn_best) #24477131564.827618
knn_mse2_test = mean_squared_error(ytest, predict_knn_best) #78323622064.82916

#r2 karsılastırması
knn_r2_2_egitim = r2_score(yegitim, tahmin_knn_best) # 0.8179033855761212
knn_r2_2_test = r2_score(ytest, predict_knn_best) #0.4220673767133427

knn_r2_2_egitim > knn_r2_2_test #True
# Anlasılacagı gibi eğitim modeli daha iyidir.

# Ancak burada goruldugu uzere KNN model optimizasyonu overfitting yapmıstır.


#######SVR çok zor calısıyor sebebini çözemedim genelde çalışmadı bile
# Dogrusal Olan SVR
from sklearn.svm import SVR

SVR_model = SVR(kernel="linear").fit(xegitim, yegitim)
y_hat_svr_l = SVR_model.predict(xegitim)

mean_squared_error(yegitim, y_hat_svr_l)
r2_score(yegitim, y_hat_svr_l)

# KNN (n=5) ile karşılaştırmamız gerek 

predicted_y_svr_l = SVR_model.predict(xtest)
mean_squared_error(ytest, predicted_y_svr_l)
r2_score(ytest, predicted_y_svr_l)


# Dogrusal Olmayan SVR

SVR_model_nl = SVR(kernel="rbf").fit(xegitim, yegitim)
y_hat_svr_nl = SVR_model.predict(xegitim)    

mean_squared_error(yegitim, y_hat_svr_nl)    
r2_score(yegitim, y_hat_svr_nl)    

predicted_y_svr_nl = SVR_model.predict(xtest)    
mean_squared_error(ytest, predicted_y_svr_nl)    
r2_score(ytest, predicted_y_svr_nl)

# En iyi modeli belirlemek icin Grid Search uygularsak ;
params_svr = {"C":np.arange(0.1,0.5, 3, 0.4)}
gs_SVR_model_l = GridSearchCV(SVR_model, params_svr, cv = 10).fit(xegitim, yegitim)
gs_SVR_model_nl = GridSearchCV(SVR_model_nl, params_svr, cv = 10).fit(xegitim, yegitim)

gs_SVR_model_l.best_params_
gs_SVR_model_nl.best_params_

bp = pd.Series(gs_SVR_model_l.best_params_)[0]

best_l_svr_model = SVR(kernel="linear", C = bp).fit(xegitim, yegitim)
best_nl_svr_model = SVR(kernel="rbf", C = bp).fit(xegitim, yegitim)

y_hat_l_best = best_l_svr_model.predict(xtest)
y_hat_nl_best = best_nl_svr_model.predict(xtest)

mean_squared_error(ytest, y_hat_l_best)
r2_score(ytest, y_hat_l_best)

mean_squared_error(ytest, y_hat_nl_best)
r2_score(ytest, y_hat_nl_best)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

scl = StandardScaler()

#standartlastırma yapalım
scl.fit(xegitim)
xegitim_scl = scl.transform(xegitim)
xtest_scl = scl.transform(xtest)

ann_model = MLPRegressor().fit(xegitim_scl, yegitim)
ann_model.n_layers_
#kaç gizli tabakam var bakıyorum. 3taneymis
ann_model.hidden_layer_sizes  #(100,)

y_hat_ann0 = ann_model.predict(xegitim_scl)
mean_squared_error(yegitim, y_hat_ann0) #301968122113.9755

y_predict_ann0 = ann_model.predict(xtest_scl)
mean_squared_error(ytest, y_predict_ann0) #293886138506.55145

r2_score(yegitim, y_hat_ann0)
r2_score(ytest, y_predict_ann0)


# Grid Search ile Optimizasyon ;
from sklearn.model_selection import GridSearchCV

params_ann = {"alpha" : [0.2,0.01,0.03,0.004],
              "hidden_layer_sizes" : [(10,10),(100,125,150),(300,200,100)],
              "activation" : ["relu", "logistic"]}

gs_ann_model = GridSearchCV(ann_model, params_ann, cv = 5)
gs_ann_model.fit(xegitim, yegitim)

gs_ann_model.best_params_

best_ann = MLPRegressor(alpha = 0.2, hidden_layer_sizes = (300,200,100),
                        activation = "relu")

model_best = best_ann.fit(xegitim_scl, yegitim)

best_yhat = model_best.predict(xegitim_scl)
mean_squared_error(yegitim, best_yhat)
r2_score(yegitim, best_yhat) 
#cok cok cok iyi

best_predicted = model_best.predict(xtest_scl)
mean_squared_error(ytest, best_predicted)
r2_score(ytest, best_predicted)

#

from sklearn.tree import DecisionTreeRegressor

cart_model = DecisionTreeRegressor().fit(xegitim, yegitim)
fitted_cart = cart_model.predict(xegitim)

mean_squared_error(yegitim, fitted_cart)
r2_score(yegitim, fitted_cart)
#0.9994762518544558

preds_cart = cart_model.predict(xtest)
mean_squared_error(ytest, preds_cart)
r2_score(ytest, preds_cart)
#0.7484371355230902

# yeğittim 0.9994762518544558 ve ytest 0.7448508344292011
# birebir tahmin etmiyor, birebir tahmin etmeside iyi değil zaten.

# Model Optimizasyonu ile test setine optimizasyon yapıyorum
cart_pars = {"min_samples_split": range(2,200),
             "max_leaf_nodes": range(2,20)}
from sklearn.model_selection import GridSearchCV
grid_cart_model = GridSearchCV(cart_model, cart_pars, cv=10)
grid_cart_model.fit(xegitim, yegitim)
grid_cart_model.best_params_

best_cart_model = DecisionTreeRegressor(max_leaf_nodes=9,
                                        min_samples_split=32).fit(xegitim, yegitim)
preds_best_cart = best_cart_model.predict(xtest)
mean_squared_error(ytest, preds_best_cart )
r2_score(ytest, preds_best_cart )
#r2  0.6113526904486507 ve optime ettik

# Bagging (Booststrap Aggregation)
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(bootstrap_features=(True)).fit(xegitim, yegitim)
fits_bag = bagging_model.predict(xegitim)
mean_squared_error(yegitim, fits_bag) #3976207143.4987164
r2_score(yegitim, fits_bag) #0.9704191703524776 baya iyi bir değer

preds_bag = bagging_model.predict(xtest)
mean_squared_error(ytest, preds_bag) #22436552747.02311
r2_score(ytest, preds_bag)#0.8344456570731142 test için yine iyi

# Cart ile karsılastırdıgımızda daha iyi diyebiliriz ama eğitim için test burda daha iyi.

# Model Optimizasyonu
bag_pars = {"n_estimators": range(2,40)}
grid_bag_model = GridSearchCV(bagging_model, bag_pars, cv=10)
grid_bag_model.fit(xegitim, yegitim)

grid_bag_model.best_params_

best_bag_model = BaggingRegressor(n_estimators=37).fit(xegitim, yegitim)
best_preds = best_bag_model.predict(xtest)

mean_squared_error(ytest, best_preds)#17984058086.554413
r2_score(ytest, best_preds)#0.8672996269414206


# Random Forest (RF)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor().fit(xegitim, yegitim)
fits_rf = rf_model.predict(xegitim)

mean_squared_error(yegitim, fits_rf) #2381612811.3543644
r2_score(yegitim, fits_rf)#0.9822820893588958

preds_rf = rf_model.predict(xtest)
mean_squared_error(ytest, preds_rf)#17630483835.571945
r2_score(ytest, preds_rf)#0.8699085728636071
# Random Forrest Bagging'e gore daha iyi sonuc verdi, çok az bir fark ama sonuca bakarsak daha iyi



# Model Optimizasyonu
rf_pars = {"max_depth": range(1,6),
           "max_features": [2,4],
           "n_estimators": [100,350]}
grid_rf_model = GridSearchCV(rf_model, rf_pars, cv=15, n_jobs=-1)
grid_rf_model.fit(xegitim, yegitim)

grid_rf_model.best_params_

best_rf_model = RandomForestRegressor(max_depth=4,
                                      max_features=3,
                                      n_estimators=350).fit(xegitim, yegitim)
best_preds_rf = best_rf_model.predict(xtest)
mean_squared_error(ytest, best_preds_rf)
r2_score(ytest, best_preds_rf)
# Optimize ettik ancak hızlı cozum vermesi icin 
# parametre uzayını kısıtlı tuttuk ve bu nedenle
# optimize ettigimiz model normal rf modelimizden
# daha kotu sonuc verdi




# Gradient Boosting Machines
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor().fit(xegitim, yegitim)
gb_fits = gb_model.predict(xegitim)
mean_squared_error(yegitim, gb_fits) #13668482492.771591
r2_score(yegitim, gb_fits) #0.8983138861817326

gb_preds = gb_model.predict(xtest)
mean_squared_error(ytest, gb_preds)#18972730081.704205
r2_score(ytest, gb_preds)#0.8600044357249823


# Model Optimizasyonu
gb_pars = {"learning_rate": [0.001,0.01,0.1,0.2],
           "max_depth": [3,5,10,50,100],
           "n_estimators": [100,200,500,1000,2000],
           "subsample": [1,0.5,0.75]}

grid_gb_model = GridSearchCV(gb_model, gb_pars, cv=10, n_jobs=-1,verbose=2) 
grid_gb_model.fit(xegitim, yegitim)

grid_gb_model.best_params_

best_gb_model = GradientBoostingRegressor(learning_rate=0.01,
                                          max_depth=5,
                                          n_estimators=500,
                                          subsample=0.5).fit(xegitim, yegitim)
best_gb_preds = best_gb_model.predict(xtest)
mean_squared_error(ytest, best_gb_preds)
r2_score(ytest, best_gb_preds)


# gb_pars'daki parametreleri degistirdik ve dolayısıyla best parametrelerde degisiklik olacaktır.
# yeniden calıstırırsak gorecegiz.



# XGboost 
!pip install xgboost
conda install xgboost
import xgboost as xgb
from xgboost import XGBRFRegressor

xgb_model = XGBRFRegressor().fit(xegitim, yegitim)
xgb_fits = xgb_model.predict(xegitim)
mean_squared_error(yegitim, xgb_fits)
r2_score(yegitim, xgb_fits)

xgb_preds = xgb_model.predict(xtest)
mean_squared_error(ytest, xgb_preds)
r2_score(ytest, xgb_preds)

# Model Optimizasyonu
xgb_params = {"colsample_bytree":[0.4,0.5,0.6,0.7,0.9,1],
              "n_estimators":[100,250,500,1000,2000],
              "max_depth":[2,3,4,5,6],
              "learning_rate":[0.1,0.01,0.5]}

grid_xgb_model = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2)
grid_xgb_model.fit(xegitim, yegitim)

grid_xgb_model.best_params_

best_model = XGBRFRegressor(colsample_bytree=0.9,
                            learning_rate=0.1,
                            max_depth=6,
                            n_estimators=100).fit(xegitim, yegitim)

best_xgb_preds = best_model.predict(xtest)
mean_squared_error(ytest, best_xgb_preds)
r2_score(ytest, best_xgb_preds)



