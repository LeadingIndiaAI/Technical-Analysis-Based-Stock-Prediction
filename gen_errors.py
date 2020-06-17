#Import Libraries
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import math
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error



#The function for finding the directional asymmetry between predicted and actual prices
def directional_asymmetry(y_hat, y_test):
  next_real = pd.Series(np.reshape(y_test, (y_test.shape[0]))).shift(-1)
  next_pred = pd.Series(np.reshape(y_hat, (y_hat.shape[0]))).shift(-1)
  curr_real = pd.Series(np.reshape(y_test, (y_test.shape[0])))[:y_test.shape[0] - 1]
  next_real.dropna(inplace=True)
  next_pred.dropna(inplace=True)
  direction_count = 0
  for i in range(next_real.shape[0]):
    if next_real[i] > curr_real[i] and next_pred[i] > curr_real[i]:
      direction_count += 1
    elif next_real[i] < curr_real[i] and next_pred[i] < curr_real[i]:
      direction_count += 1
    elif next_real[i] == curr_real[i] and next_pred[i] == curr_real[i]:
      direction_count += 1
  return 1 - (direction_count / next_real.shape[0])


# Passing the path to the directory of the dataset
os.chdir('C:/Users/User/Personal/Internship/Bennett University/Project/Dataset/30/CSV Files/')
files = os.listdir()

with open('../Errors.csv', 'w', newline='') as error_file:
    writer = csv.writer(error_file)
    writer.writerow('Stock,LSTM (MAE),SVR_Linear (MAE),SVR_Polynomial (MAE),SVR_Rbf (MAE),Random Forest (MAE),Gradient Boosting (MAE),LSTM (MSE),SVR_Linear (MSE),SVR_Polynomial (MSE),SVR_Rbf (MSE),Random Forest (MSE),Gradient Boosting (MSE),LSTM (Dir. Symm.),SVR_Linear (Dir. Symm.),SVR_Polynomial (Dir. Symm.),SVR_Rbf (Dir. Symm.),Random Forest (Dir. Symm.),Gradient Boosting (Dir. Symm.),Model'.split(','))
# Running the models generated for each stock data
for file in files:

    # Read the csv file
    df = pd.read_csv(file, date_parser=True)
    df = df[['Open', 'High', 'Low', 'Close']]
    df_copy = df.copy()
    stock_open = df['Open']
    stock_high = df['High']
    stock_low = df['Low']
    stock_close = df['Close']


    #Computing Technical Indicator values
    
    #Compute MACD and MACD Histogram
    macd, macdsignal, macdhist = talib.MACD(stock_close, fastperiod=12, slowperiod=26, signalperiod=9)
    dict = {'MACD': macd, 'MSIG': macdsignal}
    macdata = []
    macdata = pd.DataFrame(data=dict)
    macdata.dropna(inplace=True)
    macdata['MACD_Signal1'] = macdata.apply(lambda x : 1 if x['MACD'] > x['MSIG'] else 0, axis = 1)
    n_days = len(macdata['MACD'])
    Signal = np.array(macdata['MACD_Signal1'])
    psy = []
    for d in range(0, n_days):
        if Signal[d] == 1:
            psycology = 1
            psy.append(psycology)
        elif Signal[d] == 0:
            psycology = 0
            psy.append(psycology)
    macdata['MACD_Signal'] = psy
    del macdata['MACD_Signal1']
    dict = {'MHIST': macdhist, 'PrevMHIST': macdhist.shift(1)}
    machdata = []
    machdata = pd.DataFrame(data=dict)
    machdata.dropna(inplace=True)
    machdata['MHIST_Signal1'] = machdata.apply(lambda x : 1 if x['MHIST'] > x['PrevMHIST'] else 0, axis = 1)
    n_days = len(machdata['MHIST'])
    Signal = np.array(machdata['MHIST_Signal1'])
    psy = []
    for d in range(0, n_days):
        if Signal[d] == 1:
            psycology = 1
            psy.append(psycology)
        elif Signal[d] == 0:
            psycology = 0
            psy.append(psycology)
    machdata['MHIST_Signal'] = psy
    del machdata['MHIST_Signal1']


    #Compute Average Directional Index (ADX)
    adx = talib.ADX(stock_high, stock_low, stock_close, timeperiod=14)
    dict = {'Close': stock_close, 'ADX': adx }
    adxdata = []
    adxdata = pd.DataFrame(data=dict)
    adxdata.dropna(inplace=True)
    adxdata['adx1'] = adxdata.apply(lambda x : 1 if x['ADX'] > 25 else 0, axis=1)
    adxdata['adx2'] = adxdata.apply(lambda x : -1 if x['ADX'] < 20 else 0, axis=1)
    adxdata['Signw'] = adxdata.apply(lambda x : x['adx1'] + x['adx2'] , axis=1)


    #Compute Parabolic SAR

    sar = talib.SAR(stock_high, stock_low, acceleration=0.02, maximum=0.2)

    dict = {'Close' : stock_close, 'SAR' : sar}

    sardata = []
    sardata = pd.DataFrame(data=dict)
    sardata.dropna(inplace=True)

    sardata['sar1'] = sardata.apply(lambda x: 1 if x['Close'] > x['SAR'] else 0, axis=1)
    sardata['sar2'] = sardata.apply(lambda x: -1 if x['Close'] < x['SAR'] else 0, axis=1)
    sardata['Sign1'] = sardata.apply(lambda x: x['sar1'] + x['sar2'], axis=1)






    # #Compute Relatie Strength Index (RSI)
    rsi = talib.RSI(stock_close, timeperiod=14)
    dict = {'Close': stock_close, 'RSI': rsi }
    rsidata = []
    rsidata = pd.DataFrame(data=dict)
    rsidata.dropna(inplace=True)
    rsidata['rsi1'] = rsidata.apply(lambda x : 1 if x['RSI'] < 30 else 0, axis=1)
    rsidata['rsi2'] = rsidata.apply(lambda x : -1 if x['RSI'] > 70 else 0, axis=1)
    rsidata['Sign1'] = rsidata.apply(lambda x : x['rsi1'] + x['rsi2'], axis=1)


    # Compute Stochastic 
    slowk, slowd = talib.STOCH(stock_high, stock_low, stock_close, fastk_period=13, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    dict  = {'Close' : stock_close, 'SOK' : slowk, 'SOD' : slowd}

    sodata = []
    sodata = pd.DataFrame(data=dict)
    sodata.dropna(inplace=True)

    sodata['d1'] = sodata.apply(lambda x: 1 if x['SOK'] < 20 and x['SOD'] < 20 else 0, axis=1)
    sodata['d2'] = sodata.apply(lambda x: -1 if x['SOK'] > 80 and x['SOD'] > 80 else 0, axis=1)
    sodata['Sign1'] = sodata.apply(lambda x: x['d1'] + x['d2'], axis=1)



    #Compute Commodity Chanel Index

    real = talib.CCI(stock_high, stock_low, stock_close, timeperiod=13)

    dict = {'Close': stock_close, 'CCI': real }

    ccidata = []
    ccidata = pd.DataFrame(data=dict)
    ccidata.dropna(inplace=True)

    ccidata['cci1'] = ccidata.apply(lambda x : 1 if x['CCI'] < -100 else 0, axis=1)
    ccidata['cci2'] = ccidata.apply(lambda x : -1 if x['CCI'] > 100 else 0, axis=1)
    ccidata['Sign1'] = ccidata.apply(lambda x : x['cci1'] + x['cci2'], axis=1)




    #Compute Bollinger Bands
    upper, middle, lower = talib.BBANDS(stock_close, timeperiod=26)
    dict = {'Close': stock_close, 'Middle': middle, 'Upper': upper, 'Lower': lower }
    bbdata = []
    bbdata = pd.DataFrame(data=dict)
    bbdata.dropna(inplace=True)
    #Generate the Long and Short Signals
    n_days = len(bbdata['Middle'])
    cash = 1
    stock = 0
    position = []
    spread = stock_close
    ma = middle
    upper_band = upper
    lower_band = lower
    for d in range(0, n_days):
        # Long if spread < lower band & if not bought yet
        if spread[d] < lower_band[d] and cash == 1:
            signal = 1
            cash = 0
            stock = 1
            position.append(signal)
        # Take Profit if spread > moving average & if already bought
        elif spread[d] > ma[d] and stock == 1:
            signal = 3
            cash = 1
            stock = 0
            position.append(signal)
        # Short if spread > upper band and no current position
        elif spread[d] > upper_band[d] and cash == 1:
            signal = -1
            cash = 0
            stock = -1
            position.append(signal)
        # Take Profit if spread < moving average & if already short
        elif spread[d] < ma[d] and stock == -1:
            signal = 3
            cash = 1
            stock = 0
            position.append(signal)
        else:
            signal = 0
            position.append(signal)
    bbdata['Position1'] = position
    bbdata['Position1'] = bbdata['Position1'].replace(to_replace=0, method= 'ffill')
    bbdata['Position1'] = bbdata['Position1'].replace(3,0)
    bbdata['Position'] = bbdata['Position1']
    del bbdata['Position1']
    t_days = len(bbdata['Middle'])
    Signal = np.array(bbdata['Position'])
    pos = []
    for d in range(0, t_days):
        if Signal[d] == 0:
            strategy = 0
            pos.append(strategy)
        elif Signal[d] == 1:
            strategy = 1
            pos.append(strategy)
        elif Signal[d] == -1:
            strategy = 0
            pos.append(strategy)
    bbdata['Strategy'] = pos


    # Computing Next DataPoint Move
    stock_move = stock_close.shift(-1)
    dict = {'Close': stock_close, 'Move': stock_move}
    sdmdata = []
    sdmdata = pd.DataFrame(data=dict)
    sdmdata.dropna(inplace=True)



    # Renaming Technical Indicator DataFrames

    Close = pd.DataFrame({'Close': stock_close})
    NM = pd.DataFrame({'NM' : sdmdata['Move']})
    RSI = pd.DataFrame({'RSI': rsidata['Sign1']})
    SO = pd.DataFrame({'SO' : sodata['Sign1']})
    ADX = pd.DataFrame({'ADX': adxdata['Signw']})
    BB = pd.DataFrame({'BB': bbdata['Strategy']})
    MACD = pd.DataFrame({'MACD': macdata['MACD_Signal']})
    MHIST = pd.DataFrame({'MHIST': machdata['MHIST_Signal']})
    CCI = pd.DataFrame({'CCI' : ccidata['Sign1']})
    SAR = pd.DataFrame({'SAR' : sardata['Sign1']})
    # Merging into Single DataFrame

    merge1 = pd.merge(Close, NM, left_index=True, right_index=True, how='outer')
    merge2 = pd.merge(merge1, RSI, left_index=True, right_index=True, how='outer')
    merge3 = pd.merge(merge2, ADX, left_index=True, right_index=True, how='outer')
    merge4 = pd.merge(merge3, MACD, left_index=True, right_index=True, how='outer')
    merge5 = pd.merge(merge4, MHIST, left_index=True, right_index=True, how='outer')
    merge6 = pd.merge(merge5, SO, left_index=True, right_index=True, how='outer')
    merge7 = pd.merge(merge6, CCI, left_index=True, right_index=True, how='outer')
    merge8 = pd.merge(merge7, SAR, left_index=True, right_index=True, how='outer')
    df_final = pd.merge(merge8, BB, left_index=True, right_index=True, how='outer')

    df_final.dropna(inplace=True)



    #Getting Training Data from original data
    training_size = math.ceil(0.55 * df_final.shape[0])
    test_size = df_final.shape[0] - training_size
    train_data = df_final[:training_size]

    # Splitting dependent and independent variables
    X_train = train_data[['Close', 'MACD', 'MHIST', 'ADX', 'SAR', 'RSI', 'SO', 'CCI', 'BB']]
    y_train = train_data[['NM']]

    # Scaling data between 0 to 1
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    X_train = scalerX.fit_transform(X_train)
    y_train = scalerY.fit_transform(y_train)

    # Reshaping data to requirred dimensions
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_trainLSTM, y_trainLSTM = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), np.array(y_train)


    # Creating models
    # LSTM Model
    model = Sequential()
    model.add(LSTM(units=40, activation='relu', return_sequences=True, input_shape=(X_trainLSTM.shape[1], 1)))
    model.add(Dropout(rate=0.1))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(rate=0.1))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # SVR Model with 'rbf' kernel
    regr_rbf = SVR(kernel='rbf', gamma=0.1)
    
    # SVR Model with 'poly' kernel
    regr_poly = SVR(kernel='poly', degree=2)

    # SVR Model with 'linear' kernel
    regr_lin = SVR(kernel='linear')

    # Random Forest Regressor Model
    regr_rfr = RandomForestRegressor(n_estimators=150, criterion='mse', oob_score=True)

    # Gradient Boosting Regressor Model
    regr_gbr = GradientBoostingRegressor(n_estimators=150, criterion='friedman_mse', loss='ls')

    # Training Models
    model.fit(X_trainLSTM, y_trainLSTM, batch_size=128, epochs=30)
    regr_rbf.fit(X_train, y_train)
    regr_poly.fit(X_train, y_train)
    regr_lin.fit(X_train, y_train)
    regr_rfr.fit(X_train, y_train)
    regr_gbr.fit(X_train, y_train)


    # Getting test data from original data
    test_data = df_final[training_size - 1:]

    # Splitting dependent and independent variables
    X_test = test_data[['Close', 'MACD', 'MHIST', 'ADX', 'SAR', 'RSI', 'SO', 'CCI', 'BB']]
    y_test = test_data[['NM']]

    # Scaling data between 0 to 1
    X_test = scalerX.fit_transform(X_test)
    y_test = scalerY.fit_transform(y_test)

    # Reshaping data to match sizes required by models
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_testLSTM, y_testLSTM = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)), np.array(y_test)

    # Predicting prices using the models created
    y_hatLSTM = model.predict(X_testLSTM)
    y_hat_rbf = regr_rbf.predict(X_test)
    y_hat_poly = regr_poly.predict(X_test)
    y_hat_lin = regr_lin.predict(X_test)
    y_hat_rfr = regr_rfr.predict(X_test)
    y_hat_gbr = regr_gbr.predict(X_test)

    # Inverse scaling to get real time values back
    y_hatLSTM = scalerY.inverse_transform(y_hatLSTM)
    y_testLSTM = scalerY.inverse_transform(y_testLSTM)

    # Reshaping data again to match the dimensions of the scaler
    y_hat_poly = np.reshape(y_hat_poly, (y_hat_poly.shape[0], 1))
    y_hat_rbf = np.reshape(y_hat_rbf, (y_hat_rbf.shape[0], 1))
    y_hat_lin = np.reshape(y_hat_lin, (y_hat_lin.shape[0], 1))
    y_hat_rfr = np.reshape(y_hat_rfr, (y_hat_rfr.shape[0], 1))
    y_hat_gbr = np.reshape(y_hat_gbr, (y_hat_gbr.shape[0], 1))

    # Inverse scaling
    y_hat_poly = scalerY.inverse_transform(y_hat_poly)
    y_hat_rbf = scalerY.inverse_transform(y_hat_rbf)
    y_hat_lin = scalerY.inverse_transform(y_hat_lin)
    y_hat_rfr = scalerY.inverse_transform(y_hat_rfr)
    y_hat_gbr = scalerY.inverse_transform(y_hat_gbr)
    y_test = scalerY.inverse_transform(y_test)

    # ERRORS

    # Mean squared error of each model
    MSE_rbf = mean_squared_error(y_hat_rbf, y_test)
    MSE_poly = mean_squared_error(y_hat_poly, y_test)
    MSE_lin = mean_squared_error(y_hat_lin, y_test)
    MSE_rfr = mean_squared_error(y_hat_rfr, y_test)
    MSE_gbr = mean_squared_error(y_hat_gbr, y_test)
    MSE_LSTM = mean_squared_error(y_hatLSTM, y_testLSTM)

    # Mean absolute error of each model
    MAE_rbf = mean_absolute_error(y_hat_rbf, y_test)
    MAE_poly = mean_absolute_error(y_hat_poly, y_test)
    MAE_lin = mean_absolute_error(y_hat_lin, y_test)
    MAE_rfr = mean_absolute_error(y_hat_rfr, y_test)
    MAE_gbr = mean_absolute_error(y_hat_gbr, y_test)
    MAE_LSTM = mean_absolute_error(y_hatLSTM, y_testLSTM)

    # Directional Asymmetry for each model
    DA_LSTM = directional_asymmetry(y_hatLSTM, y_testLSTM)
    DA_lin = directional_asymmetry(y_hat_lin, y_test)
    DA_poly = directional_asymmetry(y_hat_poly, y_test)
    DA_rbf = directional_asymmetry(y_hat_rbf, y_test)
    DA_rfr = directional_asymmetry(y_hat_rfr, y_test)
    DA_gbr = directional_asymmetry(y_hat_gbr, y_test)

    # Scaling error values between 0 to 1
    mse = [MSE_LSTM, MSE_lin, MSE_poly, MSE_rbf, MSE_rfr, MSE_gbr]
    mae = [MAE_LSTM, MAE_lin, MAE_poly, MAE_rbf, MAE_rfr, MAE_gbr]
    da = [DA_LSTM, DA_lin, DA_poly, DA_rbf, DA_rfr, DA_gbr]
    ds = [100 - (float(x) * 100) for x in da]
    scaled_mse = minmax_scale(np.transpose(mse))
    scaled_mae = minmax_scale(np.transpose(mae))
    scaled_da = minmax_scale(np.transpose(da))
    mean_metric = []
    models = ['LSTM', 'SVR-Linear', 'SVR-Polynomial', 'SVR-Rbf', 'Random Forest', 'Gradient Boosting']
    for i in range(len(da)):
        mean_metric.append((scaled_mae[i] + scaled_mse[i] + scaled_da[i]) / 3)

    # Adding the error values to a CSV File
    with open('../Errors.csv', 'a+', newline='') as error_file:
        writer = csv.writer(error_file)
        writer.writerow([file.split('_')[0]] + scaled_mae.tolist() + scaled_mse.tolist() + ds +[models[mean_metric.index(min(mean_metric))]])
    print(file + " predicted successfully!")

print("\n\nERROR FILE GENERATED!!")