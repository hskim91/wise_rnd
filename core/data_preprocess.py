# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:55:27 2020

@author: user
"""

# module import
import sys
import numpy as np
import pandas as pd
from scipy import signal
import scipy
from matplotlib import font_manager, rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns

# PLOT SETTING
sys._enablelegacywindowsfsencoding()
mpl.font_manager._rebuild()
korFontPath = 'C:\\Windows\\Fonts\\malgunsl.ttf'
korFontName = font_manager.FontProperties(fname=korFontPath).get_name()
rc('font', family=korFontName)
mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# FONT SIZE
plt.rc('axes', titlesize=19)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=12)

class c_Preprocessing:
    def __init__(self, dataset, framerate, Ng, Np):
        # -----------------------------------------------------------
        # Signal preprocessing class initialize
        #
        # DB에서 불러온 진동 신호에 대한 클래스
        # 조회기간을 어떻게 처리할것인지? 정해지는 경우 해당 클래스에서 데이터를 직접 끌어오도록 처리
        # dataset : 진동센서 데이터프레임
        # framerate : 센서의 framerate
        # Ng : 기어 이 개수
        # Np : 피니언 이 개수
        # -----------------------------------------------------------
        
        self.dataset = dataset
        self.framerate = framerate
        self.Ng = Ng
        self.Np = Np
        

        
        
    def getDataset(self):
        # -----------------------------------------------------------
        # DB에서 분석데이터 가져오기
        # -----------------------------------------------------------
#         self.Amplitude = dataset.Amplitude
        pass
    
    def Denoising(self, method):
        # -----------------------------------------------------------
        # Autoencoder를 통해 신호 디노이징
        # 데이터 > 디노이징 > 데이터로 저장
        # -----------------------------------------------------------
        pass

        
    def dataSegmentation(self, window_size = 30, sliding_window = False):
        # -----------------------------------------------------------
        # 길이가 긴 데이터에 대한 segmentation
        # window_size : 각 segmentation에 속하는 data point의 개수
        # sliding Window : 슬라이딩 윈도우 기법을 적용하여 데이터를 처리
        # -----------------------------------------------------------
        
        pass    

    
    def TimeDomainFeature_Extract(self, window_size = 20, td = True, fd = False):
        # -----------------------------------------------------------
        # 진동신호에 대해
        ## window_size : 전체 신호를 window로 분할한 뒤 각 윈도우에 대한 특징 추출할 때 윈도우의 크기
        ## td = True 인 경우 시간영역 특징벡터 추출
        ## fd = True 인 경우 주파수영역 특징벡터 추출
        # -----------------------------------------------------------
        n = len(self.dataset) / window_size
    
        if n - round(n) == 0:
            df = pd.DataFrame()
            for i in range(0, len(self.dataset), window_size):
                temp = self.dataset.loc[i : i + window_size - 1] 
                feature_df = pd.DataFrame()
                
                if td:
                    feature_df.loc[i, 'RMS'] = np.sqrt(np.mean(temp ** 2))
                    feature_df['Mean'] = np.mean(temp)
                    feature_df['Var'] = np.var(temp)
                    feature_df['Skewness'] = scipy.stats.skew(temp)
                    feature_df['Kurtosis'] = scipy.stats.kurtosis(temp)
                    feature_df['SF'] = feature_df['RMS'] / np.mean(np.abs(temp))
                    feature_df['CF'] = np.max(np.abs(temp)) / feature_df['RMS']
                    feature_df['IF'] = np.max(np.abs(temp)) / np.mean(np.abs(temp))
                    feature_df['MF'] = np.max(np.abs(temp)) / (np.mean(np.abs(temp))**2)
                    feature_df['PTP'] = temp.max() - temp.min()

                if fd:
                    pass
    
                else:
                    pass
    
                feature_df.columns = ['RMS','Mean','Var','Skewness','Kurtosis','SF','CF','IF','MF','PTP']
                df = pd.concat([df, feature_df])
            
            df.reset_index(inplace=True, drop=True)
            return df
        else:
            print('try different window size')
        
        
    def make_spectrum(self, window_size = 10, full = False):
        # -----------------------------------------------------------
        # FFT를 통해 스펙트럼 계산
        # 주파수영역을 반환
        # window_size : 전체 신호를 window로 분할한 뒤 각 윈도우에 대한 FFT를 수행할 때 윈도우의 크기
        # Full : 허수부의 계산 여부
        # -----------------------------------------------------------
        n = len(self.dataset) / window_size
        if n - round(n) == 0:
            FFTSpectrum = pd.DataFrame()
            for i in range(0, len(self.dataset), window_size):
                temp = self.dataset.loc[i : i + window_size - 1] 
                n = len(temp)
                d = 1/self.framerate
                
                if full:
                    hs = np.fft.fft(temp)
                    fs = np.fft.fftfreq(n, d)
        
                else:
                    hs = np.fft.rfft(temp)
                    fs = np.fft.fftfreq(n, d)
                temp_freq = pd.concat([ pd.DataFrame(fs), pd.DataFrame(np.abs(hs))], axis=1)
                temp_freq.columns = ['fs','hs']
                temp_freq.set_index(fs, inplace=True)
                temp_freq = temp_freq[temp_freq.index > 0]
                FFTSpectrum = pd.concat([FFTSpectrum, pd.DataFrame(temp_freq['hs']).T])
                
            FFTSpectrum.reset_index(inplace = True, drop = True)
            
            return  FFTSpectrum
        else:
            print('try different window size')
    
    
    def make_spectrogram(self, log = False, window_size=20, step_size=10, eps=1e-10):
        # -----------------------------------------------------------
        # STFT를 통해 스펙트로그램 계산
        # 시간-주파수영역인 freq, times, spectrogram array반환
        # window_size : 전체 신호를 window로 분할한 뒤 각 윈도우에 대한 STFT를 수행할 때 윈도우의 크기
        # step_size : overlapping되는 size
        # eps : log spectrogram 계산 시 예외처리를 위한 엡실론값
        # -----------------------------------------------------------
        nperseg = int(round(window_size * self.framerate / 1e3))
        noverlap = int(round(step_size - self.framerate / 1e3))
        freqs, times, spec = signal.spectrogram(self.dataset,
                                                fs = self.framerate,
                                                window = 'hann',
                                                nperseg = nperseg,
                                                noverlap = noverlap,
                                                detrend = False)
        if log == False:
            return freqs, times, spec.T.astype(np.float32)
        
        else:
            return freqs, times, np.log(spec.T.astype(np.float32) + eps)

        
    def plot_spectrogram(self, log = False, window_size = 20, step_size = 10):
        # -----------------------------------------------------------
        # STFT를 통해 스펙트로그램 계산
        # 시간-주파수영역의 2D이미지 반환
        # window_size : 전체 신호를 window로 분할한 뒤 각 윈도우에 대한 STFT를 수행할 때 윈도우의 크기
        # step_size : overlapping되는 size
        # eps : log spectrogram 계산 시 예외처리를 위한 엡실론값
        # -----------------------------------------------------------
        if log:
            freqs, times, spectrogram = self.make_spectrogram(log = True, window_size = 20, step_size = 10)
            
        else:
            freqs, times, spectrogram = self.make_spectrogram(window_size = 20, step_size = 10)
            
        fig = plt.figure(figsize=(40,40))
        ax1 = fig.add_subplot(211)
        ax1.imshow(spectrogram.T, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax1.set_yticks(freqs[::16])
        ax1.set_xticks(times[::16])
        ax1.set_title('Spectrogram')
        ax1.set_ylabel('freqs in Hz')
        ax1.set_xlabel('Seconds')
        plt.show()
        # plt.grid(b=None)
        
    def PrincipalComponentAnalysis(self):
        pca = PCA().fit(self.dataset)
        var = pca.explained_variance_
        cmap = sns.color_palette()
        # plt.subplots(figsize=(30,10))
        plt.bar(np.arange(1, len(var)+1), var/np.sum(var), align='center', color=cmap[0])
        plt.step(np.arange(1,len(var)+1), np.cumsum(var)/np.sum(var), where="mid", color=cmap[1])
        plt.show()
        
        n_component = input('input # of components : ')
        pca = PCA(n_components=n)
        self.dataset = pca.fit_transform(self.datset)
        
        return self.dataset
        
    def AD_ML_split(self, test_size = 0.3):
        # -----------------------------------------------------------
        # 1차년도 이상탐지 분석에 사용한 train_test_split
        # test_size : 전체 데이터세트에서 test세트가 차지하는 비율
        # -----------------------------------------------------------
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                stratify = self.y,
                                                                                random_state=42
                                                                                )
                                                                                
        return self.X_train, self.X_test, self.y_train, self.y_test
        
        
    def FC_ML_split(self, test_size = 0.3):
        # -----------------------------------------------------------
        # 1차년도 결함모드분류 분석에 사용한 train_test_split
        # test_size : 전체 데이터세트에서 test세트가 차지하는 비율
        # -----------------------------------------------------------
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size = test_size, 
                                                                                shuffle = True,
                                                                                stratify = self.y,
                                                                                random_state=42
                                                                                )
                                                                                
        return self.X_train, self.X_test, self.y_train, self.y_test