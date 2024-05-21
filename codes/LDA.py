# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

class LDA():
    #-------------------
    # 1. 学習データの設定と、全体および各カテゴリの平均の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def __init__(self,X,Y):
        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]  # 学習データ数
        self.xDim = X.shape[1]  # 入力の次元数
        
        # 各カテゴリに属す入力データ
        self.Xneg = X[Y[:,0]==-1]
        self.Xpos = X[Y[:,0]==1]

        # 全体および各カテゴリに属すデータの平均
        self.m = np.mean(self.X,axis=0,keepdims=True)
        self.mNeg = np.mean(self.Xneg,axis=0,keepdims=True)
        self.mPos = np.mean(self.Xpos,axis=0,keepdims=True)
    #-------------------

    #-------------------
    # 2. 固有値問題によるモデルパラメータの最適化
    def train(self):
        # カテゴリ間分散共分散行列Sinterの計算
        Sinter = np.matmul((self.mNeg-self.mPos).T,self.mNeg-self.mPos)
        
        # カテゴリ内分散共分散行列和Sintraの計算
        Xneg = self.Xneg - self.mNeg
        Xpos = self.Xpos - self.mPos
        Sintra = np.matmul(Xneg.T,Xneg) + np.matmul(Xpos.T,Xpos)
        
        # 固有値問題を解き、最大固有値の固有ベクトルを獲得
        [L,V] = np.linalg.eig(np.matmul(np.linalg.inv(Sintra),Sinter))
        self.w = V[:,[np.argmax(L)]]
    #-------------------

    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # method: 分類境界の計算方法
    #     "mean": 全体平均
    #     "midpoint": 各クラス平均の中点
    #     "int_div_var": 各クラス平均間を分散で内分
    #     "int_div_std": 各クラス平均間を標準偏差で内分
    def predict(self,x,method='mean'):
        self.b = 0

        if method == 'midpoint':
            self.m = (self.mNeg + self.mPos) / 2 # 各クラス平均ベクトルの中点
            self.b = - np.matmul(self.m,self.w) / 2 # 
        elif method == 'int_div_var' or method == 'int_div_std':
            Xneg = self.Xneg - self.mNeg
            Xpos = self.Xpos - self.mPos
            Sneg = np.matmul(Xneg.T,Xneg)
            Spos = np.matmul(Xpos.T,Xpos)
            sigma_neg = float(self.w.T @ Sneg @ self.w) / Xneg.shape[0]
            sigma_pos = float(self.w.T @ Spos @ self.w) / Xpos.shape[0]
            if method == 'int_div_std':
                sigma_neg = np.sqrt(sigma_neg)
                sigma_pos = np.sqrt(sigma_pos)
            
            self.m = ((sigma_pos * self.mNeg) + (sigma_neg * self.mPos)) / (sigma_neg + sigma_pos)
            
        self.b = -np.matmul(self.m, self.w)

        return np.sign(np.matmul(x,self.w)+self.b)
    #-------------------
    
    #-------------------
    # 4. 正解率の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    def accuracy(self,x,y,method = 'mean'):
        return np.sum(self.predict(x,method=method)==y)/len(x)
    #-------------------
    
    #-------------------
    # 5. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # title: タイトル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        if X.shape[1] != 2: return
    
        #fig = plt.figure(figsize=(6,4),dpi=100)
        
        # 最小と最大の点
        X1 = np.arange(np.min(X[:,0]),np.max(X[:,0]),(np.max(X[:,0]) - np.min(X[:,0]))/100)
        X2 = (np.matmul(self.m,self.w)[0] - X1*self.w[0])/self.w[1]

        # データと線形モデルのプロット
        plt.plot(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],'cx',markersize=14,label="カテゴリ-1")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markersize=14,label="カテゴリ+1")
        plt.plot(self.m[0,0],self.m[0,1],'ko',markersize=12,label="全体の平均")
        plt.plot(X1,X2,'r-',label="f(x)")
        
        # 各軸の範囲とラベルの設定
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.xlabel(xLabel,fontsize=14)
        plt.ylabel(yLabel,fontsize=14)
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------