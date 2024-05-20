# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import japanize_matplotlib

class perceptron():
    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×1のnumpy.ndarray）
    def __init__(self,X,Y, seed=0):
        # 学習データの設定
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]  # 学習データ数
        self.xDim = X.shape[1]  # 入力の次元数

        # 行列Xに「1」の要素を追加
        self.Z = np.concatenate([self.X,np.ones([self.dNum,1])],axis=1)

        np.random.seed(seed=seed)
        # パラメータの拡張表現
        self.V = np.random.normal(size=[self.xDim+1,1])        
        self.w = self.V[:-1]
        self.b = self.V[-1]

        self.lr = 10e-3
    #-------------------

    #-------------------
    # 2. パーセプトロンの学習規則を用いてパラメータを更新
    def update(self, alpha = 1e-3, mergin=0):

        P, fx = self.predict(self.X)

        for i in np.where(self.Y*fx <= mergin)[0]:
            self.V[:,0] += self.lr * self.Y[i] * self.Z[i,:]

        # パラメータw,bの決定
        self.w = self.V[:-1]
        self.b = self.V[-1]
    #-------------------
    
    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        f_x = (np.matmul(x,self.w) + self.b) / np.linalg.norm(self.w)
        return np.sign(f_x),f_x
    #-------------------

    #-------------------
    # 5. 正解率の計算
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def accuracy(self,X,Y,thre=0.5):
        P,_ = self.predict(X)
        
        # 正解率
        accuracy = np.mean(Y==P)
        return accuracy
    #-------------------

    #------------------- 
    # 6. データと線形モデルのプロット
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName: 画像の保存先（文字列）
    def plotResult(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        if X.shape[1] != 1: return
        
        fig = plt.figure(figsize=(8,5),dpi=100)
        
        # 線形モデルの直線の端点の座標を計算
        Xlin = np.array([[0],[np.max(X)]])
        Yplin = self.predict(Xlin)

        # データと線形モデルのプロット
        plt.plot(X,Y,'.',label="データ")
        plt.plot(Xlin,Yplin,'r',label="線形モデル")
        plt.legend()
        
        # 各軸の範囲とラベルの設定
        plt.ylim([0,np.max(Y)])
        plt.xlim([0,np.max(X)])
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)
        
        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #------------------- 

    #-------------------
    # 7. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×１のnumpy.ndarray）
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # title: タイトル（文字列）
    # fName: 画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        plt.clf()
        fig = plt.figure(figsize=(6,6),dpi=100)
        plt.close()
        
        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],'bx',markerSize=14,label="ラベル-1")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markerSize=14,label="ラベル1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),100),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),100))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        Pmesh,_ = self.predict(Xmesh)
        Pmesh = np.reshape(Pmesh,X1.shape)

        # 予測値のプロット
        CS = plt.contourf(X1,X2,Pmesh,cmap="bwr",alpha=0.3,vmin=0,vmax=1)
        CS = plt.contour(X1,X2,Pmesh,linewidths=2,colors=['black'],levels=[0])

        # 各軸の範囲とラベルの設定
        plt.axis('square')
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.title(title,fontSize=14)
        plt.xlabel(xLabel,fontSize=14)
        plt.ylabel(yLabel,fontSize=14)
        plt.legend()
        
        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------
