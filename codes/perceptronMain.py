# -*- coding: utf-8 -*-
import numpy as np
import perceptron as perceptron
import data

#-------------------
# 1. データの作成
myData = data.classification(negLabel=-1,posLabel=1, path = '../data')
myData.makeData(dataType=3)
#-------------------

#-------------------
# 2. データを学習と評価用に分割
dtrNum = int(len(myData.X)*0.9)  # 学習データ数
# 学習データ（全体の90%）
Xtr = myData.X[:dtrNum]
Ytr = myData.Y[:dtrNum]

# 評価データ（全体の10%）
Xte = myData.X[dtrNum:]
Yte = myData.Y[dtrNum:]
#-------------------

#-------------------
# 3. 入力データの標準化
xMean = np.mean(Xtr,axis=0)
xStd = np.std(Xtr,axis=0)
Xtr = (Xtr-xMean)/xStd
Xte = (Xte-xMean)/xStd
#-------------------

#--------------------
# 4. perceptron モデルの学習
for seed in range(0,80,10):
    myModel = perceptron.perceptron( Xtr, Ytr, seed = seed )

    acc = myModel.accuracy(Xtr,Ytr)
    ite = 0
    while acc < 1.0:
        if ite%1==0:
            print(f"反復:{ite}")
            print(f"モデルパラメータ:\nw={myModel.w},\nb={myModel.b}")
            print(f"正解率={myModel.accuracy(Xte,Yte):.2f}")
            print("----------------")
            
        # モデルパラメータの更新
        myModel.update(alpha=0.001, mergin=1.0)
        acc = myModel.accuracy(Xtr,Ytr)
        ite += 1
        myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/perceptron_train_{seed:02}_{ite:03}.png")
        #myModel.plotResult(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/perceptron_result_train_{myData.dataType}_{ite:03}.png")
    myModel.plotModel2D(X=Xtr,Y=Ytr,xLabel=myData.xLabel,yLabel=myData.yLabel,fName=f"../results/perceptron_train_{seed:02}_end.png")

