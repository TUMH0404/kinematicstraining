# coding: utf-8
import common
import numpy as np
# jupyter notebookを使う人はコメントアウトする
#%matplotlib inline


## ここで，ファイル名を指定する。
name = ["20180302_8.csv","20180302_8.csv"]
ff = ["open","close"]

def COP(fname,figname):
    datcop=common.Text2Numpy(filename=fname)
    r = common.FFT_cop(freq=120,df=datcop["cop"],start=100,end=2500,bias=False)[:,1:3]
    matrix,center,ang,d0,d1 = common.COPpca(r)
    common.FIGshow(figname,d0,d0@common.Rotation(matrix,0))
    dat = d0@common.Rotation(matrix,0)
    #矩形面積と総軌跡長
    a = np.max(dat,axis=0)-np.min(dat,axis=0)
    rectarea = a[0]*a[1]

    dat1 = np.diff(dat,axis=0)
    totaltraj = np.sum(np.linalg.norm(dat1,ord=2,axis=1))
    print(f"矩形面積:{rectarea}")
    print(f"総軌跡長：{totaltraj}")
    print(f"補正した角度：{ang}")
    print(f"中心点：x_{center[0]},y_{center[1]}")

    
for n1,n2 in zip(name,ff):
    print(f"{n2}\n")
    COP(n1,n2)
    print("\n\n")