# coding: utf-8

#################################
######      ココから     ########
#################################

## ライブラリのインポート ##
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## テキストデータをnumpy形式に変換する(解析のための下準備) ##
###  必ず使用する  ###
def Text2Numpy(**args):
    """
    大学にある重心動揺計の計測データ（csv）を取り込み，pythonで処理しやすいように変換するための関数
    filename:処理したいファイル名
    """    
    ## データを開く ##
    with open(args["filename"],"r",encoding="shift-jis") as dat:
        datacop= dat.read()

    ## 書き出し先を作る ##
    COPdata = {"ID":None,"sex":None,"age":None,"height":None,"weight":None,"cop":[]}
    attribute = {"ID":0,"sex":2,"age":3,"height":4,"weight":5}
    ## 処理1 ##
    ### データの整理 ###
    d0 = datacop.split("\n") # スプリット
    COPdata["ID"] = d0[attribute["ID"]].split(",")[1][1:-1]# IDの取り出し
    COPdata["sex"] = d0[attribute["sex"]].split(",")[1][1:-1]# 性別の取り出し
    COPdata["age"] = (d0[attribute["age"]].split(",")[1][1:-1])# 年齢の取り出し
    COPdata["height"] = (d0[attribute["height"]].split(",")[1][1:-1])# 身長の取り出し
    COPdata["weight"] = (d0[attribute["weight"]].split(",")[1][1:-1])# 体重の取り出し

    for i in d0[7:]:
        if len(i)==0:
            break
    
        nn = i.split(",")
        COPdata["cop"].append(list(map(float,nn[0:3])))

    COPdata["cop"] = np.array(COPdata["cop"])
    
    ## 処理1完了 ##
    return COPdata



## フィルタリング（ローパス）とバイアス ##
###  必ず使用する  ###
def FFT_cop(**ss):
    """
    フィルタリング（ローパス）とバイアス
    処理したデータが適切か図示して確認する
    df:FFTしたいデータ
    freq:パスしたい周波数(Hz)
    bias：取りたい場合はFalseをTrueに変える
    start:開始点
    end:終了点
    （もし，startとend間のデータ数が2のn乗個でない場合は自動的に2のn乗個にする）
    return:処理後のデータ，処理前（a）のデータ
    """
    
    v1 = ss["start"]
    v2 = ss["end"] 
    df = ss["df"]
   # wave = ss["wave"]
    bs = ss["bias"]
    samp = ss["sampling"]
    HZ = ss["hz"]
    
    myd = df[v1:v2,:]
    if np.log2(myd.shape[0])*10%10!=0:
        nn = np.log2(myd.shape[0])*10//10
        myd = df[v1:(2**int(nn)+v1),:]
    
    ww = int(myd.shape[0]*HZ/(1/samp))  ## データ数：１/サンプル間隔(Hz)＝波数:周波数(Hz)
                                        # hz = np.linspace(0,1.0/samp,myd.shape[0])[wave]
    x = FFT(myd[:,1],ww,bs)
    y = FFT(myd[:,2],ww,bs)    
    #x = FFT(myd[:,1],wave,bs)
    #y = FFT(myd[:,2],wave,bs)
    
   
    
    fig = plt.figure(figsize=(7,7))
    plt.plot(myd[:,1],myd[:,2])
    plt.plot(x,y)
    plt.title(f"The confirmation of a processing data on figure(wave:{ww})")
    
    return np.c_[myd[:,0],x,y]




## 主成分分析結果の取得 ##
###  必ず使用する  ###
def COPpca(dat):
    '''
    COPデータで主軸変換，次元削減のためのパラメータを抽出する。
    返り値：（固有ベクトル，固有ベクトルの角度，変換前のデータ（バイアスとる），変換後のデータ）
    '''
    pca = PCA(n_components=2)
    pca.fit(dat)
    
    # 共分散行列
    A = pca.get_covariance()
    
    # 固有値と固有ベクトル
    e_val, e_vec = np.linalg.eig(A)
    
    dd0 = dat - pca.mean_
    dat0 = dd0@e_vec
    
    fig = plt.figure(figsize=(5,5))
    plt.plot(dd0[:,0],dd0[:,1])
    plt.plot(dat0[:,0],dat0[:,1])
    plt.show()
    
    return pca.components_,pca.mean_, np.rad2deg(np.arccos(e_vec[0,0])), dd0, dat0


## データの表示と保存 ##
###  必ず使用する  ###
def FIGshow(pngfilename,*args):
    '''
    補正したCOPデータの図を描く。
    '''
    myd = list(args)
    col = ["k","b","m","r","g","c"]
    lim0 = [MaxMin(i) for i in myd]
    limmax = max(lim0)
    ll = []
    fig = plt.figure(figsize=(7,7))
    plt.grid()
    for n,d in enumerate(myd):
        plt.scatter(d[:,0],d[:,1],s=5,c=col[n])
        ll.append(str(n))
        
    plt.legend(ll)
    plt.ylim(-limmax,limmax)
    plt.xlim(-limmax,limmax)
    

    plt.savefig(f"{pngfilename}.png")
    

################################
######      ココまで     #######
################################


## 反時計回りにどれだけ回転させるか ##
def Rotation(mat,rot=None):
    '''
    反時計回りに90°のときは，1
    反時計回りに270°のときは，3
    反時計回りに180のときは，2
    反転したいときは，0
    反転した値を反時計回りに90°まわしたいときは，-1
    反転した値を反時計回りに180°まわしたいときは，-2
    反転した値を反時計回りに270°まわしたいときは，-3


    上記以外はNone(default)
    '''
    if rot == 2:
        Rmat = (-1)*mat
    elif rot ==1:
        Rmat = mat@np.array([[0, -1], [1, 0]])
    elif rot == 3:
        Rmat = (-1)*mat@np.array([[0,-1],[1,0]])
    elif rot == 0:
        Rmat = np.fliplr(mat)
    elif rot == -1:
        Rmat = np.fliplr(mat)@np.array([[0, -1], [1, 0]])
    elif rot == -2:
        Rmat = (-1)*np.fliplr(mat)
    elif rot == -3:
        Rmat = (-1)*np.fliplr(mat)@np.array([[0, -1], [1, 0]])
    else:
        Rmat = mat
    
    return Rmat



######## 重要  #########
def FFT(a,wave,bias):
    '''
    高速フーリエ変換
    '''
    xfft = np.fft.fft(a, n=None, axis=-1, norm=None)
    xfft[wave:(-1)*wave]=0
    if bias:
        xfft[0]=0
    xifft = np.fft.ifft(xfft)
    return xifft.real
######################



## サブ関数 ##
def MaxMin(dat):
    mm = np.min(dat)
    mx = np.max(dat)
    if mm<0:
        mm=-1*mm
    
    if mm>mx:
        lims = mm
    else:
        lims = mx
    
    lims = np.ceil(lims)
    return lims

def MM(dat1,dat2):
    A = MaxMin(dat1)
    B = MaxMin(dat2)
    if A>B:
        return (-A,A)
    else:
        return (-B,B)

#########################



## 以上  ##