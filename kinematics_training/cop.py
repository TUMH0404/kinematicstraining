import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class COP:    
    def __init__(self, fn):
        df0 = pd.read_csv(fn, encoding='shift_jis', header=None, nrows=6)
        df = pd.read_csv(fn, encoding='shift_jis', skiprows=7, header=None)
        self.__r = np.array([df[1], df[2]]).T
        pca = PCA(n_components=2)
        pca.fit(self.__r)
        self.__df0 = df0
        self.__df = df.iloc[:,0:9]
        self.__pca = pca

    @property
    def personal_info(self):
        lis = [self.__df0.iloc[0,1]]
        if self.__df0.iloc[2,1]=='男':
            lis.append('m')
        elif self.__df0.iloc[2,1]=='女':
            lis.append('f')
        else:
            lis.append('unknown')
        try:
            lis.append(int(float(self.__df0.iloc[3,1])))
        except:
            lis.append(self.__df0.iloc[3,1])
        try:
            lis.append(float(self.__df0.iloc[4,1]))
        except:
            lis.append(self.__df0.iloc[4,1])
        try:
            lis.append(float(self.__df0.iloc[5,1]))
        except:
            lis.append(self.__df0.iloc[5,1])
        return lis
    
    @property
    def original_DF(self):
        self.__df.columns=['t','x','y','Lx','Ly','Lweight','Rx','Ry','Rweight']
        return self.__df
        
    @property
    def original_r(self):
        return self.__r
    
    @property
    def transformed_r(self):        
        if self.__pca.components_[0,1] < 0:
            return -self.__pca.transform(self.__r)
        else:
            return self.__pca.transform(self.__r)

    @property
    def contribution_ratio(self):
        return self.__pca.explained_variance_ratio_
    
    @property
    def rectangle_areas(self):
        rO = self.original_r
        rT = self.transformed_r
        rO_range = np.max(rO, axis=0) - np.min(rO, axis=0)
        rT_range = np.max(rT, axis=0) - np.min(rT, axis=0)        
        return {'original':rO_range[0]*rO_range[1], 'transformed':rT_range[0]*rT_range[1]}
    
    def draw_track(self, transformed=True):
        if transformed == False:
            r = self.original_r
            center = self.__pca.mean_
            range_min = np.min(r.T[1]) - center[1]
            range_max = np.max(r.T[1]) - center[1]
            legend_label = 'original r'
        else:
            r = self.transformed_r
            center = np.array([0, 0])
            range_min = np.min(r.T[0])
            range_max = np.max(r.T[0])
            legend_label = 'transformed r'
        if abs(range_min) > range_max:
            range_tmp = abs(range_min)
        else:
            range_tmp = range_max
        range_lim = abs(range_tmp)*1.25
        plt.figure(figsize=(10,10))
        plt.plot(r.T[0], r.T[1])
        plt.xlim(-range_lim + center[0], range_lim + center[0])
        plt.ylim(-range_lim + center[1], range_lim + center[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend([legend_label])
        plt.grid()
        plt.show()
    
    def draw_time_series(self):
        r = self.transformed_r
        t = np.arange(0, 3.01, 0.01)
        plt.figure(figsize=(12,8))
        plt.plot(r.T[0])
        plt.plot(r.T[1])
        plt.xlabel('time(sec)')
        plt.ylabel('position(cm)')
        plt.xticks([0,500,1000,1500,2000,2500,3000], [0,5,10,15,20,25,30])
        plt.legend(['x', 'y'])
        plt.grid()
        plt.show()


        
