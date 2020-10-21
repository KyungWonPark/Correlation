import numpy as np
import plotly.express as px
import plotly

def loadCSV(path):
    return np.loadtxt(path, delimiter=",")

def zeroTest(e):
    if np.abs(e) < 0.00000001:
        return True
    else:
        return False

def getZeroEigValCount(eigVal):
    cnt = 0
    for e in eigVal:
        if zeroTest(e):
            cnt = cnt + 1
        else:
            break
            
    return cnt

class Cluster:
    def __init__(self, identifier):
        self.identifier = identifier
        self.size = 0
        self.members = []
        
        return
    
    def addMember(self, idx):
        self.members.append(idx)
        self.size = self.size + 1
        
    def delMember(self, idx):
        self.members.remove(idx)
        self.size = self.size - 1
        return
    
def cSort(Cluster):
    return Cluster.size

def getSortedRealClusters(clusters):
    realClusters = []
    
    for i in range(len(clusters)):
        c = clusters[i]
        if c.size >= 2:
            realClusters.append(c)
            
    realClusters.sort(key=cSort, reverse=True)
    
    return realClusters

def getClusters(eigVal, eigVec): # Returns only essential clusters
    clusters = []
    clusterCnt = getZeroEigValCount(eigVal)
    
    isAllocated = [{'clusterIdx': -1, 'value': 0} for x in range(len(eigVal))]
    
    for i in range(clusterCnt):
        clusters.append(Cluster(eigVal[i]))
        
        c = clusters[i]
        for j in range(len(eigVec[i])):
            if not zeroTest(eigVec[i][j]):
                if isAllocated[j]['clusterIdx'] == -1:
                    c.addMember(j)
                    isAllocated[j]['clusterIdx'] = i
                    isAllocated[j]['value'] = eigVec[i][j]
                else:
                    if np.abs(eigVec[i][j]) > np.abs(isAllocated[j]['value']):
                        clusters[isAllocated[j]['clusterIdx']].delMember(j)
                        c.addMember(j)
                        isAllocated[j]['clusterIdx'] = i
                        isAllocated[j]['value'] = eigVec[i][j]
                
    realClusters = getSortedRealClusters(clusters)
    
    return realClusters

def getXYZ(idx, greyList):
    return greyList[idx][0], greyList[idx][1], greyList[idx][2]

def getANC(clusters):
    accSize = 0
    for i in range(len(clusters)):
        accSize = accSize + clusters[i].size
        
    return accSize / len(clusters)

def getWCC(clusters, c2):
    accAvgPearson = 0
    for i in range(len(clusters)):
        c = clusters[i]
        
        accPearson = 0
        n = c.size
        for idx0 in range(n):
            for idx1 in range(0, idx0 + 1):
                idxFrom = c.members[idx0]
                idxTo = c.members[idx1]
                accPearson = accPearson + c2[idxFrom][idxTo]
 
        accAvgPearson = accAvgPearson + (accPearson / (n * (n + 1) / 2))
        
        return accAvgPearson
    
def printC(Cluster):
    print("ID: " + str(Cluster.identifier))
    print("MemberCnt: " + str(Cluster.size))
    
    return

import pandas as pd

def dfOneCluster(clusters, greyList, idx):
    data = []
    
    # Add background
    if clusters[idx].size < (13362 / 2):
        for voxel in greyList:
                point = [-1, 'Background', voxel[0], voxel[1], voxel[2], 2]
                data.append(point)
    
    c = clusters[idx]
    for i in range(c.size):
        voxelIdx = c.members[i]
        [x, y, z] = getXYZ(voxelIdx, greyList)
        size = int(np.log((13362 * 6) / c.size)) + 2
        point = [voxelIdx, str(idx), x, y, z, size]
        data.append(point)
        
    df = pd.DataFrame(data, columns = ['id', 'cluster', 'x', 'y', 'z', 'size'])
    return df
 
def dfNCluster(clusters, greyList, n):
    data = []

    if n > len(clusters):
        n = len(clusters)
        
    # Add background
    if clusters[0].size < (13362 / 2):
        for voxel in greyList:
                point = [-1, 'Background', voxel[0], voxel[1], voxel[2], 2]
                data.append(point)
        
    for i in range(n):
        c = clusters[i]
        
        for j in range(c.size):
            voxelIdx = c.members[j]
            [x, y, z] = getXYZ(voxelIdx, greyList)
            size = int(np.log((13362 * 6) / c.size)) + 2
            point = [voxelIdx, str(i), x, y, z, size]
            data.append(point)
        
    df = pd.DataFrame(data, columns = ['id', 'cluster', 'x', 'y', 'z', 'size'])
    return df

def save3DFigure(path, df):
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', opacity=0.4, size='size', size_max=16)
    fig.update_layout(
        scene = dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Z Coordinate",
            xaxis = dict(nticks=5, ticks='outside', tickwidth=4, range=[0, 46], backgroundcolor="rgb(200, 200, 230)"),
            yaxis = dict(nticks=10, ticks='outside', tickwidth=4, range=[0, 55], backgroundcolor="rgb(230, 200,230)"),
            zaxis = dict(nticks=5, ticks='outside', tickwidth=4, range=[0, 46], backgroundcolor="rgb(230, 230,200)"),
        )
    )
    
    plotly.offline.plot(fig, filename=path, auto_open=False)
    return

# c2 = loadCSV(RESULTDIR + "c2-tilda.csv")
# greyList = loadCSV(RESULTDIR + "greyList.txt")
# eigVal = loadCSV(RESULTDIR + "eigen-value-thr-0.000000.csv")
# eigVec = loadCSV(RESULTDIR + "eigen-vector-thr-0.000000.csv")

# Init
RESULTDIR = "/home/iksoochang2/kw-park/Result/"
OUTPUTDIR = "/home/iksoochang2/kw-park/Result/Analysis/"

c2 = loadCSV(RESULTDIR + "c2-tilda.csv")
greyList = loadCSV(RESULTDIR + "greyList.txt")

thrs = [
    "0.000000","0.050000","0.100000","0.150000","0.200000","0.250000","0.300000","0.350000","0.400000","0.450000",
    "0.500000","0.550000","0.600000","0.650000","0.700000","0.750000","0.800000","0.850000","0.900000","0.950000",
    "0.960000","0.970000","0.980000","0.990000",
]

for thr in thrs:
    print("Processing: " + thr)
    eigVal = loadCSV(RESULTDIR + "eigen-value-thr-" + thr + ".csv")
    eigVec = loadCSV(RESULTDIR + "eigen-vector-thr-" + thr + ".csv")

    cls = getClusters(eigVal, eigVec)
    df = dfNCluster(cls, greyList, 12)

    save3DFigure(OUTPUTDIR + "thr-" + thr + "-12-largest-clusters.html")