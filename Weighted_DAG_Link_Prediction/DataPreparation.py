import os
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV



class ParseData():
    def __init__(self, rawdata:np.ndarray, isUniform:bool=True  ):
        super().__init__()
        self.elementToIndice = {}
        self.pairsWeightsArray = None
        self.data = rawdata
        self.isUniform = isUniform

    def empiricalPDF(self):
        def scores(pdf, data):
            scores = pdf.score_samples(data)
            return np.mean( scores[scores != float('-inf')] )

        bandwidths = np.arange(1,10.1,.1)
        kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']
        gridSearch = GridSearchCV(
                            KernelDensity()
                            , {'bandwidth': bandwidths, 'kernel' : kernels}
                            , scoring = scores
                        )
        weights = self.data[:,-1]
        gridSearch.fit( weights )
        return gridSearch.best_estimator_

    def calculatePairWeights(self):
        indice = 0
        pairWeightsDict = {}
        pairsWeightsLst = []
        PDF = None

        if self.isUniform:
            PDF = lambda x : 1
        else:
            PDF = self.empiricalPDF()
        for row in self.data:
            if row[0] not in self.elementToIndice:
                self.elementToIndice[ row[0] ] = indice
                indice += 1
            if row[1] not in self.elementToIndice:
                self.elementToIndice[ row[1] ] = indice
                indice += 1
            pair = ( self.elementToIndice[ row[0] ], self.elementToIndice[ row[1] ] )
            wt = PDF(row[2])
            if pair not in pairWeightsDict:
                pairWeightsDict[ pair ] = [ [ wt * np.log( row[2] ) , wt ] ]
            else:
                pairWeightsDict[ pair ].append( [ wt * np.log( row[2] ) , wt ] )
        for pair, weights in pairWeightsDict.items():
            weights = np.asarray(weights)
            pairsWeightsLst.append( [ pair[0] , pair[1] , weights[:,0].sum() / weights[:,1].sum()  ] )

        self.pairsWeightsArray = np.asarray(pairsWeightsLst)

    def saveResults(
             self
             , filePath:str
             , elementToIndiceFile:str='Element_Indice_Dictionary'
             , pairsWeightsArrayFile:str='Training_Data'
        ):
         with open( os.path.join( filePath, f'{elementToIndiceFile}.pkl'), 'wb') as f:
             pickle.dump( self.elementToIndice , f)
         f.close()

         np.save( os.path.join( filePath, f'{pairsWeightsArrayFile}.npy') , self.pairsWeightsArray )