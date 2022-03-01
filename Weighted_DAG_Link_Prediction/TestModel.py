import pandas as pd
from DataPreparation import ParseData
from model import EmbedClass

rawData = pd.read_csv('soc-sign-bitcoinalpha.csv.gz', compression='gzip', header=None)

columns = ['source','target','weight','time']
rawData.columns = columns

# add a value to ensure all weights are strictly positive
maxValue = 2*rawData.loc[:,'weight'].abs().values.max()
rawData['weight'] += maxValue


# the data needs to be a numpy array and have the sequential columns of:
# source-node index,
# target-node index
# expected logarithmic edge-weight

# the ParseData class only requires source-node character, target-node character, and raw-weight
rawData = rawData.loc[:, columns[:-1] ].values # train without the 'time' feature
dataClean = ParseData(rawData)
dataClean.calculatePairWeights()

model = EmbedClass(
      dimensionEmbed = 7
    , numberElements = len(dataClean.elementToIndice)
    , sourceIndices = dataClean.pairsWeightsArray[:,0]
    , targetIndices = dataClean.pairsWeightsArray[:,1]
    , weights = dataClean.pairsWeightsArray[:,2]
)


model._train( num_epoch = 30 )


