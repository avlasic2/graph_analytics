# Link predictions of weighted directed acyclic graphs
The algorithm is impementation of the patent pending method titled [Entity resource recommendation system based on interaction vectorization](https://patentimages.storage.googleapis.com/0a/ed/f5/47c57355c3ef30/US20200151597A1.pdf). The method is based of off [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [Complex Embeddings for Simple Link Prediction](https://arxiv.org/pdf/1606.06357.pdf) which seeks to minimize 
```math
|| embed1.T*Matrix*embed2 - E[log(weight)] ||
```
by learning the embeddings of each node (embedx) and the linear operator (Matrix) by considering the temporal graph over a fixed period of time. The temporal nature of the graph is captured by calculating the expected logarithmic weight. The linear operator captures the nature of the underlying asymmetric interaction of the nodes. 

This technique can be extended to a heterogeneous graph by learning a different linear operator for each node type with respect to source node and target node. 
