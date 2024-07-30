<h2 align="center">
SGCN: Structure and Similarity-Driven Graph
Convolutional Network for Semi-Supervised
Classification
</h2>

## Abstract
Traditional Graph Convolutional Networks (GCNs) primarily utilize graph structural information for information aggregation, 
often neglecting node attribute information. This approach can distort node similarity, resulting in ineffective node feature 
representations and reduced performance in semi-supervised node classification tasks. To address these issues, we introduce 
a similarity measure based on the Minkowski distance to better capture the proximity of node features. Building on this, we 
propose SGCN, a novel graph convolutional network that integrates this similarity information with conventional graph 
structural information. To validate the effectiveness of SGCN in learning node feature representations, we introduce 
two classification models based on SGCN: SGCN-GCN and SGCN-SGCN. We evaluate their performance on semi-supervised node 
classification tasks using three benchmark datasets: Cora, Citeseer, and Pubmed. Experimental results demonstrate that our 
proposed models significantly outperform the standard GCN model in terms of classification accuracy, highlighting the 
superiority of SGCN in node feature representation learning. 
## Dependencies
SGCN requires:
- python (3.9.10)
- torch (2.2.1)
- torch_geometric(2.5.3)
- numpy (1.26.1)
- pandas (2.1.2)
- scipy (1.11.3)
- scikit-learn (1.3.2)
- torch-scatter (2.0.7)
- torch-sparse(0.6.18)
## Project directory structure
- `SGCN_code` is Project name.
- `SGCN_code/data`  is datasets.
- `SGCN_code/SGCN` is the   implementation of our SGCN.
- The other `.py `files respectively implement comparison methods such as MLP, deepwalk, GAT, GraphSAGE, etc.

## Reproduce the results

- Step1 install the same version of depedencies.
- Step2 Open the `SGCN.py` file, locate the function `self.conv1 = GCNConv(in_channels, hidden_channels)`, and 
proceed to the module where this function is implemented, `gcn_conv.py`.
- Step3 Replace the contents of the `gcn_conv.py` file with the contents of the `PYG_SGCN.py` file.
- Step4 run `SGCN.py` file to implement the model SGCN-GCN for semi-supervised node classification(Start 
by running one iteration, then stop the execution, comment out the line of code at line 115 in the `gcn_conv.py` file, 
and then proceed with the execution.
).
- Step5 Uncomment the lines of code from lines 192 to 198 in the `gcn_conv.py` file to implement semi-supervised 
node classification with the SGCN-SGCN model.

Attention 1: The function `cosine_similarity(x, edge_index)` in the `gcn_conv.py` file implements the calculation of 
the similarity matrix. Uncomment the corresponding code as needed.

Attention 2: Please replace the file paths in the code with your own file paths.



