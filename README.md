# Model-agnostic Graph Explainability from Scratch

This repository holds a graph explainability solution, which extends the work ([***GraphMask Explainer***](https://arxiv.org/pdf/2010.00577.pdf)) to heterogeneous as well as homogeneous Graphs, making this functionality ***model-agnostic***. Moreover, this implementation provides both ***node feature-level*** and ***edge-level attributes mask***, which is a ***binary-valued*** vector. All “0” values of this ***mask vector*** represent those features (and edges) of the graph that do not affect their corresponding predictions, whereas features (and edges) associated with “1” values consider to be a lot effective in influencing their associated predictions outputted by the original Graph Neural Network (GNN) model.
## Requirements
-	`PyTorch Geometric`
-	`PyTorch`
-	`Numpy`
## Usage
### Data
This implementation of ***GraphMask Explainer*** demonstrates explainability examples for ***GCN***, ***GAT***, and ***RGCN*** *layer-types* on ***Node Classification (NC)***, ***Graph Classification (GC)***, and ***Link Prediction (LP)*** tasks.
| Layer Type | Task | Dataset|
| ---------- |:----:|:------:|
| GCN | NC | Cora |
|GCN | GC | Enzymes |
| GAT | NC | Cora |
| GAT | GC | Enzymes |
| GAT | LP | Cora |
| RGCN | NC | AIFB |
| RGCN | GC | Enzymes |
### Training and Testing
-	To see the ***model-agnostic*** explainability layer’s implementation, check `graphmask_explainer.py`.
-	To train and generate explanations for any of the aforementioned tasks, run `graphmask_explainer_example.py`.
-	All hyperparameters’ settings can be tweaked (based on requirements) by altering their corresponding values provided in both `graphmask_explainer.py` and `graphmask_explainer_example.py` files.
## Results

