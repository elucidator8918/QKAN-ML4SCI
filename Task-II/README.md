## Task II: Classical Graph Neural Network (GNN)

For Task II, you will use ParticleNet’s data for Quark/Gluon jet classification available here with its corresponding description.

- Choose 2 Graph-based architectures of your choice to classify jets as being quarks or gluons. Provide a description on what considerations you have taken to project this point-cloud dataset to a set of interconnected nodes and edges.

- Discuss the resulting performance of the 2 chosen architectures.

## Graph Construction Considerations for Quark/Gluon Jet Classification

The point-cloud jet data was structured as a graph where:  

- **Nodes** represented particles with 4-momentum features $(p_T, \eta, \phi, E)$.  
- **Edges** were formed based on particle proximity in a bidirectional form, determined by $\Delta R = \sqrt{(\Delta \eta)^2 + (\Delta \phi)^2}$ ensuring no unlocalized/isolated nodes.  
- **GAT's advantage** stemmed from its attention heads, which dynamically weighted neighboring nodes, emphasizing high-energy particles critical for classification and **GCN**, with uniform aggregation, struggled to prioritize such features. 

## Performance Comparison of GCN and GAT for Quark/Gluon Jet Classification

The **Graph Attention Network (GAT)** demonstrated stronger performance than the **Graph Convolutional Network (GCN)** in classifying quark/gluon jets. The key metrics reveal clear differences:

| Metric               | GAT (Best)       | GCN (Best)       | Improvement |
|----------------------|------------------|------------------|-------------|
| **Test AUC**         | 0.879           | 0.871           | +0.008      |
| **Test Accuracy**    | 80.4%           | 79.5%           | +0.9%       |
| **Best Val AUC**     | 0.876 (epoch 38)| 0.868 (epoch 13) | +0.008      |
| **Training Epochs**  | 48              | 23              | +25         |

GAT's attention mechanism provided better feature learning, evidenced by its higher test metrics and sustained validation improvements. GCN plateaued earlier, suggesting limitations in its fixed aggregation approach.

## Training Dynamics and Efficiency  

- **GAT** required slightly longer training times (~1.5 min/epoch vs. ~1.3 min/epoch for GCN), with one outlier epoch (38) taking 8 minutes due to attention computations.  
- Both models used early stopping (10 epochs without improvement). GAT trained longer (48 epochs vs. 23 for GCN), suggesting its attention mechanism allowed for extended learning without overfitting.  
- **GCN** converged faster initially but stagnated, likely due to its fixed neighborhood aggregation, which lacks adaptive feature weighting.    
- **GAT's advantage** stemmed from its attention heads, which dynamically weighted neighboring nodes, emphasizing high-energy particles critical for classification. **GCN**, with uniform aggregation, struggled to prioritize such features.

[🔗 View Report on Weights & Biases](https://wandb.ai/sololicht/lightning_logs/reports/Graph-Neural-Network-for-Quark-Gluon-Jet-Classification--VmlldzoxMjA4ODk2OA)
