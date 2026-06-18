# 🔬 TopoGAT: Dynamic GNN for Gigapixel Pathology

TopoGAT is an enterprise-grade deep learning pipeline for computational pathology. Traditional Vision Transformers struggle with gigapixel Whole Slide Images (WSI) due to massive memory explosions, while standard Graph Convolutional Networks (GCNs) suffer from over-smoothing dense tissue features. 

TopoGAT solves both bottlenecks using a **Sparse Graph Attention (GATv2)** architecture combined with **Attention-Gated Topologies** to dynamically prune irrelevant cellular connections.

## 🚀 Key Features
* **Hardware-Aware Scaling:** Dynamically limits WSI patch extraction based on real-time VRAM to prevent Out-Of-Memory crashes.
* **Attention-Gated Topology:** Replaces fixed boundaries with a learnable sparsity gate to prevent over-smoothing.
* **Homoscedastic Hydra Loss:** A self-tuning, multi-task head that balances diagnosis, morphological reconstruction, and tissue clustering.
