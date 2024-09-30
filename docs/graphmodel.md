# **Decoder as Graph Model**

**Based on Research Papers** <br>

## **Same Architecture, Different Representation**



The primary focus is to build a single transformer or decoder-only architecture. Since our main task is not predicting the next word, we can design an entire system that effectively processes and represents information in a graph format.

### **We can construct a GPT architecture using existing sources and frameworks. By leveraging the same frameworks, we can also develop a graph-based LLM.**

--- 


![LL](https://raw.githubusercontent.com/Vishnuprasadvbhat/redo_org/master/img/graph_lm.png)
source:**[mdpi](https://www.mdpi.com/2079-9292/12/4/793)**


## **Graph-to-Graph models**

Transformers operate as graph-to-graph models, with sequences representing a specific instance of this capability. In the Graph-to-Graph Transformer architecture, attention weights are considered graph edges. By integrating graph edges into the attention weight calculations and predicting these edges using attention-like functions, we explicitly incorporate graphs into the latent representations learned by pre-trained Transformers.

Furthermore, this approach introduces iterative graph refinement, creating a unified embedding of input, output, and latent graphs. This enables non-autoregressive graph prediction, optimizing the entire graph without requiring a specialized pipeline or decoding strategy. Empirical results show that this architecture achieves state-of-the-art accuracy in modeling various linguistic structures while effectively integrating with the latent linguistic representations acquired through pretraining.



## **Building a Graph-based LLM**

![LL](https://raw.githubusercontent.com/Vishnuprasadvbhat/redo_org/master/img/graphlangmodel.png)
source:**[Graph Language Model](https://aclanthology.org/2024.acl-long.245.pdf)**

While Language Models (LMs) are essential for Natural Language Processing (NLP), their interaction with structured knowledge graphs (KGs) remains an area of active research. Current methods either linearize KGs for embedding with LMs, which neglects structural information, or utilize Graph Neural Networks (GNNs), which fail to capture text features as effectively as pretrained LMs.

This work introduces a new model called the Graph Language Model (GLM), which combines the advantages of both approaches while addressing their limitations. The GLM is initialized from a pretrained LM to improve comprehension of individual graph concepts and relationships. Its architecture incorporates graph biases to facilitate effective knowledge distribution.

As a result, GLMs can process graphs, texts, and combined inputs. Empirical evaluations in relation classification tasks indicate that GLM embeddings outperform both LM- and GNN-based baselines in supervised and zero-shot settings, demonstrating their versatility.

## **Supporting Papers:**
  - [Graph Language Model](https://aclanthology.org/2024.acl-long.245.pdf)

  - [Transformers as Graph Model](https://arxiv.org/pdf/2310.17936)

  - [Optimizing Graph using Swarm](https://arxiv.org/pdf/2402.16823)


## **Reference:**
  - [Computational Graphs for Neural Networks](https://pharath.github.io/lecture_notes/machine_learning/notes-computational-graphs/)
  - [Minimal Code](https://github.com/jgsimard/computational_graph)
  - [Example code](https://github.com/tonegas/PyNet)





