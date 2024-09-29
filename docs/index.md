# **Segmentation of the llms**

After our last meeting, we found that we were all working hard on developing a suitable model for edge deployment using various state-of-the-art methods such as pruning, fine-tuning, quantizing, and distillation. However, the segmentation of LLMs was still an area that we had not explored.

Furthermore, we discussed an existing method that could be incorporated into our research project for efficiently segmenting the LLM. The points I have expressed here are based on the research paper I have explored, but there may be other methods that outperform the proposed one.



## **Current Implementation of Neural Network**

The current implementation of neural network are based on Dictionaries, List and Queue. Here for efficient training, inferencing and deployment we can make use of Computational graphs. 
<br>


![Basic ANN](https://cdn-images-1.medium.com/max/1600/1*pbk9xtz7WbBwYPVATdl9Vw.png)

  We are going to replace the mathematical operations used in our neural network to the Node

## **Comparision of ANNs to that of Computational Graphs**


### **Graph representation of Neural Network:**

Computational graphs are a fundamental concept in machine learning and deep learning, representing the sequence of operations needed to compute a mathematical function. They provide a structured way to visualize and manage how data flows through various operations, enabling efficient computation and optimization, particularly for automatic differentiation (used in backpropagation).

![Graph representation of Neural Network](https://res.cloudinary.com/dyd911kmh/image/upload/v1658404111/neural_network_graph_f8afb378d4.png)


---

## **Why Are Computational Graphs Important?**

1. **Automatic Differentiation:**
The graph structure allows for efficient computation of gradients during backpropagation by systematically applying the chain rule.
2. **Modularity:**
Complex functions can be broken down into simpler operations, which can be reused and recomposed.
3. **Parallelization:**
Since the graph structure outlines dependencies, parts of the computation that are independent can be parallelized, improving efficiency on GPUs or distributed systems.
4. **Optimization:**
Frameworks like TensorFlow and PyTorch build computational graphs dynamically or statically, allowing for optimizations like memory management and operation fusion.

## **Types of Computational Graphs**

### **Static vs. Dynamic Computational Graphs**

#### **Static Graphs (Define-and-Run):**

The graph is defined once before execution (e.g., TensorFlow 1.x). Once the graph is constructed, it is executed as a whole.

**Pros:** Allows for optimizations like memory reuse and graph-level optimizations. <br>
**Cons:** Less flexible, requires redefinition of the graph for dynamic changes.

#### **Dynamic Graphs (Define-by-Run):**

The graph is constructed on-the-fly as operations are executed (e.g., PyTorch, TensorFlow 2.x).

**Pros:** More flexible and intuitive, especially useful for tasks like recursive neural networks or variable-length sequences. <br>
**Cons:** Can have slightly higher overhead during execution due to graph construction.
vbnet

### **Example: Computational Graph for a Simple Function**

![Functional representation of Computational Graphs](https://sslprod.oss-cn-shanghai.aliyuncs.com/stable/slides/computational_graph_backpropagation_jij68v/computational_graph_backpropagation_jij68v_1440-05.jpg)


**Suppose we have a simple function: ** 
 - \[ z = (x + y) \cdot w \]

**Where:**  
- \( x \), \( y \), and \( w \) are input variables.  
- \( + \) and \( \cdot \) are operations.

### **Corresponding Computational Graph**:

``` text
x ----+
      |
      +----( + )----> z
      |            |
y ----+            * 
                   |
w -----------------+

``` 



## **Convert LLMs into Computational Graphs**: <br>

<!-- ## **Steps to Convert LLMs into Computational Graphs** <br> -->

#### **Breakdown of LLM Components <br>**
  - **Overview**: LLMs, such as GPT, are essentially stacks of transformer layers.
  - **Each transformer layer contains operations like**:
    - Linear projections (matrix multiplications)
    - Multi-head self-attention (dot products, softmax, and weighted sums)
    - Layer normalization
    - Activation functions (e.g., GeLU, ReLU)
    - Feedforward neural networks
  - **Representation**: These operations can be represented as nodes in a computational graph.

  ---

#### Representing Forward Pass <br>
  - **Process**: In the forward pass, the input tokens (word embeddings) are passed through a series of transformer layers.
  - **Node Representation**:
    - Each layer's operations can be converted into a node in the graph.
    - The output of one node (operation) flows into the next.
  - **Self-Attention Mechanism**:
    - The self-attention mechanism itself is a subgraph.
    - Each step (e.g., attention scores calculation, softmax normalization) is broken down into individual operations.
  - **Graph Interaction**: The computational graph represents how each token interacts with others across layers.

  ---

#### **Backward Pass (Backpropagation)**
  - **Automatic Differentiation**: The backward pass is handled by automatic differentiation.
  - **Gradient Computation**:
    - Once the computational graph is constructed, frameworks like TensorFlow or PyTorch can automatically compute gradients for each parameter with respect to the loss.
    - This is done by traversing the graph in reverse.
  - **Efficiency**: This allows for efficient training of the model, optimizing the parameters using gradient descent.

  ---

#### Example: LLM Layer (Single Transformer Layer)
  - **Inputs**: Token embeddings
  - **Operations**:
    - Multi-head self-attention (with matrix multiplications, scaling, and softmax)
    - Add & Norm
    - Feedforward network (with activation function)
    - Add & Norm
  - **Outputs**: Transformed embeddings
  - **Graph Representation**: This can be represented as a directed acyclic graph (DAG), with each of these operations represented as nodes and the data flow between them as edges.


## **Key Concepts in Computational Graphs**

### **Nodes:**
- Each node in a computational graph represents a mathematical operation (e.g., addition, multiplication, activation functions) or a variable (e.g., input data, weights, biases).
- Input nodes hold the input data, and operation nodes perform functions on the data.

### **Edges:**
- The edges represent the flow of data between operations. They carry values (or tensors) that are passed from one operation to another.
- Edges define dependencies between nodes, ensuring that operations are computed in the correct order.

### **Directed Acyclic Graph (DAG):**
- Computational graphs are typically directed acyclic graphs (DAGs), meaning that data flows in one direction, and there are no cycles or loops in the graph.
- This ensures that the computation proceeds from inputs to outputs without infinite recursion.

### **Forward Pass:**
- During the forward pass, data flows through the graph from the input nodes, through the operations, and produces an output.
- This step calculates the predicted output based on the input data and model parameters.

### **Backward Pass (Backpropagation):**
- During the backward pass, the graph is used to calculate gradients of the loss function with respect to model parameters.
- Automatic differentiation (reverse-mode differentiation) is applied by traversing the graph backward, allowing efficient gradient calculation for optimization algorithms like gradient descent.


## **Use Cases for LLM Segmentation**

- **Distributed Training**: Essential for efficiently training very large LLMs (like GPT-3 or similar models) across multiple GPUs or nodes.
- **Edge Deployment**: Allows parts of the model to be deployed on edge devices while keeping others centralized.
- **Inference Pipelines**: Facilitates faster, parallelized processing for real-time applications by segmenting the LLM.

**[Basic Implementation in Colab - On Mathematic Expressions](https://colab.research.google.com/github/datasith/ML-Notebooks-TensorFlow/blob/main/Intro_Computational_Graphs.ipynb#scrollTo=DxyJDoMOs1gu)**


**[For more information: Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)**


