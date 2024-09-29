# **Segmentation of the llms**

## **Current Implementation of Neural Network**



The current implementation of neural network are based on Dictionaries, List and Queue. Here for efficient training, inferencing and deployment we can make use of Computational graphs. 


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

Pros: Allows for optimizations like memory reuse and graph-level optimizations.
Cons: Less flexible, requires redefinition of the graph for dynamic changes.

#### **Dynamic Graphs (Define-by-Run):**

The graph is constructed on-the-fly as operations are executed (e.g., PyTorch, TensorFlow 2.x).

Pros: More flexible and intuitive, especially useful for tasks like recursive neural networks or variable-length sequences.
Cons: Can have slightly higher overhead during execution due to graph construction.
vbnet

### **Example: Computational Graph for a Simple Function**

![Functional representation of Computational Graphs](https://sslprod.oss-cn-shanghai.aliyuncs.com/stable/slides/computational_graph_backpropagation_jij68v/computational_graph_backpropagation_jij68v_1440-05.jpg)


Suppose we have a simple function:  
 - \[ z = (x + y) \cdot w \]

Where:  
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

[Basic Implementation in Colab - On Mathematic Expressions](https://colab.research.google.com/github/datasith/ML-Notebooks-TensorFlow/blob/main/Intro_Computational_Graphs.ipynb#scrollTo=DxyJDoMOs1gu)


[For more information: Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)



