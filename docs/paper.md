# **Segment the LLM using Subgraphs**

Segmenting an LLM involves dividing its computational graph into smaller, independent parts, typically to enable parallel execution, distributed inference, or efficient model deployment across different hardware resources. This is useful, especially when working with extremely large models that are too resource-intensive to run on a single device or need optimization for specific deployment scenarios (e.g., edge devices or multi-GPU setups).

## **Based on the paper** **[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)**: 

![LL](https://raw.githubusercontent.com/Vishnuprasadvbhat/redo_org/master/img/linguallinked.png) <br>
source:[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)

## **Adoptable Section**

## **Subgraph Extraction from LLMs**


![LL](https://raw.githubusercontent.com/Vishnuprasadvbhat/redo_org/master/img/systemdesign.png)
source:[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)



## **Methods to Segment LLMs Using Computational Graphs**

![LL](https://raw.githubusercontent.com/Vishnuprasadvbhat/redo_org/master/img/systemdesign.png)
source:[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)

*change it to parallelism

## 1. **Pipeline Parallelism (Layer-Wise Segmentation)**
- **Description**: The LLM is segmented by dividing its layers across different devices or nodes.
- **Process**:
  - The computational graph is split at the boundaries between layers.
  - Each segment is processed sequentially by different hardware components (e.g., GPUs or TPUs).
- **Example**: In a transformer model, assign layers 1-6 to GPU 1 and layers 7-12 to GPU 2.

### How it Works:
- In the forward pass, input data is processed layer-by-layer, with each segment handled by a different device.
- Once one segment finishes, the next device picks up the output from the previous one, like an assembly line.
- In the backward pass, gradients are propagated similarly.

### Benefits:
- Reduces memory footprint per device.
- Allows the use of multiple devices in parallel, improving scalability.

### Challenges:
- Communication overhead between devices can slow down training or inference.
- Latency due to synchronization between segments.

---

## 2. **Tensor Parallelism (Within-Layer Segmentation)**
- **Description**: Tensor parallelism divides the operations within a layer instead of segmenting by layers.
- **Process**:
  - Large tensors (e.g., weight matrices in self-attention or feedforward layers) are split across multiple devices.
- **Example**: For a large matrix multiplication, split the weight matrix into smaller blocks and distribute them across GPUs.

### How it Works:
- Each device performs its portion of the tensor operation simultaneously.
- The partial results from each device are combined at the end of the operation (e.g., using all-reduce operations).

### Benefits:
- Increases parallelism within each layer, speeding up computation.
- Effective for very large models where a single layer is too large to fit into the memory of a single device.

### Challenges:
- Synchronization and communication between devices can introduce overhead.
- May require sophisticated partitioning strategies to ensure efficient memory usage and load balancing.

---

## 3. **Model Sharding (Distributed Across Different Devices)**
- **Description**: Different parts of the computational graph are distributed across heterogeneous devices (e.g., CPU, GPU, edge devices).
- **Process**:
  - Specific segments of the model run on devices optimized for their computation.
- **Example**: Run the early layers of a transformer model on a powerful cloud GPU and the later layers on edge devices for faster localized inference.

### How it Works:
- The computational graph is segmented based on device capability, memory, and power requirements.
- Certain segments are offloaded to appropriate hardware (e.g., CPU for lightweight computation, GPU for heavy tensor operations).

### Benefits:
- Enables the use of edge devices or other constrained hardware in distributed environments.
- Can reduce communication latency if segments are placed close to where data is generated or consumed.

### Challenges:
- Requires efficient management of communication between heterogeneous devices.
- Segmenting must account for device-specific performance characteristics, such as memory capacity and computation speed.

---

## 4. **Task-Specific Segmentation**
- **Description**: LLMs can be segmented based on specific tasks.
- **Process**:
  - Task-specific layers can be offloaded to different hardware resources while shared layers remain on a central server.
- **Use Case**: Particularly useful for multi-task learning or when deploying a model that handles various related tasks.

---

## 5. **Graph Partitioning**
- **Description**: Some deep learning frameworks allow automatic partitioning of computational graphs based on hardware constraints.
- **Process**:
  - Analyze the computational graph and strategically split it into segments that can be executed independently or in parallel.

## **Frameworks and Tools for Segmenting LLMs**

- **DeepSpeed (Microsoft)**: Provides tools for pipeline parallelism and tensor parallelism, enabling segmentation of large models across multiple GPUs.
- **TensorFlow**: Offers APIs for distributing computation across devices, facilitating segmentation using strategies like model parallelism and data parallelism.
- **PyTorch**: The `torch.distributed` library allows flexible partitioning of the computational graph and distribution across multiple devices.
- **Hugging Face Transformers**: The `Accelerate` library provides tools to split large models across multiple GPUs or TPUs for efficient training and inference.
- **ONNX Runtime**: Allows ONNX models to be split into segments and optimized for different hardware, suitable for distributed deployment or model partitioning.

---
## **Preparation for Mobile Deployment:**

- **Segment into Subgraphs**: The graphs are divided into smaller, independent subgraphs that can work separately on different devices.
- **Key Node Identification**: 
  - Nodes that take inputs from a single source and provide outputs to multiple nodes are identified as key points for creating subgraphs.
  - These nodes usually represent distinct layers or operations in the model, making them suitable for independent execution.

### **Subgraph Dependency Search:**
- **Dependency Management**: To manage connections between nodes in different subgraphs, a dependency search algorithm is employed.
- **Two Key Maps**:
  1. **Residual Dependency Map (RDM)**: 
     - Tracks dependencies between non-adjacent subgraphs.
     - Identifies when a subgraph relies on nodes from an earlier, but not directly preceding, subgraph.
  2. **Sequential Dependency Map (SDM)**:
     - Monitors direct dependencies between adjacent subgraphs.
     - Ensures outputs from one subgraph are used as inputs for the next subgraph.

### **Model Assignment Optimization:**

- **Assign Subgraphs to Mobile Devices**: After segmenting LLMs into subgraphs, the next step is to allocate these subgraphs as executable modules on mobile devices.
- **Consider Device Constraints**: The assignment process takes into account device limitations to minimize computation and data transmission times.
- **Profiling and Optimization**:
  - Subgraphs are compiled into sub-modules and profiled for:
    - FLOP (Floating Point Operations) count.
    - Memory requirements.
    - Data output size.
  - A primary optimizer formulates a linear optimization problem to balance local computation and data transmission.
  - Constraints are set to ensure memory usage on each device does not exceed a predetermined limit of the device's available memory.






### **Key points from the** **[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)**:

- **Challenge**: Deploying Large Language Models (LLMs) locally on mobile devices is difficult due to high memory requirements.

- **Solution**: **LinguaLinked** is introduced as a system for decentralized, distributed LLM inference on mobile devices.

- **Data Privacy**: The system processes information locally, ensuring data privacy by preventing data from leaving the device.

- **Key Strategies**:
  1. **Optimized Model Assignment**:
     - Segments LLMs and employs linear optimization to align segments with the capabilities of each device.
  2. **Optimized Data Transmission**:
     - Ensures efficient and structured data flow between model segments while preserving the integrity of the original model structure.
  3. **Runtime Load Balancer**:
     - Actively monitors and redistributes tasks among mobile devices to prevent bottlenecks, enhancing overall efficiency and responsiveness.

- **Testing Results**:
    - Extensive testing demonstrates that **LinguaLinked** supports efficient LLM inference with consistent throughput and minimal latency across various mobile devices, including both high-end and low-end Android devices.


### Reference: 


**[LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)** <br>
**[Parellelism](https://aclanthology.org/2024.naacl-industry.1.pdf)**



