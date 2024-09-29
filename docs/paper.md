# Segment the LLM using Subgraphs

## Based on the paper: 

  - [LinkedLingual](https://aclanthology.org/2024.acl-demos.16.pdf)


### Key points from the **LinguaLinked**:

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




[Parellelism](https://aclanthology.org/2024.naacl-industry.1.pdf)



Reference: 
  - 