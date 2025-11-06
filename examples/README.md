# Examples
In the [Kubetorch Examples repo](https://github.com/run-house/kubetorch-examples), 
there are self-contained examples that use Kubetorch for various use cases 
(training, inference, data processing, production workflows, etc). 
Many of the examples contain markdown and are rendered as examples 
on [our site](https://www.run.house/examples).

Kubetorch supports a broad range of use-cases, including: 

* Distributed Training: [Hello World DDP](https://github.com/run-house/kubetorch-examples/blob/main/pytorch_ddp/pytorch_ddp.py), [ResNet](https://github.com/run-house/kubetorch-examples/blob/main/pytorch_ddp/resnet/resnet_training.py), [Pre-Emption Fault-Tolerance](https://github.com/run-house/kubetorch-examples/blob/main/fault_tolerance/dynamic_world_size.py)
* Batch Inference and Data Processing: [OCR](https://github.com/run-house/kubetorch-examples/blob/main/batch_inference/simple_deepseek_ocr.py), [Embeddings](https://github.com/run-house/kubetorch-examples/blob/main/batch_inference/embedding_batch_inference.py)
* Online Inference: [Llama + vLLM](https://github.com/run-house/kubetorch-examples/blob/main/vllm_inference/llama.py), [Triton](https://github.com/run-house/kubetorch-examples/blob/main/triton/embedding.py)
* RL: [Async GRPO](https://github.com/run-house/kubetorch-examples/blob/main/reinforcement_learning/async_grpo/gsm8k_async_simple.py), [VERL](https://github.com/run-house/kubetorch-examples/blob/main/reinforcement_learning/verl_training/verl_train.py)
* Ray Workloads: [Ray Hello World](https://github.com/run-house/kubetorch-examples/blob/main/ray/ray_hello_world/ray_hello_world.py), [Ray Serve + Data](https://github.com/run-house/kubetorch-examples/blob/main/ray/ray_ocr/ray_data_serve_ocr.py), [Ray Training](https://github.com/run-house/kubetorch-examples/blob/main/ray/dlrm-movielens/dlrm_training.py)

If you have a specific use case that you don't see here, we probably have it in our archives! 
Feel free to send us an email at [support@run.house](mailto:support@run.house).
