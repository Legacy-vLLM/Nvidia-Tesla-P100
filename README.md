# Nvidia-Tesla-P100
# **Running vLLM on an NVIDIA Tesla P100 with LoRA,Paged Attention e GPTQ enabled.**

In this repository you will find the **vLLM Legacy** Compatible with the Nvidia Tesla P100 16GB HBM2.

The Nvidia Tesla P100 16GB boasts impressive hardware; its memory works at...**\~730 GB/s of memory bandwidth (HBM2) at 4096 bits (yes, 4096 and not 384 bits).**.)

However, it was launched in 2016, a year before the launch of graphics cards with "Tensor Cores" (current technology) with the emergence of… **Tensor Cores a** The Nvidia Tesla P100 ended up being abandoned in terms of software, which is why this project came about.

The **Legacy vLLM P100** resolves this bottleneck. We've ported the crucial optimizations from...***vLLM Official*** Back to the Pascal architecture in this customized version, allowing you to reactivate your P100s and serve modern models with enterprise performance and full compatibility.

**TESTED MODELS:**

| MODEL | THROUGHPUT DECODE \- 1 GPU |
| :---: | :---: |
| **Qwen/Qwen3-1.7B-GPTQ-Int8** | **107 tk/s**  |
| Qwen/Qwen 2.5 1.5B | **71 tk/s** |
| Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 | 45 tk/s |
| **microsoft/Phi-3.5-vision-instruct** | **40 tk/s** |
| JunHowie/Qwen3-8B-GPTQ-Int8 | 45 tk/s |

## 

## 

## **O Status do Legacy-vLLM na P100 (Compute 6.0)**

| Resource | Working on the P100? | Technical Details |
| ----- | ----- | :---: |
| **xFormers Attention** | **activated** | xFormers attention kernels running stably with modern vLLM. |
| **Paged Attention** | **activated** |  When you go from 5 to 20 or 50 requests, the total value of tokens/s ( throughput generation) **rooms** **drastically**.  |
| **CUDA Graphs** | **activated** | Fully supported at the CUDA 12.4 driver level on the P100. This greatly helps mitigate overhead in batch processing.*batch sizes*). |
| **OpenAi API** | **activated** | Working with performance metrics formatted for Prometheus. |
| **KV Cache by GPU** | **activated** | HBM2 memory management (which in the P100 has an excellent \~730 GB/s of bandwidth) works perfectly with vLLM dynamic allocation. |
| **Batching** | **activated** | With the dynamic allocation of vLLM. |
| **Embedding** | **activated** | Working. |
| **LoRA** | **activated** | Working. |

With the**Paged Attention (via xFormers)**, **Scheduler**, **CUDA Graphs **and** KV Cache** Activated on the P100, we reached the absolute peak of what the P100 hardware can deliver. It's the pinnacle of portability and an excellent product for reviving these hardware systems in data centers around the world.

## **Integrated Enterprise-Class Features**

* **vLLM Engine V0 Port** Automatic and transparent conversion of native GPTQ models to float16 (half)... for maximum mathematical efficiency in Pascal kernels.

* **XFormers Active Backend:** Replaces*FlashAttention-2*, ensuring low attention latency.

* **CUDA Graphs**: Captures static graphs of active execution, reducing CPU overhead during decoding in the warmup.

* **GPTQ models:** Automatic and transparent conversion of native GPTQ models to float16 (half) Bfloat16 para float16for maximum mathematical efficiency in Pascal kernels.

* **API OpenAI Compatible:** Ready to integrate directly with LangChain, LlamaIndex, LiteLLM, or any existing application that points to port 8000\.

**This is sufficient to serve:**

* Local chatbots;  
* RAG;   
* agents;  
* automation;  
* Private APIs;  
* OpenWebUI or LiteLLM integrations…

## 

## **How to Run the Legacy-vLLM Docker Test (12-Hour Trial)**

We offer a free trial version so you can validate the system's stability directly on your hardware.

### **1\. Prepare your Docker environment.**

Make sure you have Docker installed (If you already have it, go to Step 2\)

*Linux*

*curl \-fsSL https://get.docker.com \-o get-docker.sh && sudo sh get-docker.sh*

*Windows*

[*https://www.docker.com/get-started/*](https://www.docker.com/get-started/)

### **2\. Install the Nvidia Container Toolkit startup.**

Make sure you have the **nvidia-container-toolkit** properly configured on your host,

Modern Docker uses a new specification called **CDI (Container Device Interface)**to communicate with the GPUs.

**Linux:**

**Step 1: Update the NVIDIA Container Toolkit repositories**

curl \-fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg \--dearmor \-o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \\ && curl \-s \-L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\ sed 's\#deb https://\#deb \[signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\] https://\#g' | \\ sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

### 

### **Step 2: Install the Toolkit**

Update the apt and install the packagenvidia-container-toolkit:

sudo apt-get update  
sudo apt-get install \-y nvidia-container-toolkit

**Step 3:**

Open the file in the terminal using the nano editor (or your preferred editor):

sudo nano /etc/docker/daemon.json

1. If the file is empty, paste the content below. If it already contains something, add the key."runtimes"and define the"default-runtime"For NVIDIA to manage the containers:

JSON

{  
    "runtimes": {  
        "nvidia": {  
            "path": "nvidia-container-runtime",  
            "runtimeArgs": \[\]  
        }  
    },  
    "default-runtime": "nvidia"  
}

2. Save the file (in Nano, pressCtrl \+ O, after Enter to confirm, and Ctrl \+ X (to leave).

### **Step 4: Create the CDI folder and generate the devices.**

Since modern Docker uses CDI to map your P100 GPU, we need to ensure that the default directory exists and that the YAML file containing your video card metadata is created manually:

1. Create the default CDI folder on the system:

sudo mkdir \-p /etc/cdi

2. Force the toolkit utility to map the GPUs installed on your physical machine and write the configuration file:

sudo nvidia-ctk cdi generate \--output=/etc/cdi/nvidia.yaml

### **Step 5: Restart the Docker service.**

Now that the manual settings are in place and the CDI file has been generated, restart the Docker daemon to load the new runtime:

sudo systemctl restart docker

### **5\. Request your 12-Hour Trial Key**

Get your test key using this form:

[https://docs.google.com/forms/d/e/1FAIpQLSeJw9ZslKvIkDJtKjlVbirIJdfXlDiFhXMoQPivBWII\_Ok8wQ/viewform?usp=header](https://docs.google.com/forms/d/e/1FAIpQLSeJw9ZslKvIkDJtKjlVbirIJdfXlDiFhXMoQPivBWII_Ok8wQ/viewform?usp=header)

### **6\. Download the Legacy-Vllm docker image.**

docker pull legacyvllm/legacy-vllm-tesla-p100-openai-lora:020826

### **Execute the initialization command in the terminal:**

docker run \--gpus all \
\-e LEGACY\_VLLM\_LICENSE\_KEY=" \*\*Request your trial key and enter it here\*\*" \
    \-p 8000:8000 \
    \-v \~/.cache/huggingface:/root/.cache/huggingface \
legacy-vllm\\p100-openai-lora:310726 \
    \--model Qwen/Qwen2.5-1.5B-Instruct \
    \--gpu-memory-utilization 0.8 \
    \--max-model-len 2048

### 

## **Security and Licensing Model**

To meet stringent corporate security policies (including closed data centers and environments)*air-gapped*(without internet access), our licensing system uses **offline asymmetric cryptography**:

1. The Docker image contains only our **public key**.

2. The expiration time check is calculated locally at container startup.

3. **No data is sent to external servers.**Your privacy and compliance with the LGPD/GDPR are fully protected.

## 

## **Commercial Licensing**

Did you enjoy the trial? We offer monthly or annual licenses (we generate keys for your GPU cluster).

To obtain your key or commercial proposals, contact us: **https://docs.google.com/forms/d/e/1FAIpQLSeJw9ZslKvIkDJtKjlVbirIJdfXlDiFhXMoQPivBWII\_Ok8wQ/viewform?usp=header**

License & Copyright: All Rights Reserved. This repository is for demonstration and portfolio purposes only. Unauthorized commercial use, redistribution, or modification is prohibited. Contains third-party components under Apache 2.0 (see NOTICE).

This product includes software developed by the vLLM project, licensed under the Apache License 2.0, and intellectual property modifications.
