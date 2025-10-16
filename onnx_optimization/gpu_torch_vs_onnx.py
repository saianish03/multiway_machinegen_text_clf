import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil
import time
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_single_text(text, tokenizer):
    inputs = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt")
    return inputs

def measure_onnx_latency(data, tokenizer, export_model_path, output_dir):
    
    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    device_name = 'gpu'
    sess_options = onnxruntime.SessionOptions()
    # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
    # Note that this will increase session creation time so enable it for debugging only.
    # sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_{}.onnx".format(device_name))
    
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    
    session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    latency = []
    total_samples = 1000
    for i in range(total_samples):
        x = preprocess_single_text(data.iloc[i]['text'], tokenizer)
        inputs = {
            'input_ids':      x['input_ids'].cpu().numpy(),
            'input_mask': x['attention_mask'].cpu().numpy(),
        }
        start = time.time()
        ort_outputs = session.run(None, inputs)
        latency.append(time.time() - start)
        
    print("OnnxRuntime {} Inference time = {} ms".format(device_name, format(sum(latency) * 1000 / len(latency), '.2f')))
    return ort_outputs

def measure_pytorch_latency(data, model, tokenizer, device):
    # Measure the latency. It is not accurate using Jupyter Notebook, it is recommended to use standalone python script.
    latency = []

    total_samples = 1000
    with torch.no_grad():
        for i in range(total_samples):
            x = preprocess_single_text(data.iloc[i]['text'], tokenizer)
            inputs = {
                'input_ids':      x['input_ids'].to(device),
                'attention_mask': x['attention_mask'].to(device),
            }
            start = time.time()
            outputs = model(**inputs)
            latency.append(time.time() - start)
    print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))
    return outputs

def main():
    print("Loading dataset...")
    data = load_dataset("Sansh2003/subtask-b-examples-test")
    data = data['test']
    data = pd.DataFrame(data)

    MODEL_PATH = "Sansh2003/roberta-large-merged-subtaskB"
    id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    device = torch.device("cuda") # if use_gpu else "cpu")
    print(f"Using device {device}...")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels = len(id2label), id2label=id2label, label2id=label2id)
    model.to(device)
    output_dir = os.path.join(".", "onnx_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    export_model_path = os.path.join(output_dir, 'roberta-large-subtaskB-onnx.onnx')

    print("measuring pytorch latency...")
    print()
    outputs = measure_pytorch_latency(data, model, tokenizer, device)
    print()
    print("measuring onnx runtime latency...")
    print()
    ort_outputs = measure_onnx_latency(data, tokenizer, export_model_path, output_dir)
    print()
    print("measuring fp16 onnx runtime latency...")
    print()
    export_model_path = os.path.join(output_dir, 'roberta-large-subtaskB-onnx_gpu_fp16.onnx')
    fp16_ort_outputs = measure_onnx_latency(data, tokenizer, export_model_path, output_dir)
    print()
    print("***** Verifying correctness *****")
    for i in range(2):    
        print('PyTorch and ONNX Runtime output {} are close:'.format(i), np.allclose(ort_outputs[0][0][i], outputs[0][0][i].cpu(), rtol=1e-02, atol=1e-02))
        diff = ort_outputs[0][0][i] - outputs[0][0][i].cpu().numpy()
        max_diff = np.max(np.abs(diff))
        avg_diff = np.average(np.abs(diff))
        print(f'maximum_diff={max_diff} average_diff={avg_diff}')
        print()
        print('PyTorch and FP16 ONNX Runtime output {} are close:'.format(i), np.allclose(fp16_ort_outputs[0][0][i], outputs[0][0][i].cpu(), rtol=1e-02, atol=1e-02))
        diff = fp16_ort_outputs[0][0][i] - outputs[0][0][i].cpu().numpy()
        max_diff = np.max(np.abs(diff))
        avg_diff = np.average(np.abs(diff))
        print(f'maximum_diff={max_diff} average_diff={avg_diff}')
        print()
        print()

if __name__ == "__main__":
    main()
