# Multi-Way Machine-Generated Text Detection

This is a project designed to classify text based on its origin—whether it's human-written or generated by a specific language model. The classification task leverages an advanced model chosen after various attempts to accurately identify the source, enhancing our understanding of the nuances in AI-generated content.

Here’s a fun twist: All the texts being classified come from large language models, but the model doing the classification is a small language model! :)

## Overview

### Task Description
Given a full-text input, this project determines the source of the content. The possible classifications include:

- **Human-Written** - `0`
- **ChatGPT** - `1`
- **Cohere** - `2`
- **Davinci** - `3`
- **Dolly** - `4`
- **Bloomz** - `5`

This project is based on a problem statement from **SemEval 2024 - Task 8, Subtask B**. Although inspired by the competition, it was developed independently and was not submitted as part of the official contest. For more details on the original task, visit [this link](https://github.com/mbzuai-nlp/SemEval2024-task8).

### Model Performance

Detailed performance metrics and model specifics can be found in the [Model README](model/README.md).

### Demonstration

Check out the [Huggingface Spaces Gradio App](https://huggingface.co/spaces/Sansh2003/subtaskB-gradio-app) for a live demonstration of the model in action.
