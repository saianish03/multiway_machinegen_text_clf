import gradio as gr
import pandas as pd
from utils import get_predictions
from datasets import load_dataset


data = load_dataset("Sansh2003/subtask-b-examples-test")
data = data['test']
data = pd.DataFrame(data)

def generate_example():
    """
    Reads example text and label pair from a JSON file using pandas.sample.

    Args:
        data_path (str): Path to the JSON file containing example data.

    Returns:
        tuple: A tuple containing the sampled text and its corresponding label.
    """

    try:
        # Sample a random text-label pair using pandas.sample
        example = data.sample(1)
        text = example["text"].values[0]
        label = example["model"].values[0]

        return text, label
    except FileNotFoundError:
        return "Error: JSON file not found!", None

css = """
h1{
    text-align: center;
    display: block;
};
"""

def formatted_predictions(text):
    prob_dict = get_predictions(text)
    probabilities_str = '\n'.join(f'{name}:   {value}' for name, value in prob_dict.items())
    final_str = f"Probabilities:\n{probabilities_str}"
    return final_str


with gr.Blocks(theme='snehilsanyal/scikit-learn', css=css) as demo:
    
    gr.Markdown("# Multi-Way Machine-Generated Text Classification")
    gr.Markdown("\n")
    gr.Markdown(
        """
            <p style="text-align: center; font-size: 14px;">This AI-model determines if a given text was written by a human or generated by a specific language model</p>
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Type or Paste the text here...",
            value="I don't know if this is an AI generated text or Human Generated text. Let me know what it is.",
            autofocus=True,
        )
        predictions_output = gr.Textbox(label="Predictions", show_label=True)

    with gr.Row():    
        clear_button = gr.ClearButton(components=[text_input, predictions_output], value="Clear", size="sm")
        predict_button = gr.Button(value="Predict", size="sm")
        predict_button.click(formatted_predictions, inputs=[text_input], outputs=[predictions_output])

    with gr.Row():
        example_text = gr.Textbox(label="Example Text", show_copy_button=True)
        example_label = gr.Textbox(label="Actual Label", lines=1)

    with gr.Row():
        example_button = gr.Button(value="Generate Example", size="sm")
        example_button.click(generate_example, outputs=[example_text, example_label])

    demo.launch(share=True)