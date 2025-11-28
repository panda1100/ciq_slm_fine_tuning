from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template

dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # Select more to handle longer conversations
)

dataset = standardize_sharegpt(dataset)

dataset.save_pretrained("/scratch/dataset")
#dataset.save_to_disk("/scratch/dataset")