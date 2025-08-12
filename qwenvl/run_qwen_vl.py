import argparse
import torch
import gc
import os
import time
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import wandb

from comet import download_model, load_from_checkpoint


from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import os
os.environ["HF_TOKEN"] = "sloth"
os.environ["HUGGINGFACE_HUB_TOKEN"] = "sloth"

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune Qwen-VL model')
parser.add_argument('--model', type=str, required=True,
                   choices=['gemini', 'chatgpt', 'merge'],
                   help='Dataset type to load (gemini, chatgpt, or merged)')
parser.add_argument('--linear', action='store_true',
                   help='Use linear head training instead of full LoRA')
parser.add_argument('--train_samples', type=int, default=35000,
                   help='Number of training samples')
parser.add_argument('--eval_samples', type=int, default=2000,
                   help='Number of evaluation samples')

parser.add_argument('--mode', type=str, required=True,
                    choices=['eval', 'whatif'],
                    help='Operation mode: "eval" for standard evaluation, "whatif" for sensor-enhanced scenario')

# parser.add_argument('--sensor_df_path', type=str, default=None,
#                     help='Path to the sensor data CSV file (only used in "whatif" mode)')
parser.add_argument("--run_zero_shot", action="store_true", help="Run zero-shot evaluation")
parser.add_argument("--run_finetune", action="store_true", help="Run fine-tuning and evaluation")


args = parser.parse_args()

sensor_df_path = f"emissions_{args.model}.csv"


# print("Total GPUs:", torch.cuda.device_count())
# print("Default device index:", torch.cuda.current_device())
# print("Default device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# print("torch.cuda.device:", torch.cuda.device("cuda"))
# print("torch.cuda.is_available:", torch.cuda.is_available())

# Configuration
token = "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"  # Your Hugging Face token
linear = args.linear

TEST = True


# Force garbage collection
gc.collect()

# Load datasets
from datasets import load_from_disk
from datasets import concatenate_datasets

from unsloth import FastVisionModel

# Load metrics
import evaluate
print("Loading evaluation metrics...")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")
# comet_metric = evaluate.load("comet")

# Load COMET model (replaces loading from evaluate)
comet_model_path = download_model("Unbabel/wmt22-comet-da") 
comet_model = load_from_checkpoint(comet_model_path)

try:
    from sentence_transformers import SentenceTransformer, util
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    has_sbert = True
    print("SBERT model loaded successfully")
except (ImportError, Exception) as e:
    print(f"SBERT model couldn't be loaded: {e}")
    has_sbert = False

def load_and_resize_image(image_path, max_size=336):
    """Load and resize an image from path"""
    image_path = image_path.replace("/data/", "../../")  # modify this based on the image path
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize large images to save memory
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a small blank image as fallback
        return Image.new('RGB', (224, 224), color='gray')

def extract_assistant_response(full_text):
    if "assistant" in full_text:
        response = full_text.split("assistant")[-1].strip()
        for marker in ["user", "system", "human"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        return response
    return full_text

class VisionLanguageDataset(Dataset):
    def __init__(self, base_dataset, convert_to_conversations_fn, args, image_size=336):
        self.base_dataset = base_dataset
        self.convert_to_conversations = convert_to_conversations_fn
        self.image_size = image_size
        self.mode = args.mode
        self.columns_of_interest = columns_names
        self.sensor_df = pd.read_csv(sensor_df_path) if sensor_df_path else None

        self.index_mapping = []
        if self.mode == "eval":
            for sample_idx, sample in enumerate(base_dataset):
                num_conversations = 3  # Always 3 for eval
                for conv_idx in range(num_conversations):
                    self.index_mapping.append((sample_idx, conv_idx))

    def __len__(self):
        if self.mode == "eval":
            return len(self.index_mapping)
        else:
            return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.mode == "eval":
            sample_idx, conv_idx = self.index_mapping[idx]
            sample = self.base_dataset[sample_idx]
            conversations = self.convert_to_sample(sample)
            return conversations[conv_idx]
        else:
            sample = self.base_dataset[idx]
            return self.create_whatif_conversation(sample)

    def load_and_resize_image(self, path):
        path = path.replace("/data/", "../../")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if width > self.image_size or height > self.image_size:
            if width > height:
                new_width = self.image_size
                new_height = int(height * self.image_size / width)
            else:
                new_height = self.image_size
                new_width = int(width * self.image_size / height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def get_sensor_readings(self, image_path1):
        if self.sensor_df is None:
            return "No sensor data available"
        

        row_candidates = self.sensor_df[self.sensor_df["image_path1"] == image_path1.replace("/data/", "")]
        if row_candidates.empty:
            print("No sensor data available for image:", image_path1)
            return "No sensor data available"

        row = row_candidates.iloc[0]
        available_params = []
        for col, (name, unit) in self.columns_of_interest.items():
            val_1 = row.get(f'{col}_1')
            val_2 = row.get(f'{col}_2')
            if pd.isna(val_1) or pd.isna(val_2):
                continue
            if col == 'uvindex':
                val_1 = int(round(val_1))
                val_2 = int(round(val_2))
            elif col in ['pm10', 'carbon_monoxide', 'nitrogen_dioxide']:
                val_1 = round(val_1, 1)
                val_2 = round(val_2, 1)
            unit_str = f' {unit}' if unit else ''
            available_params.append(f"{name}: {val_1}, {val_2}{unit_str}")
        return "image1,image2 pairs: " + ", ".join(available_params) if available_params else "No sensor data available"

    def create_whatif_conversation(self, sample):
        try:
            image1 = self.load_and_resize_image(sample["image1"])
        except Exception as e:
            print(f"Error loading images for sample {sample.get('number', '')}: {e}")
            return {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Error loading image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "I cannot see the image."}]}
                ]
            }

        sensor_readings = self.get_sensor_readings(sample["image1"])
        prompt_with_data = f"Describe this image given the data: {sensor_readings}"


        conversation = {
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt_with_data},
                     {"type": "image", "image": image1}
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": sample["text_i1"]}]},
                {"role": "user",
                 "content": [{"type": "text", "text": sample["question"]}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": sample["answer"]}]}
            ]
        }
        return conversation

    def convert_to_sample(self, sample):
        try:
            image1 = self.load_and_resize_image(sample["image1"])
            image2 = self.load_and_resize_image(sample["image2"])
        except Exception as e:
            print(f"Error loading images for sample: {e}")
            return [{
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Error loading image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "I cannot see the image."}]}
                ]
            }] * 3

        sensor_readings = self.get_sensor_readings(sample["image1"])
        prompt_with_data = f"Describe the given image given the data: {sensor_readings}"
        

        conversation1 = {
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt_with_data},
                     {"type": "image", "image": image1}
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": sample["text_i1"]}]}
            ]
        }

        conversation2 = {
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt_with_data},
                     {"type": "image", "image": image2}
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": sample["text_i2"]}]}
            ]
        }

        conversation3 = {
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": f"What is the difference between the two images given the data: {sensor_readings} ?"},
                     {"type": "image", "image": image1},
                     {"type": "image", "image": image2}
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": sample["difference"]}]}
            ]
        }

        return [conversation1, conversation2, conversation3]
    

class PromptTemplateManager:
    """
    Centralized class to handle prompt templates and sensor data integration
    for both training and evaluation consistency.
    """
    
    def __init__(self, sensor_df_path=None, columns_of_interest=None, image_size=336):
        self.sensor_df = pd.read_csv(sensor_df_path) if sensor_df_path else None
        self.columns_of_interest = columns_of_interest or {}
        self.image_size = image_size

        # print(self.columns_of_interest)
    
    def load_and_resize_image(self, path):
        """Load and resize image - matches your dataloader logic"""
        path = path.replace("/data/", "../../")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if width > self.image_size or height > self.image_size:
            if width > height:
                new_width = self.image_size
                new_height = int(height * self.image_size / width)
            else:
                new_height = self.image_size
                new_width = int(width * self.image_size / height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image
    
    def get_sensor_readings_single(self, image_path, image_column):
        """
        Get sensor readings for a single image by looking in the specified column
        
        Args:
            image_path: Path to the image
            image_column: Either "image_path1" or "image_path2"
        """
        if self.sensor_df is None:
            return "No sensor data available"
        
        # Look up the image in the specified column
        cleaned_path = image_path.replace("/data/", "")
        row_candidates = self.sensor_df[self.sensor_df[image_column] == cleaned_path]
        
        if row_candidates.empty:
            print(f"No sensor data available for image: {image_path} in column {image_column}")
            return "No sensor data available"
        
        row = row_candidates.iloc[0]
        available_params = []

        
        # Determine which suffix to use based on the column
        suffix = "1" if image_column == "image_path1" else "2"
        
        for col, (name, unit) in self.columns_of_interest.items():
            val = row.get(f'{col}_{suffix}')
            if pd.isna(val):
                continue
            if col == 'uvindex':
                val = int(round(val))
            elif col in ['pm10', 'carbon_monoxide', 'nitrogen_dioxide']:
                val = round(val, 1)
            unit_str = f' {unit}' if unit else ''
            available_params.append(f"{name}: {val}{unit_str}")
        return ", ".join(available_params) if available_params else "No sensor data available"
    
    def get_sensor_readings_dual(self, image1_path, image2_path):
        """
        Get sensor readings for both images and pair the information
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
        """
        if self.sensor_df is None:
            return "No sensor data available"
        
        # Clean paths
        cleaned_path1 = image1_path.replace("/data/", "")
        cleaned_path2 = image2_path.replace("/data/", "")
        
        # Find row that contains both images
        row_candidates = self.sensor_df[
            (self.sensor_df["image_path1"] == cleaned_path1) & 
            (self.sensor_df["image_path2"] == cleaned_path2)
        ]
        
        if row_candidates.empty:
            print(f"No sensor data available for image pair: {image1_path}, {image2_path}")
            return "No sensor data available"
        
        row = row_candidates.iloc[0]
        available_params = []
        
        for col, (name, unit) in self.columns_of_interest.items():
            val_1 = row.get(f'{col}_1')
            val_2 = row.get(f'{col}_2')
            if pd.isna(val_1) or pd.isna(val_2):
                continue
            if col == 'uvindex':
                val_1 = int(round(val_1))
                val_2 = int(round(val_2))
            elif col in ['pm10', 'carbon_monoxide', 'nitrogen_dioxide']:
                val_1 = round(val_1, 1)
                val_2 = round(val_2, 1)
            unit_str = f' {unit}' if unit else ''
            available_params.append(f"{name}: {val_1}, {val_2}{unit_str}")
        
        return "image1,image2 pairs: " + ", ".join(available_params) if available_params else "No sensor data available"
    
    def get_single_image_description_prompt(self, image_path, is_image1=True):
        """
        Generate prompt for single image description with sensor data
        
        Args:
            image_path: Path to the image
            is_image1: True if this is image1, False if image2 (determines which column to look in)
        """
        image_column = "image_path1" if is_image1 else "image_path2"
        sensor_readings = self.get_sensor_readings_single(image_path, image_column)
        return f"Describe the given image given the data: {sensor_readings}"
    
    def get_dual_image_difference_prompt(self, image1_path, image2_path):
        """Generate prompt for comparing two images with paired sensor data"""
        sensor_readings = self.get_sensor_readings_dual(image1_path, image2_path)
        return f"What is the difference between the two images given the data: {sensor_readings}?"
    
    def get_whatif_conversation_prompt(self, image_path, is_image1=True):
        """
        Generate prompt for whatif conversation mode with sensor data
        
        Args:
            image_path: Path to the image
            is_image1: True if this is image1, False if image2
        """
        image_column = "image_path1" if is_image1 else "image_path2"
        sensor_readings = self.get_sensor_readings_single(image_path, image_column)
        return f"Describe this image given the data: {sensor_readings}"
    
    def create_evaluation_tasks(self, sample):
        """
        Create evaluation tasks with consistent prompts that match training.
        This replaces the hardcoded task definitions in your evaluation function.
        """
        possible_tasks = []
        
        # Check for image1 description task
        if 'text_i1' in sample and sample['text_i1'] and 'image1' in sample:
            possible_tasks.append({
                'instruction': self.get_single_image_description_prompt(sample['image1'], is_image1=True),
                'reference': sample["text_i1"],
                'image_keys': ["image1"],
                'task_type': "image1_description"
            })
        
        # Check for image2 description task
        if 'text_i2' in sample and sample['text_i2'] and 'image2' in sample:
            possible_tasks.append({
                'instruction': self.get_single_image_description_prompt(sample['image2'], is_image1=False),
                'reference': sample["text_i2"],
                'image_keys': ["image2"],
                'task_type': "image2_description"
            })
        
        # Check for difference task - uses paired sensor data
        if 'difference' in sample and sample['difference'] and 'image1' in sample and 'image2' in sample:
            possible_tasks.append({
                'instruction': self.get_dual_image_difference_prompt(sample['image1'], sample['image2']),
                'reference': sample["difference"],
                'image_keys': ["image1", "image2"],
                'task_type': "difference"
            })
        
        return possible_tasks


# Updated evaluation function using PromptTemplateManager
def evaluate_model_combined_updated(model, tokenizer, eval_dataset, model_name="zero_shot", max_new_tokens=64, 
                                  sensor_df_path=None, columns_of_interest=None):
    """
    Updated evaluation function using PromptTemplateManager for consistency with training
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        eval_dataset: The evaluation dataset
        model_name: Name for saving results (e.g., "zero_shot" or "finetuned")
        max_new_tokens: Maximum new tokens for generation
        sensor_df_path: Path to sensor data CSV
        columns_of_interest: Dictionary mapping column names to (display_name, unit) tuples
    """
    # Initialize the prompt manager with sensor data
    # global columns_of_interest

    prompt_manager = PromptTemplateManager(
        sensor_df_path=sensor_df_path,
        columns_of_interest=columns_names
    )
    
    predictions = []
    references = []
    sample_info = []
    
    print(f"Evaluating {model_name} model on {len(eval_dataset)} samples...")
    
    # Process all samples in a single loop
    for idx in tqdm(range(len(eval_dataset)), desc="Processing samples", 
                   dynamic_ncols=True, position=0, leave=True):
        if args.mode == 'eval':
            try:
                # Get the sample
                sample = eval_dataset[idx]
                
                # Use the centralized task creation with sensor data
                possible_tasks = prompt_manager.create_evaluation_tasks(sample)

                # global TEST
        
                # if TEST == True:
                #     print("Possible tasks for sample {}: {}".format(idx, possible_tasks))
                #     # print(possible_tasks)
                #     TEST = False
                        
                # Skip if no valid tasks found
                if not possible_tasks:
                    print(f"Skipping sample {idx} - no valid tasks found")
                    continue
                    
                # Process all possible tasks for this sample
                for task in possible_tasks:
                    try:
                        references.append(task['reference'])
                        
                        # Check if all required image paths exist
                        image_paths = [sample[img_key] for img_key in task['image_keys']]
                        image_paths = [path.replace("/data/", "../../") for path in image_paths]
                        if not all(os.path.exists(path) for path in image_paths):
                            missing_paths = [path for path in image_paths if not os.path.exists(path)]
                            print(f"Skipping task {task['task_type']} for sample {idx} - missing image files: {missing_paths}")
                            predictions.append("Error: missing image files")
                            sample_info.append({
                                "sample_index": idx,
                                "task_type": task['task_type'],
                                "instruction": task['instruction'],
                                "status": "error",
                                "error": "missing image files"
                            })
                            continue
                        
                        # Load images using the prompt manager (consistent with training)
                        images = [prompt_manager.load_and_resize_image(sample[img_key]) for img_key in task['image_keys']]
                        
                        # Format the prompt with the appropriate images
                        content = [{"type": "text", "text": task['instruction']}]
                        for img in images:
                            content.append({"type": "image", "image": img})
                            
                        messages = [{"role": "user", "content": content}]
                        
                        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                        
                        # Process the input - handle differently based on number of images
                        if len(images) == 1:
                            inputs = tokenizer(
                                images[0],
                                input_text,
                                add_special_tokens=False,
                                return_tensors="pt",
                            ).to("cuda")
                        else:
                            # For multiple images
                            try:
                                # Try processor if available
                                processor = getattr(tokenizer, "processor", None)
                                if processor:
                                    inputs = processor(
                                        images=images,
                                        text=input_text,
                                        return_tensors="pt",
                                    ).to("cuda")
                                else:
                                    # Fall back to direct tokenizer approach
                                    inputs = tokenizer(
                                        images,
                                        input_text,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    ).to("cuda")
                            except Exception as e:
                                print(f"Error with multiple images: {e}")
                                try:
                                    # Process one by one and combine
                                    pixel_values_list = []
                                    for img in images:
                                        result = tokenizer(
                                            img,
                                            input_text,
                                            add_special_tokens=False,
                                            return_tensors="pt",
                                        )
                                        pixel_values_list.append(result.pixel_values)
                                    
                                    # Combine the pixel values (model-specific)
                                    inputs = result
                                    inputs.pixel_values = torch.cat(pixel_values_list, dim=1)
                                except Exception as e2:
                                    print(f"All approaches failed for multiple images: {e2}")
                                    predictions.append(f"Error: {str(e2)[:100]}")
                                    sample_info.append({
                                        "sample_index": idx,
                                        "task_type": task['task_type'],
                                        "instruction": task['instruction'],
                                        "status": "error",
                                        "error": str(e2)[:100]
                                    })
                                    continue
                        
                        # Generate prediction
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                use_cache=True,
                                temperature=1.0,
                                do_sample=False
                            )
                        
                        # Decode the output
                        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        prediction = extract_assistant_response(prediction)
                        predictions.append(prediction)
                        
                        # Store sample info
                        sample_info.append({
                            "sample_index": idx,
                            "task_type": task['task_type'],
                            "instruction": task['instruction'],
                            "status": "success"
                        })
                        
                        # Clean up memory
                        del outputs, inputs
                        torch.cuda.empty_cache()
                        
                        # Print a few examples
                        if idx < 2:
                            print(f"\nExample {idx}, Task: {task['task_type']}:")
                            print(f"Instruction: {task['instruction']}")
                            print(f"Reference: {task['reference']}")
                            print(f"Prediction: {prediction}")
                        
                    except Exception as e:
                        print(f"Error processing task {task['task_type']} for sample {idx}: {e}")
                        predictions.append(f"Error: {str(e)[:100]}")
                        sample_info.append({
                            "sample_index": idx,
                            "task_type": task['task_type'],
                            "instruction": task['instruction'],
                            "status": "error",
                            "error": str(e)[:100]
                        })
                        continue
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
                
        elif args.mode == 'whatif':
            try:
                sample = eval_dataset[idx]
                # Check if required fields exist (mainly the question and answer fields)
                if not all(k in sample for k in ["image1", "text_i1", "question", "answer"]):
                    print(f"Skipping sample {idx} - missing required keys")
                    continue
                if not os.path.exists(sample["image1"].replace("/data/", "../../")):
                    print(f"Skipping sample {idx} - missing image file: {sample['image1']}")
                    continue
                    
                # Use consistent prompt with sensor data
                image = prompt_manager.load_and_resize_image(sample["image1"])
                initial_prompt = prompt_manager.get_whatif_conversation_prompt(sample["image1"])
                
                # Build conversation
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": initial_prompt},
                        {"type": "image", "image": image}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": sample["text_i1"]}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": sample["question"]}
                    ]}
                ]
                
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                
                # Tokenize
                inputs = tokenizer(
                    image, input_text, add_special_tokens=False, return_tensors="pt"
                ).to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, use_cache=True, temperature=1.0, do_sample=False
                    )
                
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = extract_assistant_response(prediction)
                predictions.append(prediction)
                references.append(sample["answer"])
                
                sample_info.append({
                    "sample_index": idx,
                    "task_type": "conversation_with_question_answer",
                    "instruction": "full multi-turn conversation",
                    "status": "success"
                })
                
                # Clean up
                del outputs, inputs
                torch.cuda.empty_cache()
                
                if idx < 2:
                    print(f"\nExample {idx}:")
                    print(f"Initial Prompt: {initial_prompt}")
                    print(f"Description: {sample['text_i1']}")
                    print(f"Question: {sample['question']}")
                    print(f"Reference Answer: {sample['answer']}")
                    print(f"Predicted Answer: {prediction}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append(f"Error: {str(e)[:100]}")
                references.append("")
                sample_info.append({
                    "sample_index": idx,
                    "task_type": "conversation_with_question_answer",
                    "instruction": "full multi-turn conversation",
                    "status": "error",
                    "error": str(e)[:100]
                })
                continue

    # Calculate metrics (comet was changed to load it directly from the checkpoint)
    bleu = bleu_metric.compute(
        predictions=predictions, 
        references=[[ref] for ref in references]
    )
    rouge = rouge_metric.compute(
        predictions=predictions, 
        references=references
    )
    meteor = meteor_metric.compute(
        predictions=predictions, 
        references=[[ref] for ref in references]
    )
    bertscore = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="en"
    )
    sources = ["N/A"] * len(predictions)  # create sources for comet

    # comet = comet_metric.compute(
    # predictions=predictions,
    # references=references,
    # sources=sources
    # )

    comet_inputs = [
        {"src": src, "mt": pred, "ref": ref}
        for src, pred, ref in zip(sources, predictions, references)
    ]

    # Improved prediction call
    comet_output = comet_model.predict(
        comet_inputs, 
        batch_size=8, 
        gpus=1 if torch.cuda.is_available() else 0
    )

    comet_score = comet_output.system_score

    
    # SBERT similarity (if available)
    sbert_scores = []
    if has_sbert:
        for pred, ref in zip(predictions, references):
            try:
                pred_embedding = sbert_model.encode(pred, convert_to_tensor=True)
                ref_embedding = sbert_model.encode(ref, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
                sbert_scores.append(similarity)
            except Exception as e:
                print(f"Error calculating SBERT score: {e}")
                sbert_scores.append(0.0)
    
    # Compile results
    results = {
        "bleu": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "meteor": meteor["meteor"],
        "bertscore_f1": np.mean(bertscore["f1"]),
        # "comet": comet["mean_score"],
        "comet": comet_score,
    }
    
    if has_sbert and sbert_scores:
        results["sbert_similarity"] = np.mean(sbert_scores)
    
    # Print results
    print(f"\n=== {model_name.capitalize()} Evaluation Results ===")
    for metric_name, score in results.items():
        print(f"{metric_name}: {score:.4f}")
    
    # Store detailed results
    detailed_results = {
        "metrics": results,
        "samples": []
    }
    
    for i in range(min(len(predictions), len(references))):
        sample_result = {
            "sample_index": sample_info[i]["sample_index"] if i < len(sample_info) else -1,
            "task_type": sample_info[i].get("task_type", "unknown") if i < len(sample_info) else "unknown",
            "instruction": sample_info[i].get("instruction", "unknown") if i < len(sample_info) else "unknown",
            "ground_truth": references[i],
            "generated": predictions[i],
            "status": sample_info[i].get("status", "unknown") if i < len(sample_info) else "unknown"
        }
        detailed_results["samples"].append(sample_result)
    
    # Save to a single JSON file
    output_filename = f"{args.model}_{model_name}_evaluation.json"
    if linear:
        output_filename = f"{args.model}_{model_name}_evaluation_linear.json"
    with open(output_filename, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Detailed evaluation results saved to '{output_filename}'")
    
    return results
    
    # return predictions, references, sample_info


def load_datasets():
    """Load training and evaluation datasets based on dataset_type"""
    if args.model == 'gemini':
        train_data = load_from_disk("gemini_train_data").select(range(args.train_samples))
        eval_data = load_from_disk("gemini_eval_data").select(range(args.eval_samples))
    elif args.model == 'chatgpt':
        train_data = load_from_disk("chatgpt_train_data").select(range(args.train_samples))
        eval_data = load_from_disk("chatgpt_eval_data").select(range(args.eval_samples))
    elif args.model == 'merge':
        gemini_train_dataset = load_from_disk("gemini_train_data")
        gemini_eval_dataset = load_from_disk("gemini_eval_data")

        chatgpt_train_dataset = load_from_disk("chatgpt_train_data")
        chatgpt_eval_dataset = load_from_disk("chatgpt_eval_data")

        # Concatenate
        train_dataset_raw = concatenate_datasets([chatgpt_train_dataset, gemini_train_dataset])
        eval_dataset_raw = concatenate_datasets([chatgpt_eval_dataset, gemini_eval_dataset])

        # Shuffle
        train_dataset_shuffled = train_dataset_raw.shuffle(seed=42)
        eval_dataset_shuffled = eval_dataset_raw.shuffle(seed=42)

        # Select subsets
        train_data = train_dataset_shuffled.select(range(args.train_samples))
        eval_data = eval_dataset_shuffled.select(range(args.eval_samples))
    else:
        raise ValueError(f"Unknown dataset type: {args.model}")
    
    return train_data, eval_data

def main():
    print(f"\n=== Configuration ===")
    # print(f"Dataset type: {args.model}")
    # print(f"Linear training: {args.linear}")
    # print(f"Train samples: {args.train_samples}")
    # print(f"Eval samples: {args.eval_samples}")

    
    # Initialize wandb
    os.environ["WANDB_API_KEY"] = "2c9ed44fd70465765d75ace5ad6666515944345e"  # Replace with your actual API key
    wandb.init(
        project="qwen-12b", 
        name=f"{args.model}-{args.train_samples}_{'linear' if args.linear else 'lora'}_training"
    )
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_data, eval_data = load_datasets()
    
    if args.run_zero_shot:

        # Load model for zero-shot evaluation first
        print("\n=== Loading Model for Zero-Shot Evaluation ===")
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen2.5-VL-7B-Instruct",
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
            token=token,
        )
        
        # First evaluate the model before fine-tuning (zero-shot)
        print("\n=== Evaluating Model in Zero-Shot Mode ===")
        # Use a smaller subset for zero-shot evaluation to save time
        zs_eval_data = eval_data.select(range(min(500, len(eval_data))))
        zs_results = evaluate_model_combined_updated(
            model, tokenizer, zs_eval_data, sensor_df_path=sensor_df_path,
            model_name="zero_shot", max_new_tokens=900
        )

        os.makedirs("results", exist_ok=True)

            # Save results to file for later comparison
        with open(f"results/zero_shot_{args.model}.json", "w") as f:
            json.dump(zs_results, f, indent=2)

        del model, tokenizer
        torch.cuda.empty_cache()
    
    if args.run_finetune:

        # Load fresh model instance for training
        print("\n=== Loading Model for Fine-tuning ===")

        # Create datasets for training
        train_dataset = VisionLanguageDataset(
            train_data,
            convert_to_conversations_fn=None,
            args=args,
            image_size=336
        )

        eval_dataset = VisionLanguageDataset(
            eval_data.select(range(min(200, len(eval_data)))),  # Smaller eval set for training
            convert_to_conversations_fn=None,
            args=args,
            image_size=336
        )

        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen2.5-VL-7B-Instruct",
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
            token=token,
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = True, 
            finetune_language_layers   = True, 
            finetune_attention_modules = True, 
            finetune_mlp_modules       = True, 
            r = 128,           
            lora_alpha = 256,  
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None, 
        )

        # if args.linear:
        #     print("Setting up linear head training...")
        #     for name, param in model.named_parameters():
        #         if "lm_head" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

        # Set up trainer


        print("Setting up trainer...")
        FastVisionModel.for_training(model) # Enable for training!

        # Record initial memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args= SFTConfig(
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 16,
            max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_strategy="steps",
            logging_steps = 100,
            eval_strategy  = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            save_steps = 100,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "constant",
            seed = 3407,
            output_dir = f"outputs_{args.model}",
            save_total_limit=5,
            report_to = "wandb",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 900,
        ),
        )

        # Start training
        print("\n=== Starting Training ===")
        trainer_stats = trainer.train()

        # Print training stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        # print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        # print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        # print(f"Peak reserved memory = {used_memory} GB.")
        # print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        # print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        # print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # Save model
        model_save_path = f"outputs/qwen_vl_finetuned_{args.model}"
        print(f"\nSaving model to '{model_save_path}'...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to '{model_save_path}'")

        # Clean up training dataset for evaluation
        del train_dataset
        torch.cuda.empty_cache()


    if args.linear:



        # Load fresh model instance for training
        print("\n=== Loading Model for Fine-tuning ===")

        # Create datasets for training
        train_dataset = VisionLanguageDataset(
            train_data,
            convert_to_conversations_fn=None,
            args=args,
            image_size=336
        )

        eval_dataset = VisionLanguageDataset(
            eval_data.select(range(min(200, len(eval_data)))),  # Smaller eval set for training
            convert_to_conversations_fn=None,
            args=args,
            image_size=336
        )

        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen2.5-VL-7B-Instruct",
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
            token=token,
        )

        print("Setting up linear head training...")
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


                # Record initial memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args= SFTConfig(
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 16,
            max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_strategy="steps",
            logging_steps = 100,
            eval_strategy  = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            save_steps = 100,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "constant",
            seed = 3407,
            output_dir = f"outputs_{args.model}",
            save_total_limit=5,
            report_to = "wandb",
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 900,
        ),
        )

        # Start training
        print("\n=== Starting Training ===")
        trainer_stats = trainer.train()


        # Save model
        model_save_path = f"outputs/qwen_vl_finetuned_{args.model}{'_linear'}"
        print(f"\nSaving model to '{model_save_path}'...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to '{model_save_path}'")

        # Clean up training dataset for evaluation
        del train_dataset
        torch.cuda.empty_cache()


    if args.linear or args.run_finetune:
        # Evaluate fine-tuned model
        print("\n=== Evaluating Fine-tuned Model ===")
        FastVisionModel.for_inference(model)
        model.eval()
        ft_results = evaluate_model_combined_updated(
            model, tokenizer, eval_data, sensor_df_path=sensor_df_path,
            model_name="finetuned_with_sensor_data", max_new_tokens=900
        )

        # Save fine-tuned results for later use
        os.makedirs("results", exist_ok=True)
        finetuned_result_path = f"results/finetuned_{args.model}{'_linear' if args.linear else ''}.json"

        with open(finetuned_result_path, "w") as f:
            json.dump(ft_results, f, indent=2)
        print(f"Saved fine-tuned results to {finetuned_result_path}")

        # Attempt comparison with zero-shot if available
        zero_shot_path = f"results/zero_shot_{args.model}.json"
        if os.path.exists(zero_shot_path):
            with open(zero_shot_path) as f:
                zs_results = json.load(f)

            print("\n=== Metrics Comparison (Zero-shot vs Fine-tuned) ===")
            comparison_results = {}
            for metric in ft_results:
                if metric in zs_results:
                    zs_score = zs_results[metric]
                    ft_score = ft_results[metric]
                    diff = ft_score - zs_score
                    improvement = (diff / zs_score) * 100 if zs_score > 0 else 0
                    print(f"{metric}: Zero-shot = {zs_score:.4f}, Fine-tuned = {ft_score:.4f}, Diff = {diff:+.4f} ({improvement:+.2f}%)")
                    comparison_results[metric] = {
                        "zero_shot": zs_score,
                        "fine_tuned": ft_score,
                        "difference": diff,
                        "improvement_percent": improvement
                    }

            # Save comparison
            comparison_filename = f"results/{args.model}_comparison{'_linear' if args.linear else ''}.json"
            comparison_data = {
                "dataset_type": args.model,
                "linear_training": args.linear,
                "training_time_minutes": round(trainer_stats.metrics['train_runtime']/60, 2),
                "zero_shot_results": zs_results,
                "fine_tuned_results": ft_results,
                "comparison": comparison_results
            }
            with open(comparison_filename, "w") as f:
                json.dump(comparison_data, f, indent=2)
            print(f"Comparison results saved to '{comparison_filename}'")

            # Log final results to wandb
            wandb.log({
                "final_bleu_zs": zs_results.get("bleu", 0),
                "final_bleu_ft": ft_results.get("bleu", 0),
                "final_rouge1_zs": zs_results.get("rouge1", 0),
                "final_rouge1_ft": ft_results.get("rouge1", 0),
                "final_meteor_zs": zs_results.get("meteor", 0),
                "final_meteor_ft": ft_results.get("meteor", 0),
                "final_bertscore_zs": zs_results.get("bertscore_f1", 0),
                "final_bertscore_ft": ft_results.get("bertscore_f1", 0),
                "training_time_minutes": round(trainer_stats.metrics['train_runtime']/60, 2)
            })
        else:
            print("\n[Note] Zero-shot results not found. Skipping comparison.")
            wandb.log({
                # "final_bleu_zs": zs_results.get("bleu", 0),
                "final_bleu_ft": ft_results.get("bleu", 0),
                # "final_rouge1_zs": zs_results.get("rouge1", 0),
                "final_rouge1_ft": ft_results.get("rouge1", 0),
                # "final_meteor_zs": zs_results.get("meteor", 0),
                "final_meteor_ft": ft_results.get("meteor", 0),
                # "final_bertscore_zs": zs_results.get("bertscore_f1", 0),
                "final_bertscore_ft": ft_results.get("bertscore_f1", 0),
                "training_time_minutes": round(trainer_stats.metrics['train_runtime']/60, 2)
            })
        

        wandb.finish()
        print("\n=== Training and Evaluation Complete ===")

if __name__ == "__main__":
    columns_names = {
        'temp': ('Temperature', 'C'),
        'dew': ('Dew Point', 'C'),
        'humidity': ('Relative Humidity', '%'),
        'windspeed': ('Wind Speed', 'kph'),
        'winddir': ('Wind Direction', 'degrees'),
        'uvindex': ('UV Index', None),  # No unit
        'pm10': ('PM10', 'µg/m3'),
        'carbon_monoxide': ('Carbon Monoxide', 'ppm'),
        'nitrogen_dioxide': ('Nitrogen Dioxide', 'ppm'),
        'european_aqi': ('European AQI', None),  # No unit
    }

    main()