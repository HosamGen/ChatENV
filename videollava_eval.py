import os
import av
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from datasets import Dataset
import math
import time
from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration

import argparse
# ================================================================================================
MAX_LENGTH = 900
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"

# # Configuration
DEVICE = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0])
print(DEVICE)


parser = argparse.ArgumentParser()
parser.add_argument("--use_base", action="store_true", default=False)
parser.add_argument("--model_path", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()

if not args.use_base:
    MODEL_TAG = args.model_path.split("/")[-1]

test_annotations = f'./no_sensor_annotations/{args.model}_val_annotations.json'
test_directory = "chatenv_val_videos"


# ================================================================================================

def read_video_pyav(video_path, start, end):
    """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
    container = av.open(video_path)
    video = container.streams.get(0)[0]
    
    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]
    
    av_timestamps.sort()
    start_id = bisect.bisect_left(av_timestamps, start)
    end_id = bisect.bisect_left(av_timestamps, end)

    if end_id - start_id < 10:
        end_id = min(len(av_timestamps) - 1, end_id + 10)
        start_id = max(0, start_id - 10)

    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(0, start_id)

    num_frames_to_sample = min(2, end_id - start_id + 1)
    indices = np.linspace(start_id, end_id, num_frames_to_sample).astype(int)

    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_id:
            break
        if i >= start_id and i in indices:
            frames.append(frame)

    assert len(frames) == 2, f"Got {len(frames)} frames but should be 2. Check the indices: {indices}; start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


from torch.utils.data import Dataset

class VideoLlavaDataset(Dataset):
    """PyTorch Dataset for VideoLlavaDataset."""
    
    def __init__(self, dataset, video_path):
        super().__init__()
        self.dataset = dataset
        self.video_path = video_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        clip = read_video_pyav(f'{self.video_path}/{sample["video"]}', sample.get("start", 0), sample.get("end", 1e+10))
        answer = sample['conversations'][1]['value']
        tmp_prompt = sample['conversations'][0]['value']

        prompt = f"USER: {tmp_prompt}\n ASSISTANT: Answer: {answer}"
        return prompt, clip, answer


# Load test annotations
with open(test_annotations, 'r') as file:
    test_data = json.load(file)

# Create dictionary for testing data
test_dataset_dict = {
    "video": [item['video'] for item in test_data],
    "conversations": [item['conversations'] for item in test_data],
}

from datasets import Dataset

# Convert dictionaries to HuggingFace datasets
test_dataset_tmp = Dataset.from_dict(test_dataset_dict)
test_dataset = VideoLlavaDataset(test_dataset_tmp, test_directory)

print(args.model_path)

start_time = time.time()

# Code for the base model only
if args.use_base:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )

    results = []

    # Loop through the training data
    for test in test_data:
        true_value = test['conversations'][1]['value']
        
        # Generate the predicted response
        inputs = processor(
            text=test['conversations'][0]['value'],
            videos=read_video_pyav(f'{test_directory}/{test["video"]}', 0, 1e+10),
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        generate_kwargs = {"max_new_tokens": 256, "do_sample": True, "top_p": 0.9}
        output = model.generate(**inputs, **generate_kwargs)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)

        # Create result entry
        result_entry = {
            'id': test['id'],
            'video': test['video'],
            'true': true_value,
            'generated': generated_text[0]  # Add the cleaned text from the batch
        }

        # Save each result entry incrementally to the JSON file
        with open('results_base_videollava.json', 'a') as f:
            f.write(json.dumps(result_entry) + '\n')

        print("--- %s seconds ---" % (time.time() - start_time))
    
else:
    print("Using local model")

    # Load processor and model from local directory
    processor = AutoProcessor.from_pretrained(args.model_path)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token  # if pad token is not set
    processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the model from local directory
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    results = []
    batch_size = 25  # Define your batch size here

    model.eval()
    # Check if file already exists
    file_path = f"results_{MODEL_TAG}.json"
    if not os.path.exists(file_path):
        # Create a new file with an opening array bracket if it doesn't exist
        with open(file_path, 'w') as f:
            f.write('[')
        file_has_content = False
    else:
        # Check if file has content besides the opening bracket
        file_size = os.path.getsize(file_path)
        if file_size <= 1:  # Only has '[' or is empty
            file_has_content = False
        else:
            # Check if file ends with ']' and remove it for appending
            with open(file_path, 'rb+') as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell() - 1
                
                f.seek(pos)
                last_char = f.read(1).decode()
                
                if last_char == ']':
                    # Remove the closing bracket
                    f.seek(pos)
                    f.truncate()
            file_has_content = True

    with torch.no_grad():
        # Split test data into batches
        for i in range(math.ceil(len(test_data) / batch_size)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            
             # Collect texts and videos for the current batch
            batch_texts = []
            batch_videos = []
            
            for test in test_data[start_idx:end_idx]:
                # Format the prompt exactly as in training
                prompt = f"USER: {test['conversations'][0]['value']}\n ASSISTANT:"
                batch_texts.append(prompt)
                
                try:
                    video = read_video_pyav(f'{test_directory}/{test["video"]}', 0, 1e+10)
                    batch_videos.append(video)
                except Exception as e:
                    print(f"Error reading video {test['video']}: {e}")
                    # Use a placeholder empty video if needed
                    batch_videos.append(np.zeros((2, 224, 224, 3), dtype=np.uint8))
            
            # Process the entire batch at once
            try:
                inputs = processor(text=batch_texts, videos=batch_videos, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_LENGTH,
                    do_sample=False,
                    temperature=0.1,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                if i == 0:  # Only for the first batch
                    print("PAD:", processor.tokenizer.pad_token_id)
                    print("EOS:", processor.tokenizer.eos_token_id)
                
                # Decode the generated text
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Extract just the model's answer part
                processed_responses = []
                for text in generated_texts:
                    # First, try to find the assistant's response
                    if "ASSISTANT:" in text:
                        answer_part = text.split("ASSISTANT:")[-1].strip()
                        if "Answer:" in answer_part:
                            answer_part = answer_part.split("Answer:")[-1].strip()
                        processed_responses.append(answer_part)
                    else:
                        processed_responses.append(text)
                
                # Debug: Print a sample of inputs and outputs
                if i == 0:  # Only for the first batch
                    for idx in range(min(3, len(batch_texts))):
                        print(f"\nInput {idx}: {batch_texts[idx][:100]}...")
                        print(f"Raw output {idx}: {generated_texts[idx][:100]}...")
                        print(f"Processed output {idx}: {processed_responses[idx][:100]}...")
                
            except Exception as e:
                print(f"Error in generation for batch {i+1}: {e}")
                processed_responses = ["Error in generation"] * (end_idx - start_idx)
            
            # Append new results directly to the file
            with open(file_path, 'a') as f:
                for idx, test in enumerate(test_data[start_idx:end_idx]):
                    true_value = test['conversations'][1]['value']
                    result_entry = {
                        'id': test.get('id', f"test_{start_idx+idx}"),
                        'video': test['video'],
                        'true': true_value,
                        'generated': processed_responses[idx] if idx < len(processed_responses) else "Error"
                    }
                    
                    # Add a comma before if there's already content in the file
                    if file_has_content:
                        f.write(',\n')
                    else:
                        # First entry doesn't need a comma, but subsequent ones will
                        file_has_content = True
                        f.write('\n')
                    
                    # Write the result entry
                    json.dump(result_entry, f)
            
            # Close the JSON array if it's the last batch
            if i == math.ceil(len(test_data) / batch_size) - 1:
                with open(file_path, 'a') as f:
                    f.write('\n]')  # Close the JSON array

    print("Evaluation completed!")
    print("--- %s seconds ---" % (time.time() - start_time))
