import os
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor

model_name = "google/siglip2-base-patch16-224"
save_directory = "/home/agilex/limo_ros2_ws/src/patrol_bot/models/siglip2-base-patch16-224"

print(f"Downloading {model_name} to {save_directory}...")

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

try:
    # print("Downloading Processor...")
    # processor = AutoProcessor.from_pretrained(model_name)
    # processor.save_pretrained(save_directory)
    
    print("Downloading Image Processor...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image_processor.save_pretrained(save_directory)

    print("Downloading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    print("Downloading Model...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    print("SUCCESS: Model saved to", save_directory)
except Exception as e:
    print("FAILED:", e)
