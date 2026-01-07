import torch
from transformers import VideoMAEForVideoClassification
import os

# Config from the main script
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
OUTPUT_FILE = "videomae.onnx"

def export_model():
    print(f"Loading PyTorch model: {MODEL_NAME}...")
    try:
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy input based on model config
    # VideoMAE base typically expects [batch, frames, channels, height, width]
    # usually (1, 16, 3, 224, 224) for this specific model
    num_frames = 16
    image_size = 224
    
    dummy_input = torch.randn(1, num_frames, 3, image_size, image_size)
    
    print(f"Exporting to {OUTPUT_FILE}...")
    print("This may take a moment...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_FILE,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"Successfully exported model to {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    export_model()
