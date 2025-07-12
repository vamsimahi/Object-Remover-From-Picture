import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline

from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=False
).to("cuda")


# Load image
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Show image for manual mask selection (could integrate UI later)
def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Load Segment Anything model
def load_sam_model(model_type="vit_b", checkpoint_path= r"C:\Users\Mahendra\Desktop\resume\sam_vit_b_01ec64.pth"
):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    return SamPredictor(sam)

# Generate mask using bounding box (e.g., user selection)
def generate_mask(image, box, predictor):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks[np.argmax(scores)]

# Apply inpainting using Stable Diffusion
def inpaint(image_pil, mask_pil, prompt="remove object"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    result = pipe(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]
    return result

# Main process
def remove_object(image_path, box):
    image = load_image(image_path)
    image_pil = Image.fromarray(image)

    # Load SAM
    predictor = load_sam_model()

    # Generate mask
    mask = generate_mask(image, box, predictor)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

    # Inpaint
    result = inpaint(image_pil, mask_pil)
    
    # Show result
    result.show()
    result.save("cleaned_output.png")

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\Mahendra\Desktop\apple.jpeg"
    box = np.array([100, 100, 300, 300])  # x1, y1, x2, y2 - manually define unwanted object area
    remove_object(image_path, box)
