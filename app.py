import argparse
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
from transformers import CLIPModel, CLIPProcessor


class CLIPImageSearch:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_features = None
        self.image_paths = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = 8

    def load_images(self, folder_path, progress=gr.Progress()):
        # Load images from folder
        self.image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".png")]
        images = [Image.open(path).convert("RGB") for path in self.image_paths]
        batched_images = [images[i : i + self.batch_size] for i in range(0, len(images), self.batch_size)]

        progress(0, desc="Loading images")
        for batch in progress.tqdm(batched_images):
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            if self.image_features is None:
                self.image_features = features
            else:
                self.image_features = torch.cat([self.image_features, features], dim=0)
        self.image_features = self.image_features / self.image_features.norm(p=2, dim=-1, keepdim=True)
        return f"Loaded {len(self.image_paths)} images from {folder_path}."

    def search_by_text(self, text_query):
        text_inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        similarities = (self.image_features @ text_features.T).squeeze()
        sorted_indices = similarities.argsort(descending=True).cpu().numpy()
        return [self.image_paths[i] for i in sorted_indices[:5]]

    def search_by_image(self, query_image):
        image_input = self.processor(images=query_image, return_tensors="pt", padding=True)
        image_input = {k: v.to(self.device) for k, v in image_input.items()}
        with torch.no_grad():
            query_features = self.model.get_image_features(**image_input)
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
        similarities = (self.image_features @ query_features.T).squeeze()
        sorted_indices = similarities.argsort(descending=True).cpu().numpy()
        return [self.image_paths[i] for i in sorted_indices[:5]]

    def get_attention_map(self, image):
        """
        Get the attention map for a given image.

        Args:
            image (PIL.Image or Tensor): The input image.

        Returns:
            np.ndarray: The attention map.
        """
        self.model.eval()
        image_input = self.processor(images=image, return_tensors="pt", padding=True)
        image_input = {k: v.to(self.device) for k, v in image_input.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**image_input, output_attentions=True)

        # Assuming the model returns attentions in the outputs
        attentions = outputs.attentions[-1]  # Get the last layer's attention
        attention_map = attentions.mean(dim=1).squeeze().cpu().numpy()  # Average over heads

        return attention_map

    def plot_attention_map(self, image, attention_map):
        """
        Plot the attention map on the image.

        Args:
            image (PIL.Image or Tensor): The input image.
            attention_map (np.ndarray): The attention map.
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.squeeze())

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(attention_map, cmap="jet", alpha=0.5)  # Overlay attention map
        plt.colorbar()
        plt.axis("off")

        # Save the plot to a temporary file
        plt.savefig("attention_map.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        return "attention_map.png"


def save_cifar100_as_png(data_split, output_dir):
    """
    Save CIFAR-100 dataset images as PNG files.

    Args:
        data_split (str): Split of the dataset ('train' or 'test').
        output_dir (str): Directory to save the PNG images.
    """
    # Load CIFAR-100 dataset
    dataset = CIFAR100(root="./", download=True, train=data_split == "train", transform=transforms.ToTensor())
    to_pil = ToPILImage()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the dataset and save images
    for idx, (img, label) in enumerate(dataset):
        img = to_pil(img)
        os.makedirs(output_dir, exist_ok=True)
        img.save(os.path.join(output_dir, f"{label}_{idx}.png"))


# Save training and test splits
# save_cifar100_as_png("train", "./cifar100_png/train")
# save_cifar100_as_png("test", "./cifar100_png/test")

# Initialize the CLIP image search model
clip_search = CLIPImageSearch()


def set_folder(folder_path):
    return clip_search.load_images(folder_path)


def text_search_interface(text_query):
    results = clip_search.search_by_text(text_query)
    return [Image.open(path) for path in results]


def image_search_interface(uploaded_image):
    results = clip_search.search_by_image(uploaded_image)
    return [Image.open(path) for path in results]


def attention_map_interface(image):
    attention_map = clip_search.get_attention_map(image)
    return clip_search.plot_attention_map(image, attention_map)


with gr.Blocks() as app:
    # Load images to search from a specified folder
    with gr.Tab("Load Images"):
        folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the path to your image folder")
        load_button = gr.Button("Load Images")
        load_output = gr.Textbox(label="Status")
        load_button.click(
            set_folder,
            inputs=folder_input,
            outputs=load_output,
        )

    # Search images by text
    with gr.Tab("Text Search"):
        text_input = gr.Textbox(label="Search by Text", placeholder="Enter a text query")
        text_output = gr.Gallery(label="Search Results")
        text_search_botton = gr.Button("Search")
        text_search_botton.click(
            text_search_interface,
            inputs=text_input,
            outputs=text_output,
        )

    # Search images by image
    with gr.Tab("Image Search"):
        image_input = gr.Image(label="Search by Image", type="pil")
        image_output = gr.Gallery(label="Search Results")
        image_search_botton = gr.Button("Search")
        image_search_botton.click(
            image_search_interface,
            inputs=image_input,
            outputs=image_output,
        )

    # with gr.Tab("Attention Map"):
    #     image_input = gr.Image(label="Image", type="pil")
    #     attention_map_output = gr.Image(label="Attention Map")
    #     attention_map_button = gr.Button("Generate Attention Map")
    #     attention_map_button.click(
    #         attention_map_interface,
    #         inputs=image_input,
    #         outputs=attention_map_output,
    # )


if __name__ == "__main__":
    app.queue()
    app.launch()
