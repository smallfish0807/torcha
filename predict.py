import argparse
from importlib import import_module

import yaml
import torch
from torchvision import transforms
from PIL import Image

from src.mvc import LabelDecoding
from src.utils import load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--config",
                        type=str,
                        default="config/mvc.yaml",
                        help="Path to config file (default: config/mvc.yaml)")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="Which gpu to use (default: 0)")
    parser.add_argument(
        "-l",
        "--load-model-path",
        type=str,
        default="model/mvc/model.tar",
        help="Path to load model (default: model/mvc/model.tar)")
    args = parser.parse_args()
    return args


def predict(model, inputs, decoder):
    outputs = model(inputs)
    predicts = outputs.argmax(dim=1)
    answers = [decoder(predict) for predict in predicts]
    return answers


def main():
    args = get_args()

    # Load config
    with open(args.config) as fd:
        spec = yaml.load(fd, Loader=yaml.FullLoader)

    # Set device, random seed
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available(
    ) else torch.device("cpu")
    torch.manual_seed(spec['seed'])

    # Prepare model
    model_module = import_module("src.model")
    ModelClass = getattr(model_module, spec['model'])
    model = ModelClass(**spec['model_kwargs']).to(device)
    model, _, _ = load_model(args.load_model_path, device, model)

    # Prepare decoder
    decoder = LabelDecoding(spec['chars'])

    # Prepare image
    image = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).to(device).unsqueeze(0)

    # Predict
    model.eval()
    answer, = predict(model, image, decoder)
    print(answer)


if __name__ == "__main__":
    main()
