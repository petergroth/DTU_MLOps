import sys
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from models.model import MyAwesomeModel
from models.util import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

sns.set()
np.random.seed(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--load_model_from", default="models/checkpoint.pth")
    args = parser.parse_args(sys.argv[2:])

    # Load model
    state_dict = torch.load(args.load_model_from)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    model.eval()

    # Create feature extractor
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-3])

    def fe_forward(x):
        x = x.view(-1, 1, 28, 28)  # [batch, 1, H, W]
        x = F.leaky_relu(feature_extractor[0](x))
        x = feature_extractor[2](x)
        x = F.leaky_relu(feature_extractor[1](x))
        x = feature_extractor[2](x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(feature_extractor[3](x))
        # x = F.leaky_relu(feature_extractor[4](x))
        return x.detach().squeeze()

    # Load test images
    _, testset = mnist()
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    for images, labels in testloader:
        # Extract features
        features = fe_forward(images)
        break

    features = features.numpy()

    features_embedded = TSNE(n_components=2).fit_transform(features)
    df = pd.DataFrame(
        {"x": features_embedded[:, 0], "y": features_embedded[:, 1], "label": labels}
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="x", y="y", hue="label", ax=ax, palette="Set2")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.title(
        "t-SNE visualization of 128 test images from the MNIST testset. \nFeatures are extracted from the penultimate "
        "layer. "
    )
    plt.savefig("reports/figures/02_tSNE.png", bbox_inches="tight")
