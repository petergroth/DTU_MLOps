import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchdrift
import torchvision.utils

from src.util import Classifier, MNISTData


def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--model_path", default="models/mnist-epoch=01-val_acc=0.99.ckpt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Load model
    classifier = Classifier.load_from_checkpoint(args.model_path)
    classifier.eval()

    # Prepare data

    datamodule = MNISTData(batch_size=16)
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    # Prepare data
    data = next(iter(dataloader))
    inputs, labels = data
    # Out of distribution
    data = next(iter(dataloader))
    inputs_ood, labels_ood = data
    inputs_ood = corruption_function(inputs_ood)

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(datamodule.train_dataloader(), classifier, drift_detector, num_batches=1000)
    drift_detection_model = torch.nn.Sequential(
        classifier,
        drift_detector
    )
    #input_cat = torch.cat([inputs, inputs_ood])
    #predictions = classifier(input_cat)
    #predictions = F.softmax(predictions, dim=-1)
    #predictions = predictions.max(1)[1]

    if False:
        images = torchvision.utils.make_grid(input_cat)
        fig, ax = plt.subplots(figsize=(8, 4))
        grid_img = torchvision.utils.make_grid(images, nrow=2)
        ax.imshow(grid_img.permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        print(predictions[:8])
        print(predictions[8:])

    benign_pred = classifier(inputs)
    score = drift_detector(benign_pred)
    p_val = drift_detector.compute_p_value(benign_pred)
    print(score, p_val)

    ood_pred = classifier(inputs_ood)
    score = drift_detector(ood_pred)
    p_val = drift_detector.compute_p_value(ood_pred)
    print(score, p_val)