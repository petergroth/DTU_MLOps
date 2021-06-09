from torchvision import datasets, transforms


def mnist():
    # Convert to tensors and normalise

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download and load the train data
    trainset = datasets.MNIST('data/processed/MNIST_data/', download=True, train=True, transform=transform)

    # Download and load the test data
    testset = datasets.MNIST('data/processed/MNIST_data/', download=True, train=False, transform=transform)

    return trainset, testset
