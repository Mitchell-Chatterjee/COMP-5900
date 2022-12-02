import os, glob
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# .... Captum imports..................
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset


# region Concept Data Support
def data_preprocessing(my_dir):
    """
    This method removes empty images.
    :param my_dir: This should be the path to the concepts folder.
    :return: None.
    """
    concept_folders = [x[0] for x in os.walk(my_dir)][1:]
    print(concept_folders)

    for concept in concept_folders:
        for fname in os.listdir(concept):
            if fname.endswith("color.png"):
                os.remove(os.path.join(concept, fname))


def transform(img):
    """
    Image transformation for concept classes.
    :param img: The image to transform.
    :return: Returns an image after applying a transformation to it.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


def get_tensor_from_filename(filename):
    """
    Transforms a set of images into a tensor.
    :param filename: The name of the file.
    :return: A set of images after applying a transformation.
    """
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path='data/tcav/image/imagenet/', transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)

    return tensors


def assemble_concept(name, id, concepts_path="data/tcav/image/concepts/"):
    """
    Assembles the concepts for TCAV from their respective folders.
    :param name: The concept name.
    :param id: The id of the concept.
    :param concepts_path: The path to the concepts.
    :return: A dataloader containing the concept set.
    """
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)
# endregion


# region Imagenet Data Support
def return_data_transforms():
    """
    Data augmentation and normalization for training.
    Just normalization for validation.
    :return: Data transformations.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def return_imagenet_dataset(data_dir, data_transforms):
    """
    This method returns the imagenet dataset, split into train and validation set. It also applies the data
    transformations prior to returning it.
    :param data_dir: The directory in which the data is stored.
    :param data_transforms: The set of data transformations.
    :return: A dictionary containing train and validation set datasets.
    """
    return {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'val']}


def return_dataloaders(image_datasets, batch_size=4):
    """
    This method returns a dictionary containing the dataloaders for the train and validation set.
    :param image_datasets: The dictionary containing the train and validation image sets.
    :param batch_size: The batch size for the train and validation set loaders.
    :return: A dictionary containing a train and validation set data loader.
    """
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'val']}
# endregion


def plot_metric(num_of_epochs, metric_baseline, metric_augmented, phase, metric_type):
    phase_label = 'Training'
    if phase == 'val':
        phase_label = 'Validation'

    plt.plot([epoch for epoch in range(num_of_epochs)], metric_baseline[phase], label=f'Baseline {phase_label} '
                                                                                      f'{metric_type}')
    plt.plot([epoch for epoch in range(num_of_epochs)], metric_augmented[phase], label=f'Augmented {phase_label} '
                                                                                       f'{metric_type}')

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel("Epoch Number")
    plt.ylabel(f"{phase_label} {metric_type}")
    plt.title(f"Baseline vs Augmented Model: {phase_label} {metric_type}")

    # Show the plot
    plt.show()

    # Save the figure
    plt.savefig(f'D:/University/temp/Baseline vs Augmented Model: {phase_label} {metric_type}')
