from __future__ import print_function, division

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs=25,
                conceptual_loss=None):
    since = time.time()

    # Metrics to collect for graphing
    accuracy_scores = {'train': [], 'val': []}
    loss_scores = {'train': [], 'val': []}

    # Method for saving best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_conceptual_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get the standard loss
                    if phase == 'train':
                        outputs, aux_outputs_1, aux_outputs_2 = model(inputs)
                        loss = criterion(outputs, labels) + 0.3*(criterion(aux_outputs_1, labels) + criterion(aux_outputs_2, labels))
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Add the conceptual loss if it is being used.
                    if conceptual_loss is not None:
                        aux_loss = conceptual_loss.get_conceptual_loss(inputs, labels)
                        if aux_loss is not None:
                            loss += aux_loss
                            running_conceptual_loss += aux_loss.item()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase])*100

            # Append the epoch scores to our metrics
            loss_scores[phase].append(epoch_loss)
            accuracy_scores[phase].append(epoch_acc.item())

            if conceptual_loss is not None:
                epoch_conceptual_loss = running_conceptual_loss / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} | Conceptual Loss: {epoch_conceptual_loss:.4f} '
                      f'| Acc: {epoch_acc:.2f}%')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_scores, accuracy_scores


def accuracy_by_class(model, device, dataloaders, criterion, dataset_sizes, class_names):
    phase = 'val'
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    per_class_accuracy = {'correct': [0]*len(class_names), 'total':[0]*len(class_names)}

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Add these to the total
        for label in labels.data:
            per_class_accuracy['total'][label] += 1

        # Get the standard loss
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        correct = preds == labels.data
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(correct)

        # Get accuracy by class
        for label, val in zip(labels.data, correct):
            per_class_accuracy['correct'][label] += val

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = (running_corrects.double() / dataset_sizes[phase]) * 100

    print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')

    print()
    print("Per Class Accuracy")
    for i, (cor, total) in enumerate(zip(per_class_accuracy['correct'], per_class_accuracy['total'])):
        print(f'Class name: {class_names[i]} | Acc: {(cor / total)*100:.2f}%')


def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

