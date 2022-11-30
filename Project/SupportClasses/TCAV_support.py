import numpy as np
from torch import stack
import matplotlib.pyplot as plt
from captum.concept._utils.common import concepts_to_str


class ConceptualLoss:
    """
    Class used to keep track of elements associated with the conceptual loss
    """
    def __init__(self, criterion, tcav_model, experimental_set, target_concept_name, target_class_name,
                 target_class_index, concept_key):
        self.criterion = criterion
        self.tcav_model = tcav_model
        self.experimental_set = experimental_set
        self.target_concept_name = target_concept_name
        self.target_class_name = target_class_name
        self.target_class_index = target_class_index
        self.concept_key = concept_key

    def get_tcav_scores(self, inputs, labels):
        # Here we isolate by class
        target_imgs = [img for (img, label) in zip(inputs, labels) if label == self.target_class_index]

        # If there are no images in this batch with the intended target index, return 0.
        if len(target_imgs) == 0:
            return 0

        # Now let's calculate the conceptual sensitivity for members of this class in the training batch
        target_tensor = stack([img for img in target_imgs])
        tcav_scores = self.tcav_model.interpret(inputs=target_tensor,
                                                experimental_sets=self.experimental_set,
                                                target=self.target_class_index,
                                                n_steps=5,
                                                )
        return tcav_scores

    def get_conceptual_loss(self, inputs, labels):
        tcav_scores = self.get_tcav_scores(inputs, labels)

        # TODO: This needs to be changed to support dynamic arguments
        val = [format_float(scores['sign_count'][0]) for layer, scores in tcav_scores[self.concept_key].items()]
        # TODO: Generalize this for moving away from other concepts
        # This will take the sum over the list and subtract it from the intended perfect score.
        loss = len(val) - sum(val)
        return loss


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))


def get_concept_keys(experimental_set):
    """
    Returns the concept key associated with the experimental set.
    :param experimental_set: The experimental set used for TCAV scores.
    :return: The concept key.
    """
    return concepts_to_str(experimental_set)


# TODO: Update this as we now only have one experimental set.
def plot_tcav_scores(experimental_sets, tcav_scores, layers):
    """
    Plots TCAV scores in a graph.
    :param experimental_sets: The set of concepts we wish to plot.
    :param tcav_scores: The TCAV scores.
    :param layers: The layers to check for their score.
    :return: None.
    """
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concept_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concept_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.show()