from collections import defaultdict
from typing import Dict, cast, Any

import numpy as np
import torch
from captum._utils.common import _get_module_from_name, _format_tensor_into_tuples
from torch import stack, ones_like, zeros_like, tensor, Tensor
import matplotlib.pyplot as plt
from captum.concept._utils.common import concepts_to_str


class ConceptualLoss:
    """
    Class used to keep track of elements associated with the conceptual loss
    """
    def __init__(self, criterion, tcav_model, experimental_sets, target_concept_name,
                 target_class_name, target_class_index, concept_key, weight_coeff):
        self.criterion = criterion
        self.tcav_model = tcav_model
        self.experimental_sets = experimental_sets
        self.target_concept_name = target_concept_name
        self.target_class_name = target_class_name
        self.target_class_index = target_class_index
        self.concept_key = concept_key
        self.weight_coeff = weight_coeff

        # TODO: Change this when generalizing to multiple experimental sets
        self.target_concept_index = [item.name for item in self.experimental_sets[0]].index(self.target_concept_name)

    def get_conceptual_loss(self, inputs, labels):
        # Here we isolate by class
        target_imgs = [img for (img, label) in zip(inputs, labels) if label == self.target_class_index]
        # If there are no images in this batch with the intended target index, return 0.
        if len(target_imgs) == 0:
            return None
        # Now let's calculate the conceptual sensitivity for members of this class in the training batch
        target_tensor = stack([img for img in target_imgs])

        tcav_scores = self.compute_cosine_similarity(target_tensor, n_steps=5)

        # This is for the target concept
        val = torch.tensor([abs(format_float(scores['l1_loss'][self.target_concept_index])) for layer, scores
               in tcav_scores[self.concept_key].items()])

        # Now let's consider reducing the values for the other `useless` concepts
        # alt_val = [abs(format_float(scores['sum_score'][2])) for layer,
        # scores in tcav_scores[self.concept_key].items()]

        #  + self.criterion(alt_val, zeros_like(alt_val))
        return val

    def compute_cosine_similarity(self, inputs, additional_forward_args=None, **kwargs):
        self.tcav_model.compute_cavs(experimental_sets=self.experimental_sets)

        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        )

        # Retrieves the lengths of the experimental sets so that we can sort
        # them by the length and compute TCAV scores in batches.
        exp_set_lens = np.array(
            list(map(lambda exp_set: len(exp_set), self.experimental_sets)), dtype=object
        )
        exp_set_lens_arg_sort = np.argsort(exp_set_lens)

        # compute offsets using sorted lengths using their indices
        exp_set_lens_sort = exp_set_lens[exp_set_lens_arg_sort]
        exp_set_offsets_bool = [False] + list(
            exp_set_lens_sort[:-1] == exp_set_lens_sort[1:]
        )
        exp_set_offsets = []
        for i, offset in enumerate(exp_set_offsets_bool):
            if not offset:
                exp_set_offsets.append(i)

        exp_set_offsets.append(len(exp_set_lens))

        # sort experimental sets using the length of the concepts in each set
        experimental_sets_sorted = np.array(self.experimental_sets, dtype=object)[
            exp_set_lens_arg_sort
        ]

        for layer in self.tcav_model.layers:
            layer_module = _get_module_from_name(self.tcav_model.model, layer)
            self.tcav_model.layer_attr_method.layer = layer_module
            attribs = self.tcav_model.layer_attr_method.attribute.__wrapped__(  # type: ignore
                self.tcav_model.layer_attr_method,  # self
                inputs,
                target=self.target_class_index,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=self.tcav_model.attribute_to_layer_input,
                **kwargs,
            )

            attribs = _format_tensor_into_tuples(attribs)
            # n_inputs x n_features
            attribs = torch.cat(
                [torch.reshape(attrib, (attrib.shape[0], -1)) for attrib in attribs],
                dim=1,
            )

            # n_experiments x n_concepts x n_features
            cavs = []
            classes = []
            for concepts in self.experimental_sets:
                concepts_key = concepts_to_str(concepts)
                cavs_stats = cast(Dict[str, Any], self.tcav_model.cavs[concepts_key][layer].stats)
                cavs.append(cavs_stats["weights"].float().detach().tolist())
                classes.append(cavs_stats["classes"])

            # sort cavs and classes using the length of the concepts in each set
            cavs_sorted = np.array(cavs, dtype=object)[exp_set_lens_arg_sort]
            classes_sorted = np.array(classes, dtype=object)[exp_set_lens_arg_sort]
            i = 0
            while i < len(exp_set_offsets) - 1:
                cav_subset = np.array(
                    cavs_sorted[exp_set_offsets[i]: exp_set_offsets[i + 1]],
                    dtype=object,
                ).tolist()
                classes_subset = classes_sorted[
                                 exp_set_offsets[i]: exp_set_offsets[i + 1]
                                 ].tolist()

                # n_experiments x n_concepts x n_features
                cav_subset = torch.tensor(cav_subset)
                cav_subset = cav_subset.to(attribs.device)
                assert len(cav_subset.shape) == 3, (
                    "cav should have 3 dimensions: n_experiments x "
                    "n_concepts x n_features."
                )

                experimental_subset_sorted = experimental_sets_sorted[
                                             exp_set_offsets[i]: exp_set_offsets[i + 1]
                                             ]
                self.cosine_similarity_subcomputation(
                    scores,
                    layer,
                    attribs,
                    cav_subset,
                    classes_subset,
                    experimental_subset_sorted,
                )
                i += 1

        return scores

    def cosine_similarity_subcomputation(self, scores, layer, attribs, cavs, classes,
                                         experimental_sets):
        # n_inputs x n_concepts
        # We normalize the attribs tensor first
        grad = torch.nn.functional.normalize(attribs.float(), dim=1)

        # Now take the dot product for the cosine similarity
        cos_sim = torch.matmul(grad, torch.transpose(cavs, 1, 2))
        assert len(cos_sim.shape) == 3, (
            "tcav_score should have 3 dimensions: n_experiments x "
            "n_inputs x n_concepts."
        )

        assert attribs.shape[0] == cos_sim.shape[1], (
            "attrib and tcav_score should have the same 1st and "
            "2nd dimensions respectively (n_inputs)."
        )
        # n_experiments x n_concepts

        # Here is where we should calculate the loss
        # First we set any negative values to zero
        cos_sim = torch.nn.functional.relu(cos_sim)
        # Here we compute L1 loss
        l1_loss = torch.mean(ones_like(cos_sim) - cos_sim, dim=1)

        temp = torch.split(cos_sim, cos_sim.size(dim=1), dim=2)
        temp_list = torch.empty(size=l1_loss.shape, device=cos_sim.device)
        for i, elem in enumerate(temp):
            # Compute l1
            tmp = elem[0]
            temp_list[0][i] = self.criterion(ones_like(elem[0]), elem[0])

        for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
            concepts_key = concepts_to_str(concepts)

            # sort classes / concepts in the order specified in concept_keys
            concept_ord = [concept.id for concept in concepts]
            class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

            new_ord = torch.tensor(
                [class_ord[cncpt] for cncpt in concept_ord], device=cos_sim.device
            )

            # sort based on classes
            scores[concepts_key][layer] = {
                "l1_loss": torch.index_select(
                    temp_list[i, :], dim=0, index=new_ord
                )
            }



def extract_highest_sensitivity_layer(tcav_model, experimental_sets, target_class_tensors, target_class_idx, layers,
                                      target_concept_index, concept_key):
    # Run TCAV
    tcav_scores = tcav_model.interpret(inputs=target_class_tensors,
                                            experimental_sets=experimental_sets,
                                            target=target_class_idx,
                                            n_steps=5)
    sens_scores = [abs(format_float(scores['sign_count'][target_concept_index])) for layer, scores in
                   tcav_scores[concept_key].items()]
    temp = [index for index, score in enumerate(sens_scores) if score == max(sens_scores)][0]
    return layers[temp]


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