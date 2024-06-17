# Self-Supervised-Network-with-contrastive-lossess
* This code implements a custom supervised contrastive learning approach and compares it with a traditional fully supervised learning method using the Imagenette dataset.

## Summary of the Project
### Objective: 
* To train a model using self-supervised learning with contrastive loss to achieve robust and discriminative feature representations.
### Approach:
* Train an encoder network using contrastive loss on unlabeled data to learn meaningful embeddings.
* Visualize the learned embeddings to verify the quality and separability of the features.
Compare the performance and stability of the self-supervised model with a fully supervised model trained on the same dataset.
### Outcomes
* The self-supervised contrastive learning model achieved a stable accuracy curve and well-separated clusters in the embedding space.
* The fully supervised model showed a less stable accuracy curve and tightly knit feature clusters, indicating potential difficulties in distinguishing between different classes.
## Some results and comparisons
### Accuracy movement
![result1](https://raw.githubusercontent.com/swajayresources/imageresources/main/Screenshot%202024-06-12%20015410.png)
* Above image shows a steady increase in the model's accuracy using the most basic model using a custom contrastive loss.


![result2](https://raw.githubusercontent.com/swajayresources/imageresources/main/Screenshot%202024-06-12%20015446.png)
* Above image shows an unstable movement in the accuracy using a fully supervised Resnet50 model.
### Loss Comparison
* I have mentioned another custom implmentation similiar to mine in the [Sources](#Sources) section.
  
![result3](https://raw.githubusercontent.com/swajayresources/imageresources/main/contrative%20lo.png)
* Above image shows the loss curve of the other custom implementation.
  
![result4](https://raw.githubusercontent.com/swajayresources/imageresources/main/0.04.png)
* Above image shows the loss curve of my implementation.




### The other custom implementation used these losses:-
- `tfa.losses.contrastive_loss` : TensorFlow Addons Module.
  
- `tfa.losses.npairs_loss` : TensorFlow Addons Module.
  
- `tfa.losses.triplet_hard_loss` : TensorFlow Addons Module.
  
- `tfa.losses.triplet_semihard_loss` : TensorFlow Addons Module.
  
- `supervised_nt_xent_loss` :  Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py.





### Loss Functions used by me.
- `contrastive_loss` : Source: Hadsell et al. (2006). "Dimensionality Reduction by Learning an Invariant Mapping".
Rationale: The loss is computed as the average of positive pair distances and the margin-based hinge loss for negative pairs.

- `npairs_loss` : Source: Sohn (2016). "Improved Deep Metric Learning with Multi-class N-pair Loss Objective".
Rationale: This implementation calculates the N-pair loss using exponential scaling and masking for the similarity matrix.
- `batch_hard_triplet_loss` : Same as below.
- `batch_all_triplet_loss` :  Source: Schroff et al. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering".
Rationale: This implementation covers hard, soft, and semi-hard triplet loss variants using pairwise distances and appropriate masking.
- `nt_xent_loss` :  Source: Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR) and Khosla et al. (2020). "Supervised Contrastive Learning".
Rationale: This implementation scales logits with temperature and uses a mask for positive pairs, following the SimCLR and SupContrast approaches.



* Loss Functions Used by me were custom functions with the mentioned sources.
* This was also due to the fact that Tensorflow Addons Module is now deprecated as of May 2024.
* So to add to "CUSTOM" part of this implementation ; custom contrastive loss functions.


## Potential Errors and Fixes 

* If you're running the code on Colab (like i did) you might encounter this error.
  
![result5](https://raw.githubusercontent.com/swajayresources/imageresources/main/Screenshot%202024-06-12%20000753.png)

* Try Running this code before the cell.
  
 ```bash
!pip install --upgrade scikit-image
```

  
## Sources 
* This code is a custom implementation of the [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) paper.
* Another good implementation can be found [here](https://github.com/vk1996/contrastive_learning).

> **Note :**
> - The code mentioned above is used for comparison purposes only.
> - It will not run as it contains the use of deprecated packages(TensorflowAddons).
