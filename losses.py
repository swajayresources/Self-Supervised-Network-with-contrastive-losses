import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa

import tensorflow as tf
import numpy as np







def contrastive_loss(labels, distances, margin=1.0):
    labels = tf.cast(labels, tf.float32)
    pos_part = labels * tf.square(distances)
    neg_part = (1 - labels) * tf.square(tf.maximum(margin - distances, 0))
    loss = tf.reduce_mean(pos_part + neg_part) / 2
    return loss

def triplet_loss(labels, embeddings, margin=1.0, kind='hard'):
    def pairwise_distance(embeddings):
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
        square_norm = tf.linalg.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        distances = tf.maximum(distances, 0.0)
        mask = tf.equal(distances, 0.0)
        distances = distances + tf.cast(mask, tf.float32) * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - tf.cast(mask, tf.float32))
        return distances

    def get_triplet_mask(labels):
        indices_equal = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)
        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
        mask = tf.logical_and(distinct_indices, valid_labels)
        return mask

    def batch_all_triplet_loss(labels, embeddings, margin):
        pairwise_dist = pairwise_distance(embeddings)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        mask = get_triplet_mask(labels)
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        triplet_loss = tf.multiply(mask, triplet_loss)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        return triplet_loss

    def batch_hard_triplet_loss(labels, embeddings, margin, soft=False):
        pairwise_dist = pairwise_distance(embeddings)
        mask_anchor_positive = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        anchor_positive_dist = tf.multiply(pairwise_dist, tf.cast(mask_anchor_positive, tf.float32))
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        mask_anchor_negative = tf.logical_not(mask_anchor_positive)
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * tf.cast(mask_anchor_negative, tf.float32)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        if soft:
            triplet_loss = tf.log1p(tf.exp(hardest_positive_dist - hardest_negative_dist))
        else:
            triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)
        return triplet_loss

    if kind == 'hard':
        return batch_hard_triplet_loss(labels, embeddings, margin, soft=False)
    elif kind == 'soft':
        return batch_hard_triplet_loss(labels, embeddings, margin, soft=True)
    elif kind == 'semihard':
        return batch_all_triplet_loss(labels, embeddings, margin)

def npairs_loss(y, S):
    y = tf.cast(y, tf.int32)
    y_one_hot = tf.one_hot(y, tf.shape(S)[0])
    lshape = tf.shape(y_one_hot)
    label_sim = tf.cast(tf.matmul(y_one_hot, tf.transpose(y_one_hot)), tf.float32)
    mask = tf.eye(lshape[0])
    label_sim = label_sim - mask
    S = tf.exp(S)
    S = S * (1 - mask)
    loss = -tf.reduce_sum(label_sim * S, axis=1) / tf.reduce_sum(S, axis=1)
    return tf.reduce_mean(loss)

def nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    batch_size = tf.shape(z)[0]
    y = tf.expand_dims(y, -1)
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(tf.matmul(z, tf.transpose(z)), temperature)
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return tf.reduce_mean(loss)


def pdist_euclidean(A):
    # Euclidean pdist
    # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return tf.sqrt(D)


def square_to_vec(D):
    '''Convert a squared form pdist matrix to vector form.
    '''
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_vec = tf.gather_nd(D, list(zip(triu_idx[0], triu_idx[1])))
    return d_vec


def get_contrast_batch_labels(y):
    '''
    Make contrast labels by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    D_y = pdist_euclidean(y_col_vec)
    d_y = square_to_vec(D_y)
    y_contrasts = tf.cast(d_y == 0, tf.int32)
    return y_contrasts


def get_contrast_batch_labels_regression(y):
    '''
    Make contrast labels for regression by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    raise NotImplementedError


def max_margin_contrastive_loss(z, y, margin=1.0, metric='euclidean'):
    '''
    Wrapper for the maximum margin contrastive loss (Hadsell et al. 2006)
    `tfa.losses.contrastive_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
        metric: one of ('euclidean', 'cosine')
    '''
    # compute pair-wise distance matrix
    if metric == 'euclidean':
        D = pdist_euclidean(z)
    elif metric == 'cosine':
        D = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
    # convert squareform matrix to vector form
    d_vec = square_to_vec(D)
    # make contrastive labels
    y_contrasts = get_contrast_batch_labels(y)
    loss = contrastive_loss(y_contrasts, d_vec, margin=margin)
    # exploding/varnishing gradients on large batch?
    return tf.reduce_mean(loss)


def multiclass_npairs_loss(z, y):
    '''
    Wrapper for the multiclass N-pair loss (Sohn 2016)
    `tfa.losses.npairs_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    # cosine similarity matrix
    S = tf.matmul(z, z, transpose_a=False, transpose_b=True)
    loss = npairs_loss(y, S)
    return loss


# def triplet_loss(z, y, margin=1.0, kind='hard'):
#     '''
#     Wrapper for the triplet losses 
#     `tfa.losses.triplet_hard_loss` and `tfa.losses.triplet_semihard_loss`
#     Args:
#         z: hidden vector of shape [bsz, n_features], assumes it is l2-normalized.
#         y: ground truth of shape [bsz].    
#     '''
#     if kind == 'hard':
#         loss = tfa.losses.triplet_hard_loss(y, z, margin=margin, soft=False)
#     elif kind == 'soft':
#         loss = tfa.losses.triplet_hard_loss(y, z, margin=margin, soft=True)
#     elif kind == 'semihard':
#         loss = tfa.losses.triplet_semihard_loss(y, z, margin=margin)
#     return loss


def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    # # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss