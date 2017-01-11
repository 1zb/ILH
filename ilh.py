from __future__ import print_function, nested_scopes, unicode_literals, division
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import sys
# from scipy.sparse import dok_matrix,coo_matrix,csr_matrix
from random import choice, shuffle
import maxflow
import timeit

sys.path.append('../libsvm/python')
from svmutil import *

if sys.version_info < (3, 5):
    import cPickle
else:
    import pickle as cPickle

def read_cifar(path='../cifar-10-batches-py/'):
  """ load cifar images
  """
  files = ['data_batch_1',
           'data_batch_2',
           'data_batch_3',
           'data_batch_4',
           'data_batch_5',
           'test_batch',]

  images = []
  labels = []
  for file in files:
    with open(path + file, 'rb') as fo:
        dict=cPickle.load(fo, encoding='bytes')
        images.append(dict[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1))
        labels.append(np.asarray(dict[b'labels']).reshape(-1, 1))


  images = np.vstack(images)
  labels = np.vstack(labels).reshape(-1)
  return (images, labels)

def compute_gist(images):
  """ compute gist features of images
  """

  R = 0.2989
  G = 0.5870
  B = 0.1140

  images = (images[:,:,:,0] * R + images[:,:,:,1] * G + images[:,:,:,2] * B).astype(np.uint8)

  import gist
  features = []
  for i in range(images.shape[0]):
    features.append(gist.extract(images[i]))

  features = np.vstack(features)

  with open('features', 'wb') as fo:
    cPickle.dump(features, fo)

def load_gist():
  """ load gist features
  """
  with open('features', 'rb') as fo:
    features=cPickle.load(fo)
  return features

def generate_sparse_similarity_matrix(labels, num_positive=100, num_negative=100):
  # A = dok_matrix((labels.shape[0], labels.shape[0]))
  # A = coo_matrix((labels.shape[0], labels.shape[0]))
  A = np.zeros((labels.shape[0], labels.shape[0]))

  for i in range(labels.shape[0]):
    pos_idx = np.nonzero(labels==labels[i])[0]
    # print(pos_idx.shape)
    idx = np.random.choice(pos_idx.shape[0], num_positive)
    A[i, pos_idx[idx]] = 1
    A[pos_idx[idx], i] = 1

    neg_idx = np.nonzero(labels!=labels[i])[0]
    # print(neg_idx.shape)
    idx = np.random.choice(neg_idx.shape[0], num_negative)
    A[i, neg_idx[idx]] = -1
    A[neg_idx[idx], i] = -1

    A[i,i] = 0
  # A.setdiag(0)
  return A

def generate_submodular(W):
  """Generates a sequence of (possibly overlapping) submodular submatrices of W,
     until all rows have been sampled at least once.

  Args:
      W: matrix with zero diagonal.

  Yields: (sub_W, active_indices): submatrix (sparse) and active indices.

  See Algorithm 1 from Zhuang et al. http://arxiv.org/abs/1603.02844

  """
  num_elements = W.shape[0]
  U = set(range(num_elements))
  while U:
      cur_idx = choice(list(U))
      active_indices = [cur_idx]

      # print ("cur", cur_idx, U)
      U.remove(cur_idx)
      # possible_indices = np.nonzero(W[cur_idx, :].A.ravel() < 0)[0].tolist()
      possible_indices = np.nonzero(W[cur_idx, :].ravel() < 0)[0].tolist()
      # print ("poss", possible_indices)
      shuffle(possible_indices)
      for p in possible_indices:
          # if np.all(W[p, :].A.ravel()[active_indices] <= 0):
          if np.all(W[p, :].ravel()[active_indices] <= 0):
              active_indices.append(p)
              U.discard(p)
      active_indices = sorted(active_indices)
      # print("act", active_indices)
      # yield W[active_indices, :][:, active_indices].A, active_indices
      yield W[active_indices, :][:, active_indices], active_indices

def step_one(A):
  # current_labels = 0 * (np.random.randint(2, size=A.shape[0]) * 2 - 1)
  current_labels = np.zeros(A.shape[0])
  # start = timeit.default_timer()
  for _ in range(2):

    for submatrix, active_indices in generate_submodular(A):

      costs = A.dot(current_labels)[active_indices]
      costs = costs.astype(np.float32)
      pos_costs = costs.copy()
      neg_costs = -costs
      pos_costs[pos_costs < 0] = 0
      neg_costs[neg_costs < 0] = 0

      g = maxflow.GraphFloat()
      g.add_nodes(len(active_indices))
      for i in range(len(active_indices)):
        g.add_tedge(i, neg_costs[i], pos_costs[i])

      row, col = np.nonzero(submatrix)
      vals = -2 * submatrix[row, col]
      assert np.all(vals >= 0)
      for i, j, v in zip(row, col, vals):
        g.add_edge(i, j, v, v)
      g.maxflow()
      out = []
      for i in range(len(active_indices)):
        out.append((g.get_segment(i)==1)*2-1)
      current_labels[active_indices] = out

    # for i in range(10):
    #   ind = np.nonzero(train_labels==i)[0]
    #   print('class {0}, {1} samples,   codes count (1,{2}), \t(-1,{3}).'.format(i,
    #       ind.shape[0],
    #       np.nonzero(current_labels[ind]==1)[0].shape[0],
    #       np.nonzero(current_labels[ind]==-1)[0].shape[0]))

    # end = timeit.default_timer()
    # print('{0} seconds elapsed'.format(end-start))
    return current_labels

def train(features, labels, num_samples=5000, bit_count=8):
  for i in range(bit_count):
    t_train = timeit.default_timer()
    print('[TRAIN] processing {0:3d}th bit'.format(i))
    train_idx = np.random.choice(labels.shape[0], num_samples)
    train_labels = labels[train_idx]
    train_features = features[train_idx]

    A = generate_sparse_similarity_matrix(train_labels)
    single_bit = step_one(A)
    t_svm = timeit.default_timer()
    print('[TRAIN] <GraphCut> {0:.4f} seconds elapsed'.format(t_svm-t_train))

    y, x = single_bit.tolist(), train_features.tolist()
    m = svm_train(y, x, str('-t 0 -c 4 -q'))

    svm_save_model('models/{0:05d}-{1:03d}-b{2:03d}.model'.format(num_samples, bit_count, i), m)
    t_end = timeit.default_timer()
    print('[TRAIN] <SVM>      {0:.4f} seconds elapsed'.format(t_end-t_svm))
    print('[TRAIN] {0:3d}th bit trained. {1:.4f} seconds elapsed'.format(i, t_end-t_train))
  return

def hash(features, num_samples=5000, bit_count=8):
  bits = []
  for i in range(bit_count):
    start = timeit.default_timer()
    print('[HASH] hashing {0:3d}th bit'.format(i))
    m = svm_load_model('models/{0:05d}-{1:03d}-b{2:03d}.model'.format(num_samples, bit_count, i))
    # p_label, p_acc, p_val = svm_predict([0]*features.shape[0], features.tolist(), m)
    p_label, p_acc, p_val = svm_predict([0]*features.shape[0], features.tolist(), m , str('-q'))

    bits.append(p_label)
    end = timeit.default_timer()
    print('[HASH] {0:3d}th bit hashed. {1:.4f} seconds elapsed'.format(i, end-start))

  start = timeit.default_timer()
  bits = np.vstack(bits).transpose().astype(np.int).astype(np.str)
  bits[np.nonzero(bits=='-1')] = '0'
  bits = bits.tolist()
  bits = np.array([int(''.join(bits[i]),2) for i in range(len(bits))], dtype=np.int64)

  with open('hash-{0:05d}-{1:03d}'.format(num_samples, bit_count), 'wb') as fo:
    cPickle.dump(bits, fo)
  end = timeit.default_timer()
  print('[HASH] Hash codes saved. {0:.4f} seconds elapsed'.format(end-start))
  return

def calc_mean_ap(base_set_labels, num_test, num_samples=5000, bit_count=8):
  with open('hash-{0:05d}-{1:03d}'.format(num_samples, bit_count), 'rb') as fo:
    codes = cPickle.load(fo)

  assert len(codes)==base_set_labels.shape[0]

  # calculates dist matrix between test set and base set
  t_hamming = timeit.default_timer()
  # r = (1 << np.arange(bit_count))[:,None]
  hamming_func=np.frompyfunc(lambda x: bin(x).count('1'), 1, 1)
  dist = hamming_func(np.bitwise_xor(codes[-num_test:].reshape(-1, 1), codes.reshape(1, -1))).astype(np.int)
  t_map = timeit.default_timer()
  print('[MAP] Distances got. {0:.4f} seconds elapsed'.format(t_map-t_hamming))

  min_idx = np.argsort(dist)

  # num_neighbors = 500
  # test_labels = base_set_labels[-num_test:]
  # mean_ap = 0.0
  # for i in range(num_test):
  #   retrieved = (base_set_labels[min_idx[i]]==test_labels[i])[:num_neighbors].astype(np.int)
  #   retrieved_cum = retrieved.cumsum()
  #   retrieved_cum[retrieved==0] = 0
  #   precision=np.array([retrieved_cum[i]/(i+1.0) for i in range(retrieved_cum.shape[0])])
  #   mean_ap = mean_ap + np.mean(precision[precision>0])
  # mean_ap = mean_ap / num_test
  #
  # t_end = timeit.default_timer()
  # print('[MAP] MAP got. {0:.4f} seconds elapsed'.format(t_end-t_map))

  mean_ap = 0.0
  for i in range(num_test):
    counter = 0
    ap = 0.0
    for j in range(500):
      if base_set_labels[min_idx[i,j]]==base_set_labels[i+len(codes)-num_test]:
        counter = counter + 1
        ap = ap + counter / (j + 1.0)
    if counter == 0:
      counter = 1
    ap = ap / counter
    mean_ap = mean_ap + ap
  mean_ap = mean_ap / num_test
  return mean_ap

if __name__ == '__main__':

  t_read_cifar = timeit.default_timer()
  (color_images, labels) = read_cifar()
  t_load_gist = timeit.default_timer()
  print('Cifar10 data loaded. {0:.4f} seconds elapsed'.format(t_load_gist-t_read_cifar))

  # compute_gist(color_images)

  features = load_gist()
  num_train = 58000
  num_test = 2000
  bit_count = 8
  num_samples = 500
  t_train = timeit.default_timer()
  print('GIST features loaded. {0:.4f} seconds elapsed'.format(t_train-t_load_gist))

  train(features[:num_train], labels[:num_train], num_samples=num_samples, bit_count=bit_count)
  t_hash = timeit.default_timer()
  print('Training completed. {0:.4f} seconds elapsed'.format(t_hash-t_train))

  hash(features, num_samples=num_samples, bit_count=bit_count)
  t_mean_ap = timeit.default_timer()
  print('Hashing completed. {0:.4f} seconds elapsed'.format(t_mean_ap-t_hash))

  mean_ap = calc_mean_ap(labels, num_test, num_samples=num_samples, bit_count=bit_count)
  t_end = timeit.default_timer()
  print('Mean Average Precision: {0:.4f}'.format(mean_ap))
  print('MAP calculated. {0:.4f} seconds elapsed'.format(t_end-t_mean_ap))

  print('Total {0:.4f} seconds elapsed'.format(t_end-t_read_cifar))


  # mode = ''
  # if len(sys.argv) > 1:
  #   if sys.argv[1] == 'gist':
  #     compute_gist(color_images)
  #   elif sys.argv[1] == 'train':
  #     # features = load_gist()

  # (train_data, train_labels) = sample_without_replacement(color_images[:58000],
  #                                                         labels[:58000], 5000)
