import tensorflow as tf
import numpy as np
import pandas as pd
import os, random, time, string, h5py, argparse, sys, math, json, unicodedata, Levenshtein
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from keras.preprocessing import sequence
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell

# set parameters

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


options = {}

options['n_caption_step'] = 20
options['dim_word'] = 300
options['dim_hidden'] = 256
options['max_epochs'] = 13
options['batch_size'] = 32
options['clip'] = True
options['dropout'] = 0.5
options['learning_rate'] = 0.0003
options['zoneout'] = None
options['wordtoix_path'] = './words/wordtoix.npy'
options['ixtoword_path'] = './words/ixtoword.npy'
options['word_fts_path'] = './word_glove_fts_init.npy'
options['word_embedding_path'] ='./data/glove.840B.300d_dict.npy'
options['data_path_train'] = './data/train_data.h5'
options['data_path_dev_for_train'] = './data/dev_data.h5'
options['data_path_dev'] = './data/dev_data_5.h5'    # binary accuracy 
options['data_path_test'] = '../../../YouMakeup/data/task/step_ordering/test/step_ordering_test.json' #multiple choice accuracy
options['model_save_dir'] = './save_models'
options['result_save_dir'] = './save_result'
options['words_path'] = './words/'
options['optimizer'] = "adam"
options['opt_arg'] = {'adam':{'learning_rate':options['learning_rate'], 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}}
optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
                    "adam":tf.train.AdamOptimizer,
                    "gradientdescent":tf.train.GradientDescentOptimizer,
                    "adagrad":tf.train.AdagradOptimizer}
                    
if not os.path.exists(options['model_save_dir']):
    os.mkdir(options['model_save_dir'])
if not os.path.exists(options['result_save_dir']):
    os.mkdir(options['result_save_dir'])
if not os.path.exists(options['words_path']):
    os.mkdir(options['words_path'])

def parse_args():
    parser = argparse.ArgumentParser(description='Model for Sentence Ordering')
    parser.add_argument('--task', dest='task', help='train or dev or test',
                        default='train', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='gpu to run the model',
                        default='1', type=str)     
    parser.add_argument('--model', dest='model_id', help='choose model for dev or test',
                        default=1, type=int)                        
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
                    
    args = parser.parse_args()
    return args
 

    
def linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

class SRUCell(RNNCell):
    """Simple Recurrent Unit (SRU).
       This implementation is based on:
       Tao Lei and Yu Zhang,
       "Training RNNs as Fast as CNNs,"
       https://arxiv.org/abs/1709.02755
    """

    def __init__(self, num_units, activation=None, is_training = True, reuse=None):
        self._num_units = num_units
        self._activation = activation or tf.tanh
        self._is_training = is_training

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Run one step of SRU."""
        with tf.variable_scope(scope or type(self).__name__):  # "SRUCell"
            with tf.variable_scope("x_hat"):
                x = linear([inputs], self._num_units, False)
            with tf.variable_scope("gates"):
                concat = tf.sigmoid(linear([inputs], 2 * self._num_units, True))
                f, r = tf.split(concat, 2, axis = 1)
            with tf.variable_scope("candidates"):
                c = self._activation(f * state + (1 - f) * x)
            h = r * c + (1 - r) * x
        return h, c
        
def apply_dropout(options, inputs, size = None, is_training = True):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if options['dropout'] is None and options['zoneout'] is None:
        return inputs
    if options['zoneout'] is not None:
        return ZoneoutWrapper(inputs, state_zoneout_prob= options['zoneout'], is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - options['dropout'],
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs    
        
        
        
def bidirectional_GRU(options, inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = 256, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()#batch_size,20,300
            
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))
            
            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(options, cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]
                
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(options['batch_size'], shapes[1], 2*units))
            
            
def avg_sentence_pooling(options, memory, units, memory_len = None, scope = "avg_sentence_pooling"):
    with tf.variable_scope(scope):
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(memory_len, maxlen=options['n_caption_step']),tf.int32),tf.float32)
        avg_attn = tf.div(avg_attn, tf.cast(tf.tile(tf.expand_dims(memory_len,1),[1,options['n_caption_step']]),tf.float32) )
        attn = tf.expand_dims(avg_attn, -1)
        return tf.reduce_sum(attn * memory, 1) #, avg_attn

        
class BiGRU(object):
    def __init__(self, options):
        self.options = options
        
        self.Wemb = np.array(np.load(options['word_fts_path'], encoding = 'bytes', allow_pickle = True).tolist(),np.float32)
                
        self.w_hidden = tf.Variable(tf.random_uniform([4*self.options['dim_hidden'], self.options['dim_hidden']], -0.1,0.1), name='w_hidden')
        self.b_hidden = tf.Variable(tf.zeros([self.options['dim_hidden']]), name='b_hidden')
        self.w_regress = tf.Variable(tf.random_uniform([self.options['dim_hidden'],2], -0.1,0.1), name='w_regress')
        self.b_regress = tf.Variable(tf.zeros([2]), name='b_regress')
        
    def encode_sentence(self, caption, sentence_len, is_training = False, reuse = False, sentence_id = 1):
        with tf.variable_scope('sentence_fts'+str(sentence_id),reuse = reuse) as scope:
            sentence_emb = []
            with tf.device("/cpu:0"):
                for i in range(self.options['n_caption_step']):   
                    sentence_emb.append(tf.nn.embedding_lookup(self.Wemb, caption[:,i]))
                sentence_emb = tf.stack(sentence_emb)                           
                sentence_emb = tf.transpose(sentence_emb,[1,0,2])   
           
            contextual_word_encoding = bidirectional_GRU(
                                    self.options,
                                    sentence_emb,
                                    sentence_len,
                                    units = self.options['dim_hidden'],
                                    cell_fn = SRUCell,
                                    layers = 1,
                                    scope = "sentence_encoding",
                                    output = 0,
                                    is_training = is_training)  
            sentence_encoding = avg_sentence_pooling(self.options, contextual_word_encoding, units = self.options['dim_hidden'] , memory_len = sentence_len)
            
        return sentence_encoding
    def net(self,sentence_index_1_placeholder,sentence_1_len,sentence_index_2_placeholder,sentence_2_len, is_training = False):
        sentence_emb_1 = self.encode_sentence(sentence_index_1_placeholder,sentence_1_len,is_training = is_training, sentence_id = 1)#batchsize,512
        sentence_emb_2 = self.encode_sentence(sentence_index_2_placeholder,sentence_2_len,is_training = is_training, sentence_id = 2)#batchsize,512
      
        fuse_features =  tf.concat([sentence_emb_1,sentence_emb_2],-1) #batch_size,1024
        
        fuse_feature_hidden = tf.nn.relu(tf.nn.xw_plus_b(fuse_features , self.w_hidden, self.b_hidden))
        prediction = tf.nn.relu(tf.nn.xw_plus_b(fuse_feature_hidden, self.w_regress, self.b_regress))#batch_size,1
       
        
        
        return prediction
        
    def build_train(self,is_training = True):
        inputs = {}
        sentence_index_1_placeholder = tf.placeholder(tf.int32, [None,self.options['n_caption_step']])
        sentence_1_len = tf.placeholder(tf.int32, [None,])
        sentence_index_2_placeholder = tf.placeholder(tf.int32, [None,self.options['n_caption_step']])
        sentence_2_len = tf.placeholder(tf.int32, [None,])
        input_y = tf.placeholder(tf.int32, [None,])
        inputs['sentence_index_1_placeholder'] = sentence_index_1_placeholder
        inputs['sentence_1_len'] = sentence_1_len
        inputs['sentence_index_2_placeholder'] = sentence_index_2_placeholder
        inputs['sentence_2_len'] = sentence_2_len
        inputs['input_y'] = input_y

        GT_label =tf.one_hot(input_y,2)
        prediction = self.net(sentence_index_1_placeholder,sentence_1_len,sentence_index_2_placeholder,sentence_2_len,is_training = is_training)
                
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels =  GT_label, logits = prediction)
            self.loss = tf.reduce_mean(losses)
         
        predict_relation = tf.argmax(prediction, 1, name="predictions")
        return inputs, self.loss, predict_relation
        
    def build_dev(self, is_training = False):
        inputs = {}
        sentence_index_1_placeholder = tf.placeholder(tf.int32, [None,self.options['n_caption_step']])
        sentence_1_len = tf.placeholder(tf.int32, [None,])
        sentence_index_2_placeholder = tf.placeholder(tf.int32, [None,self.options['n_caption_step']])
        sentence_2_len = tf.placeholder(tf.int32, [None,])
        inputs['sentence_index_1_placeholder'] = sentence_index_1_placeholder
        inputs['sentence_1_len'] = sentence_1_len
        inputs['sentence_index_2_placeholder'] = sentence_index_2_placeholder
        inputs['sentence_2_len'] = sentence_2_len
        
        
        prediction_score = self.net(sentence_index_1_placeholder,sentence_1_len,sentence_index_2_placeholder,sentence_2_len,is_training = is_training)
        predict_relation = tf.argmax(prediction_score, 1, name="predictions")
        return inputs,predict_relation
        
def sentence_list_idx(input_sentence, wordtoix, ixtoword):

    current_captions = []
    for ind in range(len(input_sentence)):
        current_captions.append(bytes.decode(input_sentence[ind]))
    current_captions = np.array(current_captions)
    for ind in range(len(input_sentence)):
        for c in string.punctuation: 
            current_captions[ind] = current_captions[ind].replace(c,'')
        
        current_captions[ind] = current_captions[ind].strip()
        
        if current_captions[ind] == '':
            current_captions[ind] = '.'
    current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))
    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=options['n_caption_step'] -1)
    current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
    current_caption_length = np.array( list(map(lambda x: (x != 0).sum(), current_caption_matrix ))) # save the sentence length of this batch
    return current_caption_matrix, current_caption_length
                
def  meta_data_generate(train_data, dev_data):
    title = []
    train_sentence_1 = train_data['sentence_1']
    train_sentence_2 = train_data['sentence_2']
    for i in range(len(train_sentence_1)):
        title.append(bytes.decode(train_sentence_1[i]))
        title.append(bytes.decode(train_sentence_2[i]))
    
    dev_sentence_1 = dev_data['sentence_1']
    dev_sentence_2 = dev_data['sentence_2']
    for i in range(len(dev_sentence_1)):
        title.append(bytes.decode(dev_sentence_1[i]))
        title.append(bytes.decode(dev_sentence_2[i]))

    title = np.array(title)
   
    caption_data = pd.DataFrame({'Description':title})
    return caption_data

def get_word_embedding(word_embedding_path,wordtoix_path,ixtoword_path,extracted_word_fts_init_path):
    print('loading word features ...')
    word_fts_dict = np.load(word_embedding_path,encoding='bytes',allow_pickle=True).tolist()
    wordtoix = np.load(wordtoix_path, encoding='bytes', allow_pickle=True).tolist()
    ixtoword = np.load(ixtoword_path, encoding='bytes', allow_pickle=True).tolist()
    word_num = len(wordtoix)
    extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
    count = 0
    for index in range(word_num):
        if ixtoword[index] in word_fts_dict:
            extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
            count = count + 1
    np.save(extracted_word_fts_init_path,extract_word_fts)
	
def preProBuildWordVocab(sentence_iterator, word_count_threshold = 5): # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

    
def train(finetune = False,start_epoch = 0):
    train_data = h5py.File(options['data_path_train'],'r')
    dev_for_train_data = h5py.File(options['data_path_dev_for_train'],'r')
    
    if not os.path.exists(options['word_fts_path']):
        meta_data = meta_data_generate(train_data, dev_for_train_data)
        captions = meta_data['Description'].values
        for c in string.punctuation:
            captions = map(lambda x: x.replace(c, ''), captions)
        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab( captions, word_count_threshold=1)
        np.save(options['ixtoword_path'], ixtoword)
        np.save(options['wordtoix_path'], wordtoix)
        get_word_embedding(options['word_embedding_path'],options['wordtoix_path'],options['ixtoword_path'],options['word_fts_path'])
    else:
        wordtoix = (np.load(options['wordtoix_path'],allow_pickle=True)).tolist()
        ixtoword = (np.load(options['ixtoword_path'],allow_pickle=True)).tolist()

    model = BiGRU(options)
    inputs, loss, labels = model.build_train()
    
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu90% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically 
    sess = tf.InteractiveSession(config = config)

    optimizer = optimizer_factory[options['optimizer']](**options['opt_arg'][options['optimizer']])

    if options['clip']:
        gvs = optimizer.compute_gradients(loss)
        print(gvs)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
    else: 
        train_op = optimizer.minimize(loss)
        
    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=400)
    tf.initialize_all_variables().run()
    if finetune:
        with tf.device("/cpu:0"):
            saver.restore(sess, options['model_save_dir'] + '/model-'+str(start_epoch-1))

    train_input_sentence_1 = train_data['sentence_1']
    train_input_sentence_2 = train_data['sentence_2']
    train_caption_matrix_1,train_caption_length_1 = sentence_list_idx(train_input_sentence_1,wordtoix, ixtoword)
    train_caption_matrix_2,train_caption_length_2 = sentence_list_idx(train_input_sentence_2,wordtoix, ixtoword)
    
    train_input_label = train_data['label']
    
    train_label = []
    for ind in range(len(train_input_label)):
        train_label.append(int(train_input_label[ind]))
    train_label = np.array(train_label)
    
    dev_for_train_input_sentence_1 = dev_for_train_data['sentence_1']
    dev_for_train_input_sentence_2 = dev_for_train_data['sentence_2']
    dev_for_train_caption_matrix_1,dev_for_train_caption_length_1 = sentence_list_idx(dev_for_train_input_sentence_1, wordtoix, ixtoword)
    dev_for_train_caption_matrix_2,dev_for_train_caption_length_2 = sentence_list_idx(dev_for_train_input_sentence_2, wordtoix, ixtoword)
    
    dev_for_train_input_label = dev_for_train_data['label']
    
    dev_for_train_label = []
    for ind in range(len(dev_for_train_input_label)):
        dev_for_train_label.append(int(dev_for_train_input_label[ind]))
    dev_for_train_label = np.array(dev_for_train_label)

    
    print('*****************************start training**********************************')
    for epoch in range(options['max_epochs']):
        loss_list = []
        predict_labels = []
        GT_labels = []
        for index in range(math.floor(len(train_label)/options['batch_size'])):
            current_caption_matrix_1 = train_caption_matrix_1[index * options['batch_size']:(index + 1) * options['batch_size']]
            current_caption_length_1 = train_caption_length_1[index * options['batch_size']:(index + 1) * options['batch_size']]
            current_caption_matrix_2 = train_caption_matrix_2[index * options['batch_size']:(index + 1) * options['batch_size']]
            current_caption_length_2 = train_caption_length_2[index * options['batch_size']:(index + 1) * options['batch_size']]
            current_label = train_label[index * options['batch_size']:(index + 1) * options['batch_size']]                  

            _, t_loss, batch_labels =  sess.run([train_op,loss,labels],
            feed_dict = {inputs['sentence_index_1_placeholder']:current_caption_matrix_1,
            inputs['sentence_index_2_placeholder']:current_caption_matrix_2,
            inputs['sentence_1_len']:current_caption_length_1,
            inputs['sentence_2_len']:current_caption_length_2,
            inputs['input_y']:current_label})
            loss_list.append(t_loss)
            GT_labels.extend(current_label)
            predict_labels.extend(batch_labels)
            #################################################dev####################################
        
            if index % 500 == 0:
                
                print('Epoch ' + str(index) + ' is done. loss = ' + str(np.mean(loss_list)))
                accuracy = sum(np.array(predict_labels) == np.array(GT_labels) )/len(GT_labels)
                print('Epoch: ' + str(epoch) +' index:' + str(index) + ' training accuracy: ' +str(accuracy))
                
                loss_list = []
                predict_labels = [] 
                GT_labels = []
                batch_num = math.floor(len(dev_for_train_label)/options['batch_size'])
                for dev_for_train_index in range(batch_num):
                    current_caption_matrix_1 = dev_for_train_caption_matrix_1[dev_for_train_index * options['batch_size']:(dev_for_train_index + 1) * options['batch_size']]
                    current_caption_length_1 = dev_for_train_caption_length_1[dev_for_train_index * options['batch_size']:(dev_for_train_index + 1) * options['batch_size']]
                    current_caption_matrix_2 = dev_for_train_caption_matrix_2[dev_for_train_index * options['batch_size']:(dev_for_train_index + 1) * options['batch_size']]
                    current_caption_length_2 = dev_for_train_caption_length_2[dev_for_train_index * options['batch_size']:(dev_for_train_index + 1) * options['batch_size']]
                    current_label = dev_for_train_label[dev_for_train_index * options['batch_size']:(dev_for_train_index + 1) * options['batch_size']] 
                    
                    batch_labels =  sess.run([labels],
                    feed_dict = {inputs['sentence_index_1_placeholder']:current_caption_matrix_1,
                    inputs['sentence_index_2_placeholder']:current_caption_matrix_2,
                    inputs['sentence_1_len']:current_caption_length_1,
                    inputs['sentence_2_len']:current_caption_length_2,
                    })
                   
                    predict_labels.extend(list(batch_labels[0])) 
                    GT_labels.extend(current_label)
                    
                accuracy = sum(np.array(predict_labels) == np.array(GT_labels) )/len(GT_labels)
                print('Epoch: ' + str(epoch) +' index:' + str(index) + ' accuracy_dev_for_train: ' +str(accuracy) )
                loss_list = []
                predict_labels = []
                GT_labels = []
        print('Epoch {:d} is done. Saving the model ...'.format(epoch))
        with tf.device("/cpu:0"):
            saver.save(sess, os.path.join(options['model_save_dir'], 'model'), global_step=epoch + start_epoch)
            
def json_read(json_path):
    file = open(json_path, 'rb')
    items = []
    for line in file.readlines():
        dic = json.loads(line)
        items.append(dic)
    return items

def calculate_score(np_score):
    score = np.sum(np_score[0:4])*0.1 + np.sum(np_score[4:7])*0.2 + np.sum(np_score[7:9])*0.3 + np.sum(np_score[9:10])*0.4
    return score
    
def test(model_id):
    wordtoix = (np.load(options['wordtoix_path'],allow_pickle=True)).tolist()
    ixtoword = (np.load(options['ixtoword_path'],allow_pickle=True)).tolist()
    
    test_data =json_read(options['data_path_test'])
    
    model = BiGRU(options)
    inputs,predictions = model.build_dev()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, options['model_save_dir'] + '/model-'+str(model_id))
    true_num = 0
    prediction_dict = {}

    for question in test_data:
        question_id = question['question_id']
        step_caption = question['step_caption']
        ground_truth = question['groundtruth']
        candidate_answer = question['candidate_answer']
            
        steps_content = []
        steps_order = []
        steps_len =[]
            
        for key in step_caption.keys():
            steps_order.append(key)
            steps_content.append(unicodedata.normalize('NFKD', step_caption[key]).encode('ascii','ignore'))
        steps, steps_len = sentence_list_idx(steps_content, wordtoix, ixtoword)
            
        dev_caption_matrix_1 = []
        dev_caption_matrix_2 =[]
        dev_length_1 = []
        dev_length_2 = []
        for candidate in candidate_answer:
            for k in range(1,5):
                for i in range(len(candidate)-k):
                    dev_caption_matrix_1.append(steps[candidate[i]-1])
                    dev_length_1.append(steps_len[candidate[i]-1])
                    dev_caption_matrix_2.append(steps[candidate[i+k]-1])
                    dev_length_2.append(steps_len[candidate[i+k]-1])
                    
                    
        dev_caption_matrix_1 = np.array(dev_caption_matrix_1)
        dev_caption_matrix_2 = np.array(dev_caption_matrix_2)
        dev_length_1 = np.array(dev_length_1)
        dev_length_2 = np.array(dev_length_2)
            
        batch_score =  sess.run(predictions,
                    feed_dict = {inputs['sentence_index_1_placeholder']:dev_caption_matrix_1,
                    inputs['sentence_index_2_placeholder']:dev_caption_matrix_2,
                    inputs['sentence_1_len']:dev_length_1,
                    inputs['sentence_2_len']:dev_length_2
                    })
                      
        candidate_score = []
        if len(batch_score)!= 40:
            print('len of batch_score!=40')
            exit(0)
        #print('batch_score.shape:',np.shape(batch_score))
        candidate_score.append( calculate_score( batch_score[0:10]))
        candidate_score.append( calculate_score( batch_score[10:20]))
        candidate_score.append( calculate_score( batch_score[20:30]))
        candidate_score.append( calculate_score( batch_score[30:40]))
        predict_answer = candidate_answer[candidate_score.index( max( candidate_score ))]
        prediction_dict[question_id] = predict_answer 
        
        if list(ground_truth) == predict_answer:
            true_num +=1
           
    json_str = json.dumps(prediction_dict)
    with open(os.path.join(options['result_save_dir'], str(model_id) + '.json'), 'w') as json_file:
        json_file.write(json_str)
        
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('model_id ' + str(model_id) + ': ')
    print('question_num: ' + str(len(prediction_dict) * 1.0))   
    print('multiple choice accrracy: ' + str((true_num) / len(prediction_dict) * 1.0))
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return            
 

def dev(model_id):
    wordtoix = (np.load(options['wordtoix_path'],allow_pickle=True)).tolist()
    ixtoword = (np.load(options['ixtoword_path'],allow_pickle=True)).tolist()
    
    dev_data = h5py.File(options['data_path_dev'],'r')
    model = BiGRU(options)
    inputs,labels = model.build_dev()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, options['model_save_dir'] + '/model-'+str(model_id))
        
    dev_input_sentence_1 = dev_data['sentence_1']
    dev_input_sentence_2 = dev_data['sentence_2']
    dev_caption_matrix_1,dev_caption_length_1 = sentence_list_idx(dev_input_sentence_1, wordtoix, ixtoword)
    dev_caption_matrix_2,dev_caption_length_2 = sentence_list_idx(dev_input_sentence_2, wordtoix, ixtoword)
    
    dev_input_label = dev_data['label']
    
    dev_label = []
    for ind in range(len(dev_input_label)):
        dev_label.append(int(dev_input_label[ind]))
    dev_label = np.array(dev_label)
    
                                
    predict_labels = [] 
    GT_labels = []
    batch_num = math.floor(len(dev_label)/options['batch_size'])
    for dev_index in range(batch_num):
        current_caption_matrix_1 = dev_caption_matrix_1[dev_index * options['batch_size']:(dev_index + 1) * options['batch_size']]
        current_caption_length_1 = dev_caption_length_1[dev_index * options['batch_size']:(dev_index + 1) * options['batch_size']]
        current_caption_matrix_2 = dev_caption_matrix_2[dev_index * options['batch_size']:(dev_index + 1) * options['batch_size']]
        current_caption_length_2 = dev_caption_length_2[dev_index * options['batch_size']:(dev_index + 1) * options['batch_size']]
        current_label = dev_label[dev_index * options['batch_size']:(dev_index + 1) * options['batch_size']] 
        
        batch_labels =  sess.run([labels],
        feed_dict = {inputs['sentence_index_1_placeholder']:current_caption_matrix_1,
        inputs['sentence_index_2_placeholder']:current_caption_matrix_2,
        inputs['sentence_1_len']:current_caption_length_1,
        inputs['sentence_2_len']:current_caption_length_2,
        })
                   
        predict_labels.extend(list(batch_labels[0])) 
        GT_labels.extend(current_label)
          
    accuracy = sum(np.array(predict_labels) == np.array(GT_labels) )/len(GT_labels)
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('model_id ' + str(model_id) + ': ')
    print('binary accuracy: ' +str(accuracy) )
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    return

if __name__=='__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if args.task == 'train': 
        train(finetune = False,start_epoch = 0 )
    elif args.task == 'dev':
        dev(args.model_id)
    elif args.task == 'test':
        test(args.model_id)
    
