import os
from collections import Counter, OrderedDict
import tensorflow as tf
from itertools import chain
import numpy as np

class Dataset(object):
    def __init__(self,special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None ):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip() # 문자열 앞 뒤 공백제거
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False #구분자에 따라 문장을 분할 line = "This is an example sentence."
        #symbols = ['This', 'is', 'an', 'example', 'sentence.']

        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']  #symbols = ['This', 'is', 'an', 'example', 'sentence.', '<eos>']

        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']


       
    def build_vocab(self):  # 이 함수를 호출함으로써 문장 목록에 있는 단어들에 대하여 매칭 되는 숫자가 생기고 그 값이  self.sym2idx = OrderedDict() 에 들어가는 것
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            
            ('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()
            print(self.special)
            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size): # 빈도가 가장 높은 max size까지의(ex:5000)단어를 가져옴 sym 단어, cnt 카운트
                if cnt < self.min_freq: break
                self.add_symbol(sym)  

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))   
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = tf.concat(encoded, axis=0)

        return encoded
    
    # def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
    #         add_double_eos=False):
    #     if verbose: print('encoding file {} ...'.format(path))
    #     assert os.path.exists(path)
    #     encoded = []
    #     with open(path, 'r', encoding='utf-8') as f:
    #         for idx, line in enumerate(f):
    #             if verbose and idx > 0 and idx % 500000 == 0:
    #                 print('    line {}'.format(idx))
    #             symbols = self.tokenize(line, add_eos=add_eos,
    #                 add_double_eos=add_double_eos)
                
                
    #             encoded.append(self.convert_to_tensor(symbols))
                
    #     if ordered:
    #     #     encoded = list(chain.from_iterable(encoded))
    #         encoded = tf.concat(encoded, axis=0)

    #     return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_num(symbols))

        if ordered:
            # encoded = tf.concat(encoded,axis=0)
            encoded = list(chain.from_iterable(encoded))

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
    '''{           이렇게 sym2idx 에 단어랑 매칭 되는 숫자가 생긴다. 
    "apple": 0,
    "banana": 1,
    "cherry": 2
    }
    '''

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            # print(self.sym2idx[sym])
            return self.sym2idx[sym] # build voca를 통해 생긴 딕셔너리에서 단어 sym에 일치하는 숫자값을 리턴
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            # assert hasattr(self, 'unk_idx')
            self.idx2sym.append('UNK')
            self.sym2idx[sym] = len(self.idx2sym) - 1
            self.unk_idx = self.sym2idx['<UNK>']

            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    # def convert_to_num(self, symbols):
    #     return self.get_indices(symbols)
    def convert_to_tensor(self, symbols):
        return tf.convert_to_tensor(self.get_indices(symbols), dtype=tf.int32)

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def make_dataset(self, datadir,dataset):
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>','<UNK>']
            print(kwargs)
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass
        
        self.count_file(os.path.join(datadir, 'train.txt'))
        self.count_file(os.path.join(datadir, 'valid.txt'))
        self.count_file(os.path.join(datadir, 'test.txt'))

        self.build_vocab()
        
        train = self.encode_file(os.path.join(datadir, 'train.txt'), ordered=True) 
        valid = self.encode_file(os.path.join(datadir, 'valid.txt'), ordered=True) 
        test = self.encode_file(os.path.join(datadir, 'test.txt'), ordered=True) 

        return train,valid,test

    def __len__(self):
        return len(self.idx2sym)



class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, cache =True):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz#3 #60
        self.bptt = bptt#36 #70
        self.ext_len = ext_len if ext_len is not None else 0
        self.data = data
        self.device = device
        
        
        # Work out how cleanly we can divide the dataset into bsz parts.
        # 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
        self.n_step = len(self.data) // self.bsz
        # print(self.n_step)
        
        sliced_data = tf.slice(self.data,[0],[self.n_step * self.bsz])  
        # sliced_data = self.data[:self.n_step * self.bsz]
        '''# 시작 위치와 슬라이싱할 크기 설정
        begin = [0]  # 첫 번째 차원의 시작 위치는 0
        size = [6]   # 첫 번째 차원에서 6개의 원소를 슬라이싱

        # 데이터를 잘라내기 (tf.slice 사용)
        sliced_data = tf.slice(data, begin, size)  '''

        # Evenly divide the data across the bsz batches.
        new_shape = (self.bsz, -1)  # 나머지 차원은 자동으로 계산됨
        data_reshaped = tf.reshape(sliced_data, new_shape)
        data_transposed = tf.transpose(data_reshaped)

        # sliced_data = np.array(sliced_data)
        # data_reshaped = sliced_data.reshape(new_shape)
        # data_transposed = np.transpose(data_reshaped)
        self.data = data_transposed
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
   
        '''
        로,또,1,등,당,첨 = > 로,또,1    => 로, 등
                        등,당,첨         또, 당
                                        1, 첨
        '''
        '''
        TensorFlow 2.x에서는 tf.device()를 사용하여 장치를 지정하는 방법이 아닙니다. 대신 tf.device()로 장치 컨텍스트를 설정한 후, 해당 컨텍스트 내에서 작업을 수행해야 합니다.
        '''    
        # with tf.device(device):
     

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt


        seq_len = min(bptt, self.data.shape[0] - 1 - i) # # i값이 103227020를 넘지 않는 이상 seq_len = 70


        end_idx = i + seq_len # 70,71,72,73,74......
        beg_idx = max(0, i - self.ext_len) # 0,1,2,3,4,5
        ''' 아래 처럼 첫번째 차원을 자르는 이류
        로,또,1,등,당,첨 = > 로,또,1    => 로, 등
                        등,당,첨         또, 당
                                        1, 첨
        '''
        data = self.data[beg_idx:end_idx] # self.data[0:70],[1:71] ~
        target = self.data[i+1:i+1+seq_len] #self.data[1:71],[2:72] ~
        
 
        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, len(self.data) - 1, self.bptt):
            yield self.get_batch(i) # 제너레이터 yield를  통해 만듬

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter() # tr_iter 등의 변수로 저장되며 변수는 호출 될 때마다 값을 하나씩 벹음