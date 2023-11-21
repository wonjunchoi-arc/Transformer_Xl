
from data import Dataset

datadir = '/home/jun/workspace/wiki_short/'
data = 'wt103'

kwargs = {}
if data in ['wt103', 'wt2']:
    kwargs['special'] = ['<eos>','<UNK>']
    kwargs['lower_case'] = False

dataset = Dataset(**kwargs)


train, val, test = dataset.make_dataset(datadir,data)
print(train)