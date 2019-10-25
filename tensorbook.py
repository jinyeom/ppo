import torch as pt

class TensorBook:
  def __init__(self, title, size):
    self.title = title # book title
    self.size = size   # number of pages
    self.appx = {}     # appendix
    self._t = 0        # current page

  def __getitem__(self, index):
    row = {}
    for k, v in self.__dict__.items():
      if isinstance(v, pt.Tensor):
        row[k] = v[index]
    return row

  def __repr__(self):
    return f'TensorBook({self.title}, {self.size})'
  
  @property
  def contents(self):
    n = max(len(k) for k in self.__dict__.keys())
    s = self.title + '\n' + '\n'.join([
      '- {} ... {}'.format(k+(n-len(k))*' ', tuple(v.shape))
      for k, v in self.__dict__.items()
      if isinstance(v, pt.Tensor)
    ])
    if self.appx:
      n = max(len(k) for k in self.appx.keys())
      s += '\n- appendix\n' + '\n'.join([
        '  - {} ... {}'.format(k+(n-len(k))*' ', len(v))
        for k, v in self.appx.items()
      ])
    return s
  
  def full(self):
    return self._t == self.size

  def reset(self, hard=False):
    for k, v in self.__dict__.items():
      if isinstance(v, pt.Tensor):
        if hard:
          delattr(self, k)
        else:
          getattr(self, k).zero_()
    self.appx = {}
    self._t = 0

  def append(self, **kwargs):
    if self.full():
      # If the book is out of pages, and if you wish to add more 
      # content, you can add it to the appendix section.
      # Note that it's not meant to be used as a main storage;
      # if you end up with a large appendix, perhaps you should
      # get a book with more pages.
      for k, v in kwargs.items():
        if not k in self.appx:
          self.appx[k] = []
        self.appx[k].append(v.detach().cpu())
    else:
      for k, v in kwargs.items():
        if not hasattr(self, k):
          setattr(self, k, pt.zeros(self.size, *v.shape))
        getattr(self, k)[self._t] = v.detach().cpu()
      self._t += 1

  def export(self, path):
    if not os.path.isdir(path):
      os.makedirs(path)
    
    for k, v in self.__dict__.items():
      if isinstance(v, pt.Tensor):
        filename = '{}.pt'.format(k)
        filename = os.path.join(path, filename)
        pt.save(v, filename)
