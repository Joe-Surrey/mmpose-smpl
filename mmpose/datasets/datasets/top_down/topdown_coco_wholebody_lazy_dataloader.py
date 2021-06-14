from torch.utils.data import DataLoader
from functools import wraps



def with_lazy_load(func):
    @wraps(func)
    def lazy_load(self, *args, **kwargs):
        #print(func)
        if not self.loaded:
            self.load()
        return func(self, *args, **kwargs)

    return lazy_load


class TopDownCocoWholeBodyLazyDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TopDownCocoWholeBodyLazyDataloader, self).__init__(*args, **kwargs)
        self.loaded = False
        self.prev = None

    def load(self):
        print("loading")
        if not self.loaded:
            # load this one
            self.dataset.load()
            self.loaded = True


    def unload(self):
        self.loaded = False
        self.dataset.unload()


    @with_lazy_load
    def __iter__(self):
        # unload prev
        if self.prev is not None:  # unload previous
            print(f"unloading: {self.prev.dataset.annotations_path}")
            self.prev.unload()


        return super(TopDownCocoWholeBodyLazyDataloader, self).__iter__()

    @with_lazy_load
    def __len__(self):
        if not self.loaded:
            self.dataset.load()
            self.loaded = True
        return super(TopDownCocoWholeBodyLazyDataloader, self).__len__()