import tqdm.notebook as tqdm_notebook
import more_itertools
import tqdm
import json
import os


class FileIterable:
    """
    File utility. This avoids reading all file in memory when reading the file
    sequentially.

    Usage:
        file = FileIterable.from_filename(path)
    """
    _ext_mapping = {
        'txt': ['txt'],
        'jl': ['jsonl', 'jl'],
        'jsonl': ['jsonl', 'jl']
    }

    def __init__(self, iterable, filename=None):
        self._iterable = more_itertools.seekable(iterable)
        self._filename = filename

        self._ptr = 0  # pointer to the next item
        self._len = None

    @classmethod
    def from_filename(cls, path, func=None, fmt=None):

        if not os.path.exists(path):
            raise ValueError(f'Unkown path: {path}')

        no_fmt_given = fmt is None
        fmt = fmt if fmt is not None else 'txt'
        if not any(path.endswith(ext) for ext in cls._ext_mapping[fmt]):
            print(f'WARNING: path is {path} but format is {fmt}'
                  f'{" (by default)" if no_fmt_given else ""}.')

        return cls(cls.read_file(path, func, fmt), path)

    def to_list(self, use_tqdm=True, desc=None):
        if use_tqdm in [True, 'classic']:
            _tqdm = tqdm.tqdm
        elif use_tqdm == 'notebook':
            _tqdm = tqdm_notebook.tqdm
        else:
            _tqdm = None

        if _tqdm is not None:
            return [item for item in _tqdm(self, total=len(self), desc=desc)]
        return list(self)

    @staticmethod
    def read_file(path, func=None, fmt='txt'):
        """
        ARGS:
            path (str): hopefully a valid path.
                        Each line contains a sentence
                        Tokens are separated by a space ' '
            func (NoneType): Should be None in this simple case
            fmt (str): How we procede with each line [txt | jl]
        """
        if fmt == 'txt':
            def _read_line(line):
                return line.strip().split()
        elif fmt in ['jl', 'jsonl']:
            def _read_line(line):
                return json.loads(line)
        else:
            raise ValueError(f'Unkown file format {fmt}')

        assert func is None
        with open(path, mode="r", encoding="utf8") as f:
            for line in f:
                yield _read_line(line)

    def __getitem__(self, item):
        if isinstance(item, slice):
            # unpack the slice manually
            start = item.start if item.start else 0
            stop = item.stop if item.stop else 0
            step = item.step if item.step else 1

            return self[range(start, stop, step)]
        elif isinstance(item, range):
            return [self[i] for i in item]
        elif isinstance(item, (list, tuple)):
            assert all(isinstance(i, int) for i in item)
            return [self[i] for i in item]
        elif isinstance(item, int):
            self._iterable.seek(item)
            self._ptr = item
            try:
                return next(self)
            except StopIteration:
                raise IndexError(f'{item} is outside this generator')
        else:
            raise ValueError(f'Can get item {item} of type {type(item)}. '
                             'This class only supports slices/list[int]/tuple[int]/int.')

    def __next__(self):
        self._ptr += 1
        return next(self._iterable)

    def _length_from_ilen(self):
        ptr = self._ptr  # remember the position of the pointer
        length = more_itertools.ilen(self._iterable)  # count remaining items
        _ = self[ptr - 1]  # set back the pointer
        return length + ptr

    def __len__(self):
        if self._len is None:
            if self._filename is not None:
                self._len = int(os.popen(f'wc -l < {self._filename}').read())
            else:
                self._len = self._length_from_ilen()
        return self._len
