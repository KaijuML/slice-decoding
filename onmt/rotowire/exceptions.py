class DataAlreadyExistsError(Exception):
    def __init__(self, filename):
        msg = f'\n\t{filename}\n\tConsider using overwrite=True'
        super(DataAlreadyExistsError, self).__init__(msg)


class RotowireParsingError(Exception):
    def __init__(self, error_logs):
        msg = '\n'
        for err_name, idxs in error_logs.items():
            msg += f'{err_name} at {idxs=}\n'
        super().__init__(msg)


class SliceMismatchError(Exception):
    def __init__(self, trues, found):
        super().__init__()