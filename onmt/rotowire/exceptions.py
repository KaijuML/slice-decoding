import os


class DataAlreadyExistsError(Exception):
    def __init__(self, filename):
        msg = f'\n\t{filename}\n\tConsider using overwrite=True'
        super(DataAlreadyExistsError, self).__init__(msg)


class RotowireParsingError(Exception):
    def __init__(self, error_logs):
        super().__init__('See error logs above.')


class SliceMismatchError(Exception):
    def __init__(self, trues, found):
        super().__init__()
        
        
class UnknownElaborationError(Exception):
    def __init__(self, grounding_type):
        super().__init__(f'{grounding_type}')


class ContextTooLargeError(Exception):
    def __init__(self, sidx, ctx_size):
        super().__init__()


class MissingTemplateFileError(Exception):
    """
    Usage: template.py
    Raised when template_file is not found
    """
    def __init__(self, filename):
        self.msg = f'{os.path.abspath(filename)} was not found'
        super(MissingTemplateFileError, self).__init__(self.msg)


class MissingPlayer(Exception):
    """
    Usage: template.py
    Raised when user wants access to an unknown player in a PlayerList
    """
    pass


class ElaborationSpecificationError(Exception):
    """
    Usage: template.py
    Raised when an unknown elaboration is specifed
    """
    pass


class UnderspecifiedTemplateError(Exception):
    """
    Usage: template.py
    Raised when specifiers are not enough to pick one and only one entity idx
    """
    pass


class UnexpectedTeamData(Exception):
    """
    Usage: template.py
    Raised when an unknown team info is encountered
    """
    def __init__(self, data):
        self.msg = f'Unknown data: {data}'
        super().__init__()


class UnexpectedPlayerData(Exception):
    """
    Usage: template.py
    Raised when an unknown player info is encountered
    """
    def __init__(self, data):
        self.msg = f'Unknown data: {data}'
        super().__init__()


class SecondMatchGroupError(Exception):
    """
    Usage: template.py
    Raised when a template line is wrongly specified
    """
    def __init__(self, entity, match_group):
        self.msg = f'{entity=} {match_group=}'
        super().__init__()