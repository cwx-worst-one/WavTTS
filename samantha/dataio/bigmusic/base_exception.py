class MusicMetaError(Exception):
    """Raise this error when the entire item is supposed to be discarded"""

    pass


class MusicMetaMapError(Exception):
    """Raise this error when the element in the list is supposed to be filtered out"""

    pass
