

def normalize_list_size(list_to_normalize: list, new_length: int, default_value=None):
    """
    Fill the list to get a x length
    :param list_to_normalize: list to have the new size
    :param new_length: the new length for the list
    :param default_value: the value to fill the list to get the new size
    """
    if len(list_to_normalize) >= new_length:
        return

    len_difference = new_length - len(list_to_normalize)
    list_to_normalize += [default_value] * len_difference
