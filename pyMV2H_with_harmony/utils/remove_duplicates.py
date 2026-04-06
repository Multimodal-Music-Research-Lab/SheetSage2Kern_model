from pyMV2H_with_harmony.utils.music import Music


def remove_duplicates_keys(music: Music):
    """
    Remove duplicate keys based on time, with this we don't need a OrderedSet
    :param music: Music to remove the duplicate keys
    """
    copy_keys = list(music.__keys__)
    times = list()
    for key in copy_keys:
        if key.time in times:
            copy_keys.remove(key)
        else:
            times.append(key.time)
    copy_keys.sort(key=lambda element: element.time)
    music.__keys__ = copy_keys


def remove_duplicates_tatums(music: Music):
    """
    Remove duplicate tatums based on time, with this we don't need a OrderedSet
    :param music: Music to remove the duplicate keys
    """
    copy_tatums = list(music.__tatums__)
    times = list()
    for tatum in copy_tatums:
        if tatum.time in times:
            copy_tatums.remove(tatum)
        else:
            times.append(tatum.time)
    copy_tatums.sort(key=lambda element: element.time)
    music.__tatums__ = copy_tatums


def remove_duplicates_hierarchy(music: Music):
    """
    Remove duplicate hierarchy based on time, with this we don't need a OrderedSet
    :param music: Music to remove the duplicate keys
    """
    copy_hierarchies = list(music.__hierarchy__)
    times = list()
    for hierarchy in copy_hierarchies:
        if hierarchy.time in times:
            copy_hierarchies.remove(hierarchy)
        else:
            times.append(hierarchy.time)
    copy_hierarchies.sort(key=lambda element: element.time)
    music.__hierarchy__ = copy_hierarchies
