from functools import cmp_to_key


def note_comparator():

    def compare_note_property(x, y, property_name: str) -> int:
        x_value = getattr(x, property_name)
        y_value = getattr(y, property_name)

        if x_value != y_value:
            return -1 if x_value < y_value else 1
        return 0

    def compare(x, y):

        if x.on_val != y.on_val:
            return compare_note_property(x, y, 'on_val')

        if x.pitch != y.pitch:
            return compare_note_property(x, y, 'pitch')

        if x.on != y.on:
            return compare_note_property(x, y, 'on')

        if x.voice != y.voice:
            return compare_note_property(x, y, 'voice')

        if x.off_val != y.off_val:
            return compare_note_property(x, y, 'off_val')

        return 0

    return cmp_to_key(compare)


def cluster_comparator():

    def compare(x, y):

        if x[0] != y[0]:
            return -1 if x[0] < y[0] else 1

        if x[1] != y[1]:
            return -1 if x[1] < y[1] else 1

        return 0

    return cmp_to_key(compare)
