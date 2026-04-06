
class AlignmentNode:
    def __init__(self, prev_list, value):
        self.prev_list = prev_list
        self.value = value
        count = 0
        if prev_list is not None:
            for prev in prev_list:
                count += prev.count
        self.count = max(count, 1)

    def get_alignment(self, index):

        alignment = None

        if self.prev_list is None or len(self.prev_list) == 0:
            alignment = list()
        else:
            for prev in self.prev_list:
                if index < prev.count:
                    alignment = prev.get_alignment(index)
                    break
                index -= prev.count
        alignment.append(self.value)
        return alignment
