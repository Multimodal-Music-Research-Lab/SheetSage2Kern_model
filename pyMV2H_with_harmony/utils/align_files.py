import math
from functools import reduce

from tqdm import tqdm

from pyMV2H_with_harmony.metrics import f1
from pyMV2H_with_harmony.utils.algorithm_config import NON_ALIGNMENT_PENALTY
from pyMV2H_with_harmony.utils.alignment_node import AlignmentNode
from pyMV2H_with_harmony.utils.music import Music
from pyMV2H_with_harmony.utils.mv2h import MV2H


def align_files(provided_file: Music, transcription_file: Music):
    """
    Align based on DTW algorithm
    """

    from pyMV2H_with_harmony.metrics.mv2h import mv2h

    provided_file.read_if_needed()
    transcription_file.read_if_needed()

    alignment_nodes = _get_possible_alignments(provided_file, transcription_file)
    total = 0
    best = MV2H(0.0, 0.0, 0.0, 0.0, 0.0)
    best_music = None

    for node in alignment_nodes:
        total += node.count

    with tqdm(total=total) as pbar:
        for node_alignment in alignment_nodes:
            for j in range(node_alignment.count):
                alignment = node_alignment.get_alignment(j)
                shifted_music = transcription_file.align(provided_file, alignment)
                candidate: MV2H = mv2h(provided_file, shifted_music)
                if candidate > best:
                    best = candidate
                    best_music = shifted_music
                pbar.update()

    return best_music, best


def _get_possible_alignments(provided_file: Music, transcription_file: Music):
    previous_cells = _get_align_matrix(provided_file, transcription_file)
    ARRAY_SIZE = len(previous_cells)
    cache_matrix = create_list_of_size(
        ARRAY_SIZE, lambda: create_list_of_size(len(previous_cells[0]), list)
    )
    return _get_possible_alignments_from_matrix(
        len(previous_cells) - 1,
        len(previous_cells[0]) - 1,
        previous_cells,
        cache_matrix,
    )


def _get_possible_alignments_from_matrix(i: int, j: int, matrix: list, cache: list):
    alignments = cache[i][j]
    if not len(alignments) == 0:
        return alignments

    if i == 0 and j == 0:
        return alignments

    for previous_cell in matrix[i][j]:
        if previous_cell == -1:
            # This transcription note was aligned with nothing in the ground truth.
            alignments.append(
                AlignmentNode(
                    _get_possible_alignments_from_matrix(i - 1, j, matrix, cache), -1
                )
            )
        elif previous_cell == 1:
            # This ground truth note was aligned with nothing in the transcription.
            for node in _get_possible_alignments_from_matrix(i, j - 1, matrix, cache):
                if node.value != -1:
                    alignments.append(node)
        else:
            # The current transcription and ground truth notes were aligned.
            alignments.append(
                AlignmentNode(
                    _get_possible_alignments_from_matrix(i - 1, j - 1, matrix, cache),
                    j - 1,
                )
            )
    return alignments


def _get_align_matrix(provided_file: Music, transcription_file: Music):
    provided_notes = provided_file.get_notes_grouped_by_onset()
    transcription_notes = transcription_file.get_notes_grouped_by_onset()

    p_notes_map = _get_note_pitch_maps(provided_notes)
    t_notes_map = _get_note_pitch_maps(transcription_notes)

    ARRAY_SIZE = (len(provided_notes) + 1, len(transcription_notes) + 1)

    previous_cells = create_list_of_size(
        ARRAY_SIZE[0], lambda: create_list_of_size(ARRAY_SIZE[1], list)
    )
    distances = [
        [0 for _ in range(ARRAY_SIZE[1])] for _ in range(ARRAY_SIZE[0])
    ]  # create_list_of_size(ARRAY_SIZE[0], lambda: create_list_of_size(ARRAY_SIZE[1], 0))
    distances[0] = [math.inf] * ARRAY_SIZE[1]
    distances[0][0] = 0.0

    for i in range(1, ARRAY_SIZE[0]):
        distances[i][0] = math.inf

    for j in range(1, ARRAY_SIZE[1]):
        for i in range(1, ARRAY_SIZE[0]):
            distance = get_distance(p_notes_map[i - 1], t_notes_map[j - 1])

            distance_i_1 = distances[i - 1][j] + NON_ALIGNMENT_PENALTY
            distance_j_1 = distances[i][j - 1] + NON_ALIGNMENT_PENALTY
            distance_i_j_1 = distances[i - 1][j - 1] + distance

            min_distance = min(distance_j_1, distance_i_1, distance_i_j_1)
            previous_cell = previous_cells[i][j]
            if distance_i_1 == min_distance:
                previous_cell.append(-1)
            if distance_j_1 == min_distance:
                previous_cell.append(1)
            if distance_i_j_1 == min_distance:
                previous_cell.append(0)
            distances[i][j] = min_distance
    return previous_cells


def _get_note_pitch_maps(notes_lists):
    maps = list()
    for note_list in notes_lists:
        notes_dict = dict()
        maps.append(notes_dict)
        for note in note_list:
            if note.pitch in notes_dict.keys():
                notes_dict[note.pitch] = notes_dict[note.pitch] + 1
            else:
                notes_dict[note.pitch] = 1
    return maps


def get_distance(p_note_map, t_note_map):
    """
    Get the distance between a given ground truth note set and a possible transcription note set.

    :param p_note_map: The pitch map of a ground truth note set.
    :param t_note_map: he pitch map of a possible transcription note set.
    :return: The alignment score. 1 - its F-measure.
    """
    true_positives = 0
    false_positives = 0

    for t_key, count in t_note_map.items():
        if t_key in p_note_map:
            p_count = p_note_map[t_key]
            true_positives += min(count, p_count)
            if count > p_count:
                false_positives += count - p_count
        else:
            false_positives += count

    if true_positives == 0:
        return 1.0

    p_count = sum(p_note_map.values())
    false_negatives = p_count - true_positives
    return 1 - f1.f1_score(true_positives, false_positives, false_negatives)


def create_list_of_size(size: int, value) -> list:
    """
    Helper function to create an list of size x with initial value
    Use this method only for dynamic lists, make preference for numpy arrays initializers like:
    np.zeros(SHAPE) etc.
    :param size: size of the list
    :param value: lambda function to initialize the value
    :return: the list of size x with all elements equal to return of value
    """
    return [value() if callable(value) else value for _ in range(size)]
