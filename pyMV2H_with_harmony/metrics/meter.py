from ..utils.matches import grouping_match
from ..utils.music import Music
from ..utils.pojos import GROUPING, HIERARCHY, TATUM
from ..utils.remove_duplicates import (
    remove_duplicates_hierarchy,
    remove_duplicates_tatums,
)
from .f1 import f1_score


def meter_score(p_music: Music, t_music: Music):
    p_music.read_if_needed()
    t_music.read_if_needed()

    t_groupings = _get_groupings(t_music)
    p_groupings = _get_groupings(p_music)

    true_positives = 0

    for t_group in t_groupings:
        p_group_index = 0
        while p_group_index < len(p_groupings):
            p_group = p_groupings[p_group_index]
            if grouping_match(t_group, p_group):
                true_positives += 1
                p_groupings.pop(p_group_index)
                p_group_index += 1
                break
            p_group_index += 1

    false_positives = len(t_groupings) - true_positives
    false_negatives = len(p_groupings)

    return f1_score(true_positives, false_positives, false_negatives)


def _get_groupings(music: Music) -> [GROUPING]:
    # Remove any duplicated key or tatum and sort them
    remove_duplicates_tatums(music)
    remove_duplicates_hierarchy(music)

    groups = list()

    tatums_iter = iter(music.__tatums__)
    hierarchies_iter = iter(music.__hierarchy__)

    current_tatum: TATUM = next(tatums_iter, None)
    current_hierarchy: HIERARCHY = next(hierarchies_iter, None)

    if current_tatum is None or current_hierarchy is None:
        return list()

    next_tatum = next(tatums_iter, None)
    next_hierarchy = next(hierarchies_iter, None)

    while next_hierarchy is not None and next_hierarchy.time <= next_tatum.time:
        current_hierarchy = next_hierarchy
        next_hierarchy = next(hierarchies_iter, None)

    tatums_per_sub_beat = current_hierarchy.tpsb
    tatums_per_beat = tatums_per_sub_beat * current_hierarchy.sbpb
    tatums_per_bar = tatums_per_beat * current_hierarchy.bpb

    tatum_num = (
        0 if current_hierarchy.al == 0 else tatums_per_bar - current_hierarchy.al
    )
    sub_beat_start = (
        current_tatum.time if (tatum_num % tatums_per_sub_beat == 0) else -1
    )
    beat_start = current_tatum.time if (tatum_num % tatums_per_beat == 0) else -1
    bar_start = current_tatum.time if (tatum_num % tatums_per_bar == 0) else -1

    while next_tatum is not None:
        current_tatum = next_tatum
        next_tatum = next(tatums_iter, None)
        tatum_num += 1
        while next_hierarchy is not None and next_hierarchy.time <= next_tatum.time:
            current_hierarchy = next_hierarchy
            next_hierarchy = next(hierarchies_iter, None)

            tatums_per_sub_beat = current_hierarchy.tpsb
            tatums_per_beat = tatums_per_sub_beat * current_hierarchy.sbpb
            tatums_per_bar = tatums_per_beat * current_hierarchy.bpb
            tatum_num = (
                0
                if current_hierarchy.al == 0
                else tatums_per_bar - current_hierarchy.al
            )

        # Check for grouping starts/ends

        # Sub beat
        if tatum_num % tatums_per_sub_beat == 0:
            if sub_beat_start != -1:
                groups.append(GROUPING(sub_beat_start, current_tatum.time))
                sub_beat_start = current_tatum.time

        # Beat
        if tatum_num % tatums_per_beat == 0:
            if beat_start != -1:
                groups.append(GROUPING(beat_start, current_tatum.time))
                beat_start = current_tatum.time

        # Bar
        if tatum_num % tatums_per_bar == 0:
            if bar_start != -1:
                groups.append(GROUPING(bar_start, current_tatum.time))
                bar_start = current_tatum.time

    return groups
