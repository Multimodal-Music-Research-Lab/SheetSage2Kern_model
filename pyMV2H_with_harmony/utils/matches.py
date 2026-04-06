from contextlib import nullcontext
from typing import Optional

from pyMV2H_with_harmony.utils.algorithm_config import GROUPING_EPSILON, ONSET_DELTA
from pyMV2H_with_harmony.utils.pojos import GROUPING, NOTE


def note_match(note_a: NOTE, note_b: NOTE) -> bool:
    """
    Return if both notes match using ONSET_DELTA tolerance
    :param note_a: first note to check
    :param note_b: second note to check
    :return: a boolean that indicates if both notes are a match
    """
    return note_a.pitch == note_b.pitch and abs(note_a.on - note_b.on) <= ONSET_DELTA


def grouping_match(group_a: GROUPING, group_b: GROUPING) -> bool:
    """
    Decide whether this grouping matches the given grouping, using the error threshold
    If both its start and end times are within the epsilon
    of this one's start and end times, it is a match.

    :param group_a: first group to check match
    :param group_b: second group to check the match
    :return: a boolean that indicates if both groups are a match
    """
    return (
        abs(group_a.start_time - group_b.start_time) <= GROUPING_EPSILON
        and abs(group_a.end_time - group_b.end_time) <= GROUPING_EPSILON
    )


def match_note_list(
    p_notes: list,
    t_notes: list,
    export_decisions_in: Optional[str] = None,
    line_prefix: Optional[str] = None,
) -> dict:
    match_notes = dict()
    t_notes_copy = list(t_notes)
    p_notes_copy = list(p_notes)

    with (
        open(export_decisions_in, "a") if export_decisions_in else nullcontext()
    ) as csv:
        t_index = 0
        while t_index < len(t_notes_copy):
            t_note = t_notes_copy[t_index]
            p_index = 0
            while p_index < len(p_notes_copy):
                p_note = p_notes_copy[p_index]
                if note_match(t_note, p_note):
                    # Found a match
                    match_notes[t_note] = p_note
                    _write_note_decision(csv, t_note, "tp", line_prefix)
                    t_notes_copy.pop(t_index)
                    p_notes_copy.pop(p_index)
                    t_index -= 1
                    break
                p_index += 1
            t_index += 1

        for p_note in p_notes_copy:
            _write_note_decision(csv, p_note, "fn", line_prefix)

        for t_note in t_notes_copy:
            _write_note_decision(csv, t_note, "fp", line_prefix)

        return match_notes


def _write_note_decision(file, note: NOTE, decision: str, line_prefix: str):
    if file is None:
        return
    line_components = [
        f"{x}" for x in [line_prefix, note.pitch, note.on_val, note.off_val, decision]
    ]
    file.write(",".join(line_components) + "\n")
