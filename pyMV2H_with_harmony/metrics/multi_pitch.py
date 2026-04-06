from functools import lru_cache

from .f1 import f1_score
from ..utils.freeze_args import freeze_args
from ..utils.matches import note_match, match_note_list


@freeze_args()
@lru_cache(maxsize=128)
def multi_pitch_accuracy(
    p_notes: list,
    t_notes: list,
    return_match_notes=False,
    return_f1_detail=False,
    details_line_prefix=None,
    export_details_in_file=None
):
    """
    Computes the multi-pitch-accuracy metric.
    Parameters:
        p_notes (list): List of notes predicted by the model.
        t_notes (list): List of notes of the source truth.
        return_match_notes (bool, optional): Whether to return the match notes.
        return_f1_detail (bool, optional): Whether to return the f1 score details (precision, recall, f1 score).
        details_line_prefix (str, optional): A prefix string to prepend to each line of the details file.
        export_details_in_file (path, optional): Write per-note decision of f1_score in format: (prefix, note, start, end, [tp, fp, fn])
    """
    match_notes = match_note_list(
        p_notes, t_notes,
        export_decisions_in=export_details_in_file, line_prefix=details_line_prefix
    )
    true_positives = len(match_notes.keys())

    if return_match_notes:
        return f1_score(
            true_positives=true_positives,
            false_positives=abs(true_positives - len(t_notes)),
            false_negatives=abs(true_positives - len(p_notes)),
            return_details=return_f1_detail
        ), match_notes
    else:
        return f1_score(
            true_positives=true_positives,
            false_positives=abs(true_positives - len(t_notes)),
            false_negatives=abs(true_positives - len(p_notes)),
            return_details=return_f1_detail
        )
