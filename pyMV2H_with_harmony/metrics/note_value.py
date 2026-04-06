from ..metrics.multi_pitch import multi_pitch_accuracy
from ..metrics.voice import voice_score
from ..utils.algorithm_config import DURATION_DELTA
from ..utils.music import Music
from ..utils.pojos import NOTE


def note_value_score(
    p_music: Music, t_music: Music, multi_pitch_match=None, voice_match=None
) -> float:
    p_music.read_if_needed()
    t_music.read_if_needed()

    notes_check = voice_match
    multi_pinch_check = multi_pitch_match

    if notes_check is None:
        _, notes_check = voice_score(p_music, t_music, return_match_mapping=True)

    if multi_pinch_check is None:
        _, multi_pinch_check = multi_pitch_accuracy(
            p_music.__notes__, t_music.__notes__, return_match_notes=True
        )
    score_sum = 0.0
    for t_note in notes_check:
        score_sum += _get_note_score(multi_pinch_check[t_note], t_note)
    if len(notes_check) == 0:
        return 0
    return score_sum / len(notes_check)


def _get_note_score(p_note: NOTE, t_note: NOTE) -> float:
    p_duration = abs(p_note.on_val - p_note.off_val)
    t_duration = abs(t_note.on_val - t_note.off_val)

    diff = abs(p_duration - t_duration)
    if diff < DURATION_DELTA:
        return 1.0

    return max(0.0, 1.0 - diff / p_duration)
