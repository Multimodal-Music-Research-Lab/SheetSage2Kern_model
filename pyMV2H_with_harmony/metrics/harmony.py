from ..utils.music import Music
from ..utils.pojos import KEY
from ..utils.remove_duplicates import remove_duplicates_keys


def harmony_score(p_music: Music, t_music: Music):
    p_music.read_if_needed()
    t_music.read_if_needed()

    remove_duplicates_keys(p_music)
    remove_duplicates_keys(t_music)

    if len(p_music.__keys__) == 0:
        # TODO: return chord progression score
        return 0.0

    duration = p_music.__duration__
    score_map = dict()

    for t_index in range(len(t_music.__keys__)):
        t_key = t_music.__keys__[t_index]
        next_t_key_time = (
            duration
            if t_index == len(t_music.__keys__) - 1
            else min(duration, t_music.__keys__[t_index + 1].time)
        )

        # Go through each ground truth key
        for p_index in range(len(p_music.__keys__)):
            p_key = p_music.__keys__[p_index]
            next_p_key_time = (
                duration
                if p_index == len(p_music.__keys__) - 1
                else min(duration, p_music.__keys__[p_index + 1].time)
            )

            overlap_beginning = max(t_key.time, p_key.time)
            overlap_ending = max(next_t_key_time, next_p_key_time)

            # check for valid overlap
            if overlap_ending > overlap_beginning:
                overlap_len = overlap_ending - overlap_beginning
                score = _get_key_score(p_key, t_key)

                # add score to map
                if score not in score_map.keys():
                    score_map[score] = overlap_len
                else:
                    score_map[score] += overlap_len

    weighted_correct_duration = 0.0
    for score in score_map.keys():
        weighted_correct_duration += score * score_map[score]

    return weighted_correct_duration / duration


def _get_key_score(p_note: KEY, t_note: KEY) -> float:
    """
    Get the score of a transcribed key given some ground truth.
    :param p_note:
    :param t_note:
    :return: The score of the transcribed key. 1 for a perfect match,
     0.5 for correct mode but tonic off by a perfect 5th,
      0.3 for relative major or minor(CM, am),
      0.2 for parallel major or minor (CM, cm), and 0 otherwise.
    """
    # Correct
    if p_note.tonic == t_note.tonic and p_note.is_major == t_note.is_major:
        return 1.0

    # Perfect fifth higher
    if p_note.is_major == t_note.is_major and p_note.tonic == (t_note.tonic + 7) % 12:
        return 0.5

    # Perfect fifth lower
    if p_note.is_major == t_note.is_major and p_note.tonic == (t_note.tonic + 5) % 12:
        return 0.5

    # Relative major or minor
    if p_note.is_major != t_note.is_major and p_note.tonic == (t_note.tonic + 3) % 12:
        return 0.3

    # parallel major or minor
    if p_note.tonic == t_note.tonic and p_note.is_major != t_note.is_major:
        return 0.2

    return 0.0
