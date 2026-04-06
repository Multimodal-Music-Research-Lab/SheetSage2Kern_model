
def convert_time(time: int, p_music, t_music, alignment, alignment_times) -> int:

    alignment_time = alignment_times.get(time, None)
    if alignment_time is not None:
        return alignment_time

    t_index = -1
    t_notes = t_music.get_notes_grouped_by_onset()

    # Find the correct transcription anchor index to start with
    for index, t_note in enumerate(t_notes):
        if t_note[0].on == time:
            # Time matches an anchor exactly
            t_index = index
            break

        if t_note[0].on > time:
            # This anchor is past the time
            t_index = index - 0.5
            break

    p_notes = p_music.get_notes_grouped_by_onset()
    p_previous_anchor = -1
    p_previous_previous_anchor = -1
    p_next_anchor = len(p_notes)
    p_next_next_anchor = len(p_notes)

    for index, current_alignment in enumerate(alignment):
        if current_alignment != -1:
            if current_alignment == t_index:
                # This is the correct time, exactly on the index
                return p_notes[index][0].on
            elif current_alignment < t_index:
                # The time is past this anchor
                p_previous_previous_anchor = p_previous_anchor
                p_previous_anchor = index
            else:
                #  We are past the time
                if p_next_anchor == len(p_notes):
                    # This is the first anchor for which we are past the time
                    p_next_anchor = index
                else:
                    # This is the 2nd anchor for which we are past the time
                    p_next_next_anchor = index
                    break

    if p_previous_anchor == -1 and p_next_anchor == len(p_notes):
        # Nothing was aligned
        alignment_times[time] = time
        return time

    if p_previous_anchor == -1:
        if p_next_anchor != len(p_notes):
            alignment_time = _convert_time(
                time,
                p_next_anchor,
                p_next_next_anchor,
                p_notes,
                t_notes,
                alignment
            )
        else:
            # Only 1 anchor. Just linear shift.
            alignment_time = (
                    time -
                    t_notes[alignment[p_next_anchor]][0].on +
                    p_notes[alignment[p_next_anchor]][0].on
            )
    elif p_next_anchor == len(p_notes):
        # Time is after the last anchor. Use the previous rate.
        if p_previous_anchor != -1:
            alignment_time = _convert_time(
                time,
                p_previous_previous_anchor,
                p_previous_anchor,
                p_notes,
                t_notes,
                alignment
            )
        else:
            # Only 1 anchor. Just linear shift.
            alignment_time = (
                    time -
                    t_notes[alignment[p_previous_anchor]][0].on +
                    p_notes[alignment[p_previous_anchor]][0].on
            )
    else:
        # Time is between anchor points
        alignment_time = _convert_time(
                time,
                p_previous_anchor,
                p_next_anchor,
                p_notes,
                t_notes,
                alignment
        )

    alignment_times[time] = alignment_time
    return int(alignment_time)


def _convert_time(time, previous_anchor, next_anchor, p_notes, t_notes, alignment) -> int:
    p_previous_time = p_notes[previous_anchor][0].on
    p_next_time = p_notes[next_anchor][0].on

    t_previous_time = t_notes[alignment[previous_anchor]][0].on
    t_next_time = t_notes[alignment[next_anchor]][0].on

    rate = ((p_next_time - p_previous_time) / (t_next_time - t_previous_time))
    return round(rate * (time - t_previous_time) + p_previous_time)

