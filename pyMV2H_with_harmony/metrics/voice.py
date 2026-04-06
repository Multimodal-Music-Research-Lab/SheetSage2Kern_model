from functools import lru_cache

from ..utils.align_files import create_list_of_size
from ..utils.matches import match_note_list, note_match
from ..utils.music import Music
from ..utils.voice import Voice
from .f1 import f1_score


@lru_cache(maxsize=128)
def voice_score(p_music: Music, t_music: Music, return_match_mapping=False):
    p_music.read_if_needed()
    t_music.read_if_needed()

    provided_voices = [Voice() for _ in range(len(p_music.__voices__))]
    transcription_voices = [Voice() for _ in range(len(t_music.__voices__))]
    p_note_mapping = match_note_list(p_music.__notes__, t_music.__notes__)

    match_mapping = list()

    for t_note in p_note_mapping.keys():
        p_note = p_note_mapping[t_note]
        provided_voices[p_note.voice].add_note(note=p_note)
        transcription_voices[t_note.voice].add_note(note=t_note)

    for voice in provided_voices:
        voice.create_connections()

    for voice in transcription_voices:
        voice.create_connections()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # iterate over all voices on transcription
    for voice in transcription_voices:
        # go to each cluster in the transcription voice
        for note_cluster in voice.__note_clusters__.values():
            #  Create list of notes which are linked to in the transcription
            next_transcription_notes = list()
            for next_transcription_cluster in note_cluster.next_clusters:
                next_transcription_notes += next_transcription_cluster.notes

            # Go through each note in the note cluster
            for t_note in note_cluster.notes:
                p_note = p_note_mapping[t_note]

                # Find the matching ground truth note and its place in its voice
                p_voice: Voice = provided_voices[p_note.voice]
                p_cluster = p_voice.get_cluster(p_note)

                # Create list of notes which are linked to in the ground truth
                next_p_notes_final = list()
                for cluster in p_cluster.next_clusters:
                    next_p_notes_final += cluster.notes

                # Count how many tp, fp, and fn for these connection sets
                connection_true_positives = len(
                    match_note_list(next_transcription_notes, next_p_notes_final).keys()
                )
                connection_false_positives = abs(
                    connection_true_positives - len(next_transcription_notes)
                )
                connection_false_negatives = abs(
                    connection_true_positives - len(next_p_notes_final)
                )

                # Normalize counts before adding to totals, so that each connection is weighted equally
                out_weight = (
                    len(next_p_notes_final) + len(next_transcription_notes)
                ) / 2.0
                if out_weight > 0:
                    true_positives += connection_true_positives / (
                        out_weight * len(note_cluster.notes)
                    )
                    false_positives += connection_false_positives / (
                        out_weight * len(note_cluster.notes)
                    )
                    false_negatives += connection_false_negatives / (
                        out_weight * len(note_cluster.notes)
                    )

                if return_match_mapping:
                    # List of notes which are linked to in the original ground truth (including multi-pitch non-TPs)
                    next_original_p_notes = list()
                    for next_p_cluster in (
                        p_music.__voices__[p_note.voice]
                        .get_cluster(p_note)
                        .next_clusters
                    ):
                        next_original_p_notes += next_p_cluster.notes

                    # Both are the end of a voice
                    if (
                        len(next_original_p_notes) == 0
                        and len(next_transcription_notes) == 0
                    ):
                        match_mapping.append(t_note)
                    else:
                        match = False
                        for p_next_note in next_original_p_notes:
                            for t_next_note in next_transcription_notes:
                                # Check if at least one original ground truth connection was correct
                                if note_match(p_next_note, t_next_note):
                                    match = True
                                    match_mapping.append(t_note)
                                    break
                            if match:
                                break

    if return_match_mapping:
        return f1_score(true_positives, false_positives, false_negatives), match_mapping
    else:
        return f1_score(true_positives, false_positives, false_negatives)
