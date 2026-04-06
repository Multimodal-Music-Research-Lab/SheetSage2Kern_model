from ..metrics.harmony import harmony_score
from ..metrics.meter import meter_score
from ..metrics.multi_pitch import multi_pitch_accuracy
from ..metrics.note_value import note_value_score
from ..metrics.voice import voice_score
from ..utils.music import Music
from ..utils.mv2h import MV2H
from ..utils.pojos import MV2H_WEIGHTS


def mv2h(p_music: Music, t_music: Music, weights: MV2H_WEIGHTS = MV2H_WEIGHTS()):
    p_music.read_if_needed()
    t_music.read_if_needed()

    multi_pitch, multi_pitch_match = multi_pitch_accuracy(
        p_music.__notes__, t_music.__notes__, True
    )
    multi_pitch *= weights.multi_pitch

    voice, voice_match = voice_score(p_music, t_music, True)
    voice *= weights.voice
    harmony = harmony_score(p_music, t_music) * weights.harmonic
    meter = meter_score(p_music, t_music) * weights.metrical
    note_value = (
        note_value_score(p_music, t_music, multi_pitch_match, voice_match)
        * weights.note_value
    )

    return MV2H(multi_pitch, voice, harmony, meter, note_value, weights)
