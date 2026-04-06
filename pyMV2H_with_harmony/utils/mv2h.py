from ..utils.pojos import MV2H_WEIGHTS


class MV2H:
    def __init__(
        self, multi_pitch, voice, harmony, meter, note_value, weights=MV2H_WEIGHTS()
    ):
        self.__multi_pitch__ = multi_pitch
        self.__voice__ = voice
        self.__harmony__ = harmony
        self.__meter__ = meter
        self.__note_value__ = note_value
        self.__weights__ = weights

    def __sum_weights__(self) -> float:
        return (
            self.__weights__.multi_pitch
            + self.__weights__.voice
            + self.__weights__.harmonic
            + self.__weights__.metrical
            + self.__weights__.note_value
        )

    def update_weights(self, weights: MV2H_WEIGHTS):
        self.__weights__ = weights

    @property
    def multi_pitch(self) -> float:
        return self.__multi_pitch__ * self.__weights__.multi_pitch

    @property
    def voice(self) -> float:
        return self.__voice__ * self.__weights__.voice

    @property
    def harmony(self) -> float:
        return self.__harmony__ * self.__weights__.harmonic

    @property
    def meter(self) -> float:
        return self.__meter__ * self.__weights__.metrical

    @property
    def note_value(self) -> float:
        return self.__note_value__ * self.__weights__.note_value

    @property
    def mv2h(self) -> float:
        return (
            self.meter + self.voice + self.harmony + self.note_value + self.multi_pitch
        ) / self.__sum_weights__()

    def __gt__(self, other):
        other: MV2H = other
        if self.mv2h != other.mv2h:
            return self.mv2h > other.mv2h

        if self.multi_pitch != other.multi_pitch:
            return self.multi_pitch > other.multi_pitch

        if self.voice != other.voice:
            return self.voice > other.voice

        if self.meter != other.meter:
            return self.meter > other.meter

        if self.note_value != other.note_value:
            return self.note_value > other.note_value

        if self.harmony != other.harmony:
            return self.harmony > other.harmony

        return False

    def __lt__(self, other):
        other: MV2H = other
        if self.mv2h != other.mv2h:
            return self.mv2h < other.mv2h

        if self.multi_pitch != other.multi_pitch:
            return self.multi_pitch < other.multi_pitch

        if self.voice != other.voice:
            return self.voice < other.voice

        if self.meter != other.meter:
            return self.meter < other.meter

        if self.note_value != other.note_value:
            return self.note_value < other.note_value

        if self.harmony != other.harmony:
            return self.harmony < other.harmony

        return False

    def __repr__(self):
        return f"""
Multi-pitch:    {self.multi_pitch}
Voice:          {self.voice}
Meter:          {self.meter}
Value:          {self.note_value}
Harmony:        {self.harmony}
MV2H:           {self.mv2h}
WEIGHTS:        {self.__weights__}
        """
