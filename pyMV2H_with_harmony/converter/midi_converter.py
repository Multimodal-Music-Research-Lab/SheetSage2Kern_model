import pretty_midi
from mido import MidiFile

from pyMV2H_with_harmony.converter.base import BaseConverter
from pyMV2H_with_harmony.utils.pojos import (
    HIERARCHY,
    KEY,
    MIDI_TIME_SIGNATURE,
    NOTE,
    TATUM,
)


class MidiConverter(BaseConverter):
    def is_valid_file(self) -> bool:
        try:
            pretty_midi.PrettyMIDI(midi_file=self.__file__)
            return True
        except OSError:
            return False

    def get_tatum(self) -> list:
        mido_midi = MidiFile(self.__file__)
        micro_seconds_per_quarter = 500000  # Midi default value

        for event in mido_midi:
            if event.type == "set_tempo":
                micro_seconds_per_quarter = event.tempo

        time_signature = self.__midi_time_signature__

        milli_seconds_per_quarter = micro_seconds_per_quarter / 1_000
        milli_per_sub_beat = self.get_millis_per_sub_beat(
            milli_seconds_per_quarter, time_signature
        )
        current_time = 0.0
        file_duration = int(mido_midi.length * 1_000)
        tatums = list()
        while current_time <= file_duration:
            tatums.append(TATUM(current_time))
            current_time += milli_per_sub_beat
        return tatums

    def get_notes(self) -> list:
        midi = pretty_midi.PrettyMIDI(midi_file=self.__file__)
        notes = list()
        for voice, instrument in enumerate(midi.instruments):
            for note in instrument.notes:
                notes.append(
                    NOTE(
                        note.pitch,
                        note.start * 1_000,
                        note.start * 1_000,
                        note.end * 1_000,
                        voice,
                    )
                )
        return notes

    def get_millis_per_sub_beat(
        self, milli_seconds_per_quarter: float, time_signature: MIDI_TIME_SIGNATURE
    ) -> float:
        return milli_seconds_per_quarter / self.sub_beats_per_quarter(time_signature)

    def sub_beats_per_quarter(self, time_signature: MIDI_TIME_SIGNATURE) -> float:
        return (
            time_signature.denominator / 2
            if time_signature.numerator < 4 or time_signature.numerator % 3 != 0
            else time_signature.denominator / 4
        )

    @property
    def __midi_time_signature__(self) -> MIDI_TIME_SIGNATURE:
        time_signature = MIDI_TIME_SIGNATURE(4, 4)  # create default time signature 4/4
        mido_midi = MidiFile(self.__file__)
        for event in mido_midi:
            if event.type == "time_signature":
                return MIDI_TIME_SIGNATURE(event.numerator, event.denominator)
        return time_signature

    def get_key_signature(self) -> list:
        mido_midi = MidiFile(self.__file__)
        all_keys = list()
        for event in mido_midi:
            if event.type == "key_signature":
                key = self.__read_key__(event.key)
                is_minor = "m" in event.key
                all_keys.append(KEY(key, not is_minor, int(event.time * 1_000)))
        return all_keys

    def get_hierarchy(self) -> HIERARCHY:
        time_signature = self.__midi_time_signature__
        beats_per_measure = time_signature.numerator
        sub_beats_per_beat = 2
        if time_signature.numerator > 3 and time_signature.numerator % 3 == 0:
            beats_per_measure = time_signature.numerator / 3
            sub_beats_per_beat = 3
        return HIERARCHY(beats_per_measure, sub_beats_per_beat, 1, 0, 0)

    def __read_key__(self, key):
        from pyMV2H_with_harmony.utils.midi_meta_tonic_map import key_signature_decode

        for midi_entry, value in key_signature_decode.items():
            if key == value:
                return midi_entry[0]
        return 0
