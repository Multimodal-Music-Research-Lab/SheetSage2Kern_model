from functools import lru_cache

from ..utils.comparators import note_comparator
from ..utils.convert_time import convert_time
from ..utils.pojos import HIERARCHY, KEY, NOTE, TATUM
from ..utils.voice import Voice


def need_read(fn):
    def inner(*args, **kwargs):
        music_instance = args[0]
        music_instance.read_if_needed()
        return fn(*args, **kwargs)

    return inner


class Music:
    @classmethod
    def from_file(cls, file_name):
        music = cls(list(), list(), list(), list())
        music.__file_name__ = file_name
        music.__has_read__ = False
        return music

    def __init__(self, tatums, keys, notes, hierarchy):
        self.__file_name__ = None
        self.__tatums__ = tatums
        self.__keys__ = keys
        self.__notes__ = notes
        self.__hierarchy__ = hierarchy
        self.__has_read__ = True
        self.__duration__ = 0.0
        self.__voices__ = list()
        self.__post_process__()
        self.__notes_list__ = list()

    def read_if_needed(self):
        if not self.__has_read__:
            self.read()

    def read(self):
        self.__tatums__ = list()
        self.__keys__ = list()
        self.__notes__ = list()
        self.__hierarchy__ = list()
        self.__voices__ = list()
        self.__duration__ = 0.0
        self.__notes_list__ = list()

        with open(self.__file_name__, "r") as file:
            all_lines = file.readlines()
            for line in all_lines:
                self.parse_line(line)
        self.__has_read__ = True
        self.__post_process__()

    def parse_line(self, line):
        if line.startswith("Note"):
            self.__parse_note__(line)
        if line.startswith("Key"):
            self.__parse_key__(line)
        if line.startswith("Hierarchy"):
            self.__parse_hierarchy__(line)
        if line.startswith("Tatum"):
            self.__parse_tatum__(line)

    def __parse_note__(self, line):
        # Note 47 46000 46000 46474 1
        args = [int(x) for x in line.split(" ")[1:]]
        note = NOTE(*args)
        self.__notes__.append(note)

    def __parse_tatum__(self, line):
        # Tatum 3750
        args = [int(x) for x in line.split(" ")[1:]]
        self.__tatums__.append(TATUM(*args))

    def __parse_key__(self, line):
        # Key 0 Maj 0
        args = line.split(" ")[1:]
        key_to_add = KEY(int(args[0]), args[1].lower() == "maj", int(args[2]))
        if key_to_add not in self.__keys__:
            self.__keys__.append(key_to_add)

    def __parse_hierarchy__(self, line):
        # Hierarchy 3,2 1 a=0 0
        args = [int(x) for x in line.replace(",", " ").replace("a=", "").split(" ")[1:]]
        if len(args) == 3:
            # time is optional on original description, set it to zero by default
            args.append(0)
        self.__hierarchy__.append(HIERARCHY(*args))

    @need_read
    def get_notes_grouped_by_onset(self):

        if len(self.__notes_list__) != 0:
            return self.__notes_list__

        notes = list()

        most_recent_list = list()
        most_recent_value_onset_time = self.__notes__[0].on
        most_recent_list.append(self.__notes__[0])
        notes.append(most_recent_list)

        for note in self.__notes__[1:]:
            if note.on == most_recent_value_onset_time:
                most_recent_list.append(note)
            else:
                most_recent_list = list()
                most_recent_value_onset_time = note.on
                most_recent_list.append(note)
                notes.append(most_recent_list)
        self.__notes_list__ = notes
        return notes

    def __post_process__(self):
        """
        Create music structure
        """
        if self.__notes__ is None:
            return

        self.__notes__.sort(key=note_comparator())
        for note in self.__notes__:
            self.__duration__ = max(self.__duration__, note.off_val)
            # Create music voices
            while note.voice >= len(self.__voices__):
                self.__voices__.append(Voice())
            # add note to voice
            self.__voices__[note.voice].add_note(note)

        if len(self.__keys__) > 0:
            self.__duration__ = max(
                self.__duration__, max(self.__keys__, key=lambda item: item.time).time
            )
        if len(self.__tatums__) > 0:
            self.__duration__ = max(
                self.__duration__, max(self.__tatums__, key=lambda item: item.time).time
            )

        for voice in self.__voices__:
            voice.create_connections()

    def align(self, p_music, alignment: list):
        align_times = dict()

        # Convert each note into a new note
        new_notes = [
            NOTE(
                note.pitch,
                convert_time(note.on, p_music, self, alignment, align_times),
                convert_time(note.on_val, p_music, self, alignment, align_times),
                convert_time(note.off_val, p_music, self, alignment, align_times),
                note.voice,
            )
            for note in self.__notes__
        ]

        # Convert each hierarchy
        new_hierarchies = [
            HIERARCHY(
                hierarchy.bpb,
                hierarchy.sbpb,
                hierarchy.tpsb,
                hierarchy.al,
                convert_time(hierarchy.time, p_music, self, alignment, align_times),
            )
            for hierarchy in self.__hierarchy__
        ]

        # Convert each tatum
        new_tatums = [
            TATUM(
                convert_time(tatum.time, p_music, self, alignment, align_times),
            )
            for tatum in self.__tatums__
        ]

        new_keys = [
            KEY(
                key.tonic,
                key.is_major,
                convert_time(key.time, p_music, self, alignment, align_times),
            )
            for key in self.__keys__
        ]

        return Music(new_tatums, new_keys, new_notes, new_hierarchies)
