import abc
import os
from ctypes import ArgumentError

# from math import floor
from pyMV2H_with_harmony.utils.comparators import note_comparator
from pyMV2H_with_harmony.utils.midi_meta_tonic_map import key_signature_decode
from pyMV2H_with_harmony.utils.pojos import HIERARCHY


class BaseConverter(abc.ABC):
    def __init__(self, file, output):
        self.__file__ = file
        self.__output__ = output
        if not self.__check_if_file_exists__():
            raise ArgumentError("file", f"{self.__file__} not exists")

    def __check_if_file_exists__(self) -> bool:
        return os.path.exists(self.__file__)

    @abc.abstractmethod
    def is_valid_file(self) -> bool:
        pass

    @abc.abstractmethod
    def get_tatum(self) -> list:
        pass

    @abc.abstractmethod
    def get_notes(self) -> list:
        pass

    @abc.abstractmethod
    def get_hierarchy(self) -> HIERARCHY:
        pass

    @abc.abstractmethod
    def get_key_signature(self) -> list:
        pass

    def convert_file(self):
        f = open(self.__output__, "w")
        # Write Notes
        notes = self.get_notes()
        notes.sort(key=note_comparator())
        f.writelines(
            [
                f"Note {round(x.pitch)} {round(x.on)} {round(x.on_val)} {round(x.off_val)} {round(x.voice)}\n"
                for x in notes
            ]
        )
        # Write Tatums
        tatums = self.get_tatum()
        f.writelines([f"Tatum {round(x.time)}\n" for x in tatums])

        # Write Hierarchy
        hierarchy = self.get_hierarchy()
        f.write(
            f"Hierarchy {hierarchy.bpb},{hierarchy.sbpb} {hierarchy.tpsb} a={hierarchy.al} {hierarchy.time}\n"
        )

        # Write Key Signature
        key_signatures = self.get_key_signature()
        f.writelines([f"{self.__convert_key_to_line__(x)}\n" for x in key_signatures])

        f.close()

    def __convert_key_to_line__(self, key) -> str:
        # KEY(key, not is_minor, key.time)
        tonality = "Maj" if key.is_major else "Min"
        time = key.time
        return f"Key {key.tonic} {tonality} {time}"
