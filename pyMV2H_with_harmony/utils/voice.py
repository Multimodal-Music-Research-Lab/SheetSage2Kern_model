import sys
from collections import OrderedDict
from typing import Optional

# from orderedset import OrderedSet
from pyMV2H_with_harmony.utils.comparators import cluster_comparator, note_comparator
from pyMV2H_with_harmony.utils.pojos import NOTE


class NoteCluster:
    def __init__(self, onset_time, offset_time):
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.notes = list()
        self.next_clusters = list()

    @property
    def key_string(self) -> str:
        def format_string(value) -> str:
            max_length = len(f"{sys.maxsize}")
            return f"{value:0{max_length}d}"

        return f"{format_string(self.onset_time)}_{format_string(self.offset_time)}"

    def __repr__(self):
        return self.key_string


class Voice:
    def __init__(self):
        self.__notes__ = list()
        self.__note_clusters__ = OrderedDict()
        self.__connection_created__ = False

    def create_connections(self):

        if self.__connection_created__:
            return

        # remove all previous connections
        for cluster in self.__note_clusters__.values():
            cluster.next_clusters = list()

        _cluster = self.__note_clusters__
        self.__note_clusters__ = OrderedDict(
            sorted(_cluster.items(), key=cluster_comparator())
        )

        ordered_notes = list(self.__notes__)
        ordered_notes.sort(key=note_comparator())
        self.__notes__ = ordered_notes
        # create new connections
        for base_key in self.__note_clusters__.keys():
            base_cluster = self.__note_clusters__[base_key]

            for next_key in self.__note_clusters__.keys():
                next_cluster = self.__note_clusters__[next_key]
                if next_key == base_key:
                    continue
                if next_cluster.onset_time == base_cluster.offset_time:
                    # Every note cluster which begins at this one's offset time.
                    base_cluster.next_clusters.append(next_cluster)
                elif next_cluster.onset_time > base_cluster.offset_time:
                    if (
                        len(base_cluster.next_clusters) == 0
                        or base_cluster.next_clusters[0].onset_time
                        == next_cluster.onset_time
                    ):
                        # All note clusters at the earliest time after this one's offset time, if no
                        # connections were added from rule
                        base_cluster.next_clusters.append(next_cluster)
                    else:
                        break
        self.__connection_created__ = True

    def get_cluster(self, note: NOTE) -> Optional[NoteCluster]:
        key = (NoteCluster(note.on, note.off_val)).key_string
        cluster = self.__note_clusters__[key]
        if cluster is not None and note in cluster.notes:
            return cluster
        return None

    def add_note(self, note: NOTE):
        self.__notes__.append(note)
        cluster = NoteCluster(note.on, note.off_val)
        key = cluster.key_string
        if key not in self.__note_clusters__:
            self.__note_clusters__[key] = cluster
        else:
            cluster = self.__note_clusters__[key]
        cluster.notes.append(note)
