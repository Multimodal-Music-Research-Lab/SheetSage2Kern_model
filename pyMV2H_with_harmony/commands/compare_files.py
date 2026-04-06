from ..metrics.mv2h import mv2h
from ..utils import algorithm_config
from ..utils.align_files import align_files
from .base import Base


class CompareFilesCommand(Base):
    def run(self):
        from pyMV2H_with_harmony.utils.music import Music

        reference_file = Music.from_file(self.options["<reference_file>"])
        transcription_file = Music.from_file(self.options["<transcription_file>"])
        reference_file.read_if_needed()
        transcription_file.read_if_needed()

        if self.options["-p"]:
            algorithm_config.NON_ALIGNMENT_PENALTY = float(self.options["-p"])

        if self.options["-a"]:
            _, mv2h_metric = align_files(reference_file, transcription_file)
            print(mv2h_metric)
        else:
            print(mv2h(reference_file, transcription_file))
