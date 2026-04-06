from .base import Base


class MidiConverter(Base):

    def run(self):
        from ..converter.midi_converter import MidiConverter as Converter

        input_file = self.options['<input_dir>']
        output_file = self.options['<output_dir>']
        converter = Converter(file=input_file, output=output_file)
        converter.convert_file()
