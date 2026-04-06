""" "
pyMV2H
Usage:
  pyMV2H midi_converter -i <input_dir> -o <output_dir>
  pyMV2H compare_files -g <reference_file> -t <transcription_file> [-a] [-p]
  pyMV2H -h | --help
  pyMV2H --version
Options:
  -h --help                         Show this screen.
  --version                         Show version.
  -o --output                       The output file
  -t                                The transcription file
  -g                                The reference file
  -a                                Align both files
  -p                                Set the DTW insertion and deletion penalties
Examples:
  pyMV2H compare_files -g <reference_file> -t <transcription_file> -a -p 2.35
  pyMV2H midi_converter -i <midi_file> -o <output_dir>
Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/lucasmpaim/pyMV2H
"""

from inspect import getmembers, isclass

from docopt import docopt

from . import __version__


def main():
    """Main CLI entrypoint."""
    import pyMV2H_with_harmony.commands

    options = docopt(__doc__, version=__version__)
    # Here we'll try to dynamically match the command the user is trying to run
    # with a pre-defined command class we've already created.
    for k, v in options.items():
        if hasattr(pyMV2H_with_harmony.commands, k) and v:
            module = getattr(pyMV2H_with_harmony.commands, k)
            pyMV2H_with_harmony.commands = getmembers(module, isclass)
            command = [
                command[1]
                for command in pyMV2H_with_harmony.commands
                if command[0] != "Base"
            ][0]
            command = command(options)
            command.run()
