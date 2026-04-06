"""Tests for our main skele CLI module."""


from subprocess import PIPE, Popen as popen
from unittest import TestCase

from pyMV2H_with_harmony import __version__ as VERSION


class TestHelp(TestCase):
    def test_returns_usage_information(self):
        output = popen(['pyMV2H', '-h'], stdout=PIPE).communicate()[0]
        self.assertTrue('Usage:'.encode() in output)

        output = popen(['pyMV2H', '--help'], stdout=PIPE).communicate()[0]
        self.assertTrue('Usage:'.encode() in output)


class TestVersion(TestCase):
    def test_returns_version_information(self):
        output = popen(['pyMV2H', '--version'], stdout=PIPE).communicate()[0]
        self.assertEqual(output.strip(), VERSION.encode())
