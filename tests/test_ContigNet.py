import os
import sys
import ContigNet
from click.testing import CliRunner
from ContigNet.__main__ import main

__all__ = ['ContigNet',]


def test_ContigNet():
    """Test ContigNet with default CPU mode"""
    runner = CliRunner()
    host_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/host_fasta")
    virus_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/virus_fasta")

    with runner.isolated_filesystem():
        result = runner.invoke(main, [
            '--host-dir', host_dir,
            '--virus-dir', virus_dir,
            '--output', 'result.csv',
            '--cpu'
        ])

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        assert os.path.exists('result.csv'), "Output file was not created"


def test_ContigNet_with_preview():
    """Test ContigNet with preview option"""
    runner = CliRunner()
    host_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/host_fasta")
    virus_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/virus_fasta")

    with runner.isolated_filesystem():
        result = runner.invoke(main, [
            '--host-dir', host_dir,
            '--virus-dir', virus_dir,
            '--output', 'result.csv',
            '--cpu',
            '--show-preview'
        ])

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        assert os.path.exists('result.csv'), "Output file was not created"
        assert "Top 10 Predictions" in result.output, "Preview table not shown"


def test_ContigNet_help():
    """Test that help message works"""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])

    assert result.exit_code == 0
    assert "ContigNet" in result.output
    assert "--host-dir" in result.output
    assert "--virus-dir" in result.output
