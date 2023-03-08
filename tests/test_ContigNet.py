import os
import sys
import ContigNet

__all__ = ['ContigNet',]


def test_ContigNet():
    host_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/host_fasta")
    virus_dir = os.path.join(os.path.dirname(sys.modules["ContigNet"].__file__), "demo/virus_fasta")
    assert os.system(f"python -m ContigNet --host_dir {host_dir} --virus_dir {virus_dir} --output result.csv") == 0
    assert os.system(f"python -m ContigNet --host_dir {host_dir} --virus_dir {virus_dir} --output result.csv --cpu") == 0
