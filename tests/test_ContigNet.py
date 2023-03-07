import os

def test_ContigNet():
    assert os.system("python -m ContigNet --host_dir demo/host_fasta --virus_dir demo/virus_fasta --output result.csv") == 0
    # assert os.system("python -m ContigNet --host_dir demo/host_fasta --virus_dir demo/virus_fasta --output result.csv --cpu") == 0
