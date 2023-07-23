import sys, os
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.join(ABSPATH, '../../'))

from babble.datasets import generate_tacolabels_from_text

in_text_path = sys.argv[1]
taco_lab_dir = sys.argv[2]
language=sys.argv[3]

# generate lab file from text_file and output to output_dir
sucess_labs = generate_tacolabels_from_text(
    in_text_path,
    taco_lab_dir,
    language=language
)
