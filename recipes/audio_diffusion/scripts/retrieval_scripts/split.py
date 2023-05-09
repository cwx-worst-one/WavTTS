import argparse


def main(args):
    output_fws = []
    for i in range(args.parts):
        fname = f"{args.input_fname}.{i+1}"
        fw = open(fname, "w")
        output_fws.append(fw)

    i = 0
    with open(args.input_fname, "r") as f:
        for line in f:
            output_fws[i].write(f"{line}")
            i = (i + 1) % args.parts

    for fw in output_fws:
        fw.close()


if __name__ == "__main__":
    desc = "Split an input file into N parts"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("input_fname", help="Path to input file")
    parser.add_argument("parts", type=int, help="How many parts to split")
    args = parser.parse_args()
    assert args.parts > 0
    main(args)
