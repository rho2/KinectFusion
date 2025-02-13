from pathlib import Path
import sys

def main(target: Path):
    replaced = target.read_text().replace('extern "C" __constant__ GlobalParams_0', '__device__ GlobalParams_0')
    new_file = target.with_name(target.name[4:])
    new_file.write_text(replaced)

if __name__ == '__main__':
    main(Path(sys.argv[1]))
