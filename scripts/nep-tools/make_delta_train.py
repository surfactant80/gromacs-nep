#!/usr/bin/env python3
"""Subtract matched MM extxyz labels from reference extxyz labels."""

import argparse
import re
from pathlib import Path


def read_frames(path):
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        count = int(lines[i])
        yield count, lines[i + 1], lines[i + 2 : i + 2 + count]
        i += count + 2


def value(header, key):
    match = re.search(r"(?<!\S)" + re.escape(key) + r'=("([^"]*)"|\S+)', header)
    return None if match is None else (match.group(2) if match.group(2) is not None else match.group(1))


def replace_value(header, key, values):
    text = " ".join(f"{item:.16g}" for item in values)
    replacement = f'{key}="{text}"' if len(values) > 1 else f"{key}={text}"
    pattern = r"(?<!\S)" + re.escape(key) + r'=("([^"]*)"|\S+)'
    return re.sub(pattern, replacement, header, count=1)


def property_layout(header):
    tokens = value(header, "Properties").split(":")
    result = []
    offset = 0
    for i in range(0, len(tokens), 3):
        name, kind, width = tokens[i], tokens[i + 1], int(tokens[i + 2])
        result.append((name, kind, width, offset))
        offset += width
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_xyz")
    parser.add_argument("mm_xyz")
    parser.add_argument("output_xyz")
    args = parser.parse_args()
    reference = list(read_frames(args.reference_xyz))
    mm = list(read_frames(args.mm_xyz))
    if len(reference) != len(mm):
        raise SystemExit("Frame counts differ")

    output = []
    for frame_index, (ref_frame, mm_frame) in enumerate(zip(reference, mm)):
        count, header, atoms = ref_frame
        mm_count, mm_header, mm_atoms = mm_frame
        if count != mm_count:
            raise SystemExit(f"Atom counts differ in frame {frame_index}")
        layout = property_layout(header)
        if [(x[0], x[1], x[2]) for x in layout] != [
            (x[0], x[1], x[2]) for x in property_layout(mm_header)
        ]:
            raise SystemExit(f"Properties differ in frame {frame_index}")

        for key in ("energy", "free_energy"):
            ref_value, mm_value = value(header, key), value(mm_header, key)
            if ref_value is not None and mm_value is not None:
                header = replace_value(header, key, [float(ref_value) - float(mm_value)])
        for key in ("virial", "stress"):
            ref_value, mm_value = value(header, key), value(mm_header, key)
            if ref_value is not None and mm_value is not None:
                ref_numbers = list(map(float, ref_value.split()))
                mm_numbers = list(map(float, mm_value.split()))
                header = replace_value(
                    header, key, [a - b for a, b in zip(ref_numbers, mm_numbers)]
                )

        force_fields = [item for item in layout if item[0].lower() in ("force", "forces")]
        delta_atoms = []
        for atom_index, (ref_line, mm_line) in enumerate(zip(atoms, mm_atoms)):
            ref_tokens, mm_tokens = ref_line.split(), mm_line.split()
            if ref_tokens[0] != mm_tokens[0]:
                raise SystemExit(f"Species/order differs in frame {frame_index}, atom {atom_index}")
            for _, kind, width, offset in force_fields:
                if kind.upper() != "R":
                    raise SystemExit("Force property must be real")
                for component in range(width):
                    ref_tokens[offset + component] = (
                        f"{float(ref_tokens[offset + component]) - float(mm_tokens[offset + component]):.16g}"
                    )
            delta_atoms.append(" ".join(ref_tokens))
        output.extend([str(count), header, *delta_atoms])

    Path(args.output_xyz).write_text("\n".join(output) + "\n")
    print(f"Wrote {len(reference)} delta-labeled frames to {args.output_xyz}")


if __name__ == "__main__":
    main()
