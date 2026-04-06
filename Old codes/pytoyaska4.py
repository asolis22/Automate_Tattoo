from pathlib import Path


def generate_jbi():
    lines = []

    lines.append("/JOB")
    lines.append("//NAME SQUARE5")
    lines.append("//POS")
    lines.append("///NPOS 5,0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    # EXACT safe base point from your file style
    base = [-3626, 25307, 0, 0, -64000, -407]

    # small SAFE offsets
    offsets = [
        (0, 0),
        (1000, 0),
        (1000, 1000),
        (0, 1000),
        (0, 0)
    ]

    for i, (dx, dy) in enumerate(offsets):
        p0 = base[0] + dx
        p1 = base[1] + dy
        p2 = base[2]
        p3 = base[3]
        p4 = base[4]
        p5 = base[5]

        lines.append(f"C{i:05d}={p0},{p1},{p2},{p3},{p4},{p5}")

    lines.append("//INST")
    lines.append("///DATE 2026/01/21 16:46")  # match your format EXACTLY
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")

    # EXACT motion format like your working job
    lines.append("MOVJ C00000 VJ=25.00")
    lines.append("MOVL C00001 V=375.0")
    lines.append("MOVJ C00002 VJ=25.00")
    lines.append("MOVL C00003 V=375.0")
    lines.append("MOVJ C00004 VJ=25.00")

    lines.append("END")

    return "\n".join(lines)


def main():
    file = Path("SQUARE5.JBI")
    file.write_text(generate_jbi(), encoding="utf-8")
    print("Generated SQUARE5.JBI")


if __name__ == "__main__":
    main()