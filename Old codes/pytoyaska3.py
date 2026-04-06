from pathlib import Path


def generate_jbi(job_name="SQUARE5"):
    lines = []

    # HEADER
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append("///NPOS 5,0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    # BASE SAFE POINT (COPY ONE FROM YOUR REAL FILE)
    base = [-3000, 25000, 0, 0, -64000, -400]

    # Small SAFE offsets (square shape)
    offsets = [
        (0, 0),
        (2000, 0),
        (2000, 2000),
        (0, 2000),
        (0, 0)
    ]

    for i, (dx, dy) in enumerate(offsets):
        p = base.copy()
        p[0] += dx
        p[1] += dy

        lines.append(
            f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}"
        )

    # INSTRUCTIONS
    lines.append("//INST")
    lines.append("///DATE 2026/04/01 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")

    lines.append("NOP")

    # Move to start
    lines.append("MOVJ C00000 VJ=5.00")

    # Repeat square 5 times
    for _ in range(5):
        lines.append("MOVL C00001 V=100.0")
        lines.append("MOVL C00002 V=100.0")
        lines.append("MOVL C00003 V=100.0")
        lines.append("MOVL C00004 V=100.0")

    lines.append("END")

    return "\n".join(lines)


def main():
    file = Path("SQUARE5.JBI")
    file.write_text(generate_jbi())
    print("Generated SQUARE5.JBI")


if __name__ == "__main__":
    main()