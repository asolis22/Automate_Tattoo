from pathlib import Path


def build_big_square_points(base, width=3500, height=3500):
    """
    Build a larger square in pulse space around a known safe in-air base point.

    base = [J1, J2, J3, J4, J5, J6]
    width/height are pulse offsets, not mm.
    Start conservative and increase slowly.
    """
    j1, j2, j3, j4, j5, j6 = base

    return [
        [j1,         j2,          j3, j4, j5, j6],  # start
        [j1 + width, j2,          j3, j4, j5, j6],  # right
        [j1 + width, j2 + height, j3, j4, j5, j6],  # up-right
        [j1,         j2 + height, j3, j4, j5, j6],  # up-left
        [j1,         j2,          j3, j4, j5, j6],  # back home
    ]


def write_jbi_pulse_job(points, filename="BIGSQ.JBI", job_name="BIGSQ"):
    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(points)},0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")

    for i, p in enumerate(points):
        lines.append(f"C{i:05d}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}")

    lines.append("//INST")
    lines.append("///DATE 2026/04/01 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")

    # Approach first point slowly
    lines.append("MOVJ C00000 VJ=5.00")

    # Draw the square in the air
    lines.append("MOVL C00001 V=80.0")
    lines.append("MOVL C00002 V=80.0")
    lines.append("MOVL C00003 V=80.0")
    lines.append("MOVL C00004 V=80.0")

    lines.append("END")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"Saved JBI file: {filename}")


def main():
    # IMPORTANT:
    # Replace this base with a REAL safe pulse point from the robot's
    # current in-air pose or from a working exported job point.
    #
    # Example placeholder only:
    safe_base = [-3626, 25307, 0, 0, -64000, -407]

    # Bigger square than before
    points = build_big_square_points(
        base=safe_base,
        width=3500,
        height=3500,
    )

    write_jbi_pulse_job(points, filename="BIGSQ.JBI", job_name="BIGSQ")


if __name__ == "__main__":
    main()