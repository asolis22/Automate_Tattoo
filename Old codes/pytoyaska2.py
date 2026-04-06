from pathlib import Path


def mm(value: float) -> str:
    return f"{value:.3f}"


def deg(value: float) -> str:
    return f"{value:.4f}"


def build_square_points(
    center_x=300.0,
    center_y=0.0,
    trace_z=100.0,
    lift_z=130.0,
    square_size=50.0,
    rx=180.0,
    ry=0.0,
    rz=0.0,
):
    half = square_size / 2.0

    # Approach / trace / exit
    c0 = (center_x - half, center_y - half, lift_z,  rx, ry, rz)  # approach
    c1 = (center_x - half, center_y - half, trace_z, rx, ry, rz)  # start
    c2 = (center_x + half, center_y - half, trace_z, rx, ry, rz)
    c3 = (center_x + half, center_y + half, trace_z, rx, ry, rz)
    c4 = (center_x - half, center_y + half, trace_z, rx, ry, rz)
    c5 = (center_x - half, center_y - half, lift_z,  rx, ry, rz)  # exit

    return [c0, c1, c2, c3, c4, c5]


def generate_jbi(
    job_name="SQUARE5",
    tool_no=0,
    repeat_count=5,
    movej_speed_percent=5,
    movl_speed_mm_min=100,
    center_x=300.0,
    center_y=0.0,
    trace_z=100.0,
    lift_z=130.0,
    square_size=50.0,
):
    pts = build_square_points(
        center_x=center_x,
        center_y=center_y,
        trace_z=trace_z,
        lift_z=lift_z,
        square_size=square_size,
    )

    labels = [f"C{i:05d}" for i in range(len(pts))]

    lines = []
    lines.append("/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append("//POS")
    lines.append(f"///NPOS {len(pts)},0,0,0,0,0")
    lines.append(f"///TOOL {tool_no}")
    lines.append("///POSTYPE BASE")
    lines.append("///RECTAN")

    for label, pt in zip(labels, pts):
        x, y, z, rx, ry, rz = pt
        lines.append(
            f"{label}={mm(x)},{mm(y)},{mm(z)},{deg(rx)},{deg(ry)},{deg(rz)}"
        )

    lines.append("//INST")
    lines.append("///DATE 2026/04/01 12:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.append(f"SETTOOL TL#{tool_no}")
    lines.append(f"MOVJ {labels[0]} VJ={movej_speed_percent}.00")
    lines.append(f"MOVL {labels[1]} V={movl_speed_mm_min}.0")

    for _ in range(repeat_count):
        lines.append(f"MOVL {labels[2]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {labels[3]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {labels[4]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {labels[1]} V={movl_speed_mm_min}.0")

    lines.append(f"MOVL {labels[5]} V={movl_speed_mm_min}.0")
    lines.append("END")

    return "\n".join(lines) + "\n"


def main():
    job_name = "SQUARE5"
    output_file = Path(f"{job_name}.JBI")

    jbi_text = generate_jbi(
        job_name=job_name,
        tool_no=0,              # use 0 unless your setup needs a different tool
        repeat_count=5,
        movej_speed_percent=5,  # very slow/safe
        movl_speed_mm_min=100,  # very slow/safe
        center_x=300.0,         # adjust later if needed
        center_y=0.0,
        trace_z=100.0,          # 100 mm above surface
        lift_z=130.0,           # lifted approach/exit
        square_size=50.0,
    )

    output_file.write_text(jbi_text, encoding="utf-8")
    print(f"Generated: {output_file.resolve()}")


if __name__ == "__main__":
    main()