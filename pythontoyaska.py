# generate_square_jbi.py
#
# Creates a simple Yaskawa INFORM .JBI file that traces
# a square in the air above a surface, repeating it 5 times.
#
# IMPORTANT:
# 1) Review the generated .JBI before loading it.
# 2) Confirm the selected USER FRAME and TOOL are correct.
# 3) Test at very low speed first.
# 4) Header/format details can vary a bit by controller / shop conventions,
#    so you may need to adjust the header if your controller expects a slightly
#    different export style.
#
# This version writes Cartesian positions in a simple readable layout.

from pathlib import Path


def mm(value: float) -> str:
    """Format mm values neatly for JBI text."""
    return f"{value:.3f}"


def deg(value: float) -> str:
    """Format angle values neatly for JBI text."""
    return f"{value:.4f}"


def build_positions(
    origin_x=300.0,
    origin_y=0.0,
    safe_z=100.0,
    square_size=50.0,
    rx=180.0,
    ry=0.0,
    rz=0.0,
):
    """
    Build positions for:
    - one approach point above the square start
    - the 4 square corners at constant Z
    - one exit point above the square start

    Assumes the path floats above the table/platform.
    """
    half = square_size / 2.0

    # Square centered at (origin_x, origin_y), all at safe_z
    p_approach = (origin_x - half, origin_y - half, safe_z + 30.0, rx, ry, rz)
    p1 = (origin_x - half, origin_y - half, safe_z, rx, ry, rz)
    p2 = (origin_x + half, origin_y - half, safe_z, rx, ry, rz)
    p3 = (origin_x + half, origin_y + half, safe_z, rx, ry, rz)
    p4 = (origin_x - half, origin_y + half, safe_z, rx, ry, rz)
    p_exit = (origin_x - half, origin_y - half, safe_z + 30.0, rx, ry, rz)

    return [p_approach, p1, p2, p3, p4, p_exit]


def generate_jbi(
    job_name="SQUARE5",
    user_frame=1,
    tool_no=1,
    repeat_count=5,
    movej_speed_percent=10,
    movl_speed_mm_min=300,
    origin_x=300.0,
    origin_y=0.0,
    safe_z=100.0,
    square_size=50.0,
):
    """
    Returns the contents of a simple .JBI file as a string.
    """

    positions = build_positions(
        origin_x=origin_x,
        origin_y=origin_y,
        safe_z=safe_z,
        square_size=square_size,
    )

    # Position labels
    names = ["P000", "P001", "P002", "P003", "P004", "P005"]

    lines = []

    # Header
    lines.append(f"/JOB")
    lines.append(f"//NAME {job_name}")
    lines.append(f"//POS")
    lines.append(f"///NPOS {len(positions)},0,0,0,0,0")
    lines.append(f"///TOOL {tool_no}")
    lines.append(f"///POSTYPE USER")
    lines.append(f"///USER {user_frame}")
    lines.append(f"///RECTAN")

    # Position block
    for label, pos in zip(names, positions):
        x, y, z, rx, ry, rz = pos
        lines.append(
            f"{label}="
            f"{mm(x)},{mm(y)},{mm(z)},"
            f"{deg(rx)},{deg(ry)},{deg(rz)}"
        )

    # Instruction block
    lines.append(f"//INST")
    lines.append(f"///DATE 2026/04/01 12:00")
    lines.append(f"///ATTR SC,RW")
    lines.append(f"///GROUP1 RB1")
    lines.append(f"NOP")
    lines.append(f"SETTOOL TL#{tool_no}")
    lines.append(f"SETUF UF#{user_frame}")

    # Move to approach point first
    lines.append(f"MOVJ {names[0]} VJ={movej_speed_percent}.00")

    for _ in range(repeat_count):
        lines.append(f"MOVL {names[1]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {names[2]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {names[3]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {names[4]} V={movl_speed_mm_min}.0")
        lines.append(f"MOVL {names[1]} V={movl_speed_mm_min}.0")

    # Leave safely
    lines.append(f"MOVL {names[5]} V={movl_speed_mm_min}.0")
    lines.append(f"END")

    return "\n".join(lines) + "\n"


def main():
    output_dir = Path(".")
    job_name = "SQUARE5"
    output_file = output_dir / f"{job_name}.JBI"

    jbi_text = generate_jbi(
        job_name=job_name,
        user_frame=1,          # Change to your taught user frame
        tool_no=1,             # Change to your taught tool number
        repeat_count=5,
        movej_speed_percent=10,  # very conservative
        movl_speed_mm_min=300,   # slow linear motion
        origin_x=300.0,          # adjust for your cell
        origin_y=0.0,
        safe_z=100.0,            # 100 mm above the surface
        square_size=50.0,        # 50 mm square
    )

    output_file.write_text(jbi_text, encoding="utf-8")
    print(f"Generated: {output_file.resolve()}")


if __name__ == "__main__":
    main()