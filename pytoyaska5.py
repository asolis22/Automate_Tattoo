def generate_square_job(filename="SQUARE.JBI"):
    # Square points (YOU CAN EDIT THESE)
    # Format: (X, Y, Z, Rx, Ry, Rz)
    points = [
        (0, 0, 0, 0, 0, 0),          # Start
        (20000, 0, 0, 0, 0, 0),      # Right
        (20000, 20000, 0, 0, 0, 0),  # Up
        (0, 20000, 0, 0, 0, 0),      # Left
        (0, 0, 0, 0, 0, 0)           # Back to start
    ]

    with open(filename, "w") as f:
        # HEADER
        f.write("/JOB\n")
        f.write("//NAME SQUARE\n")
        f.write("//POS\n")
        f.write(f"///NPOS {len(points)},0,0,0,0,0\n")
        f.write("///TOOL 0\n")
        f.write("///POSTYPE PULSE\n")
        f.write("///PULSE\n")

        # WRITE POINTS
        for i, p in enumerate(points):
            label = f"C{i:05d}"
            f.write(f"{label}={p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}\n")

        # INSTRUCTIONS
        f.write("//INST\n")
        f.write("///DATE 2026/04/01 12:00\n")
        f.write("///ATTR SC,RW\n")
        f.write("///GROUP1 RB1\n")
        f.write("NOP\n")

        # Move to first point
        f.write("MOVJ C00000 VJ=25.00\n")

        # Draw square
        for i in range(1, len(points)):
            f.write(f"MOVL C{i:05d} V=200.0\n")

        f.write("END\n")

    print(f"✅ Job file '{filename}' created!")


# Run it
generate_square_job()