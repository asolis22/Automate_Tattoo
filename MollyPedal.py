from gpiozero import OutputDevice

RELAY_PIN = 17  # GPIO17 (physical pin 11)

# Relay is ACTIVE LOW
relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=False)

print("Type:")
print("  1 = relay ON (pedal down)")
print("  0 = relay OFF (pedal up)")
print("  q = quit")

try:
    while True:
        cmd = input("> ").strip()

        if cmd == "1":
            print("Pedal DOWN → relay ON → circuit CLOSED")
            relay.on()

        elif cmd == "0":
            print("Pedal UP → relay OFF → circuit OPEN")
            relay.off()

        elif cmd == "q":
            break

        else:
            print("Use 1, 0, or q")

finally:
    print("Cleanup: relay OFF")
    relay.off()
    relay.close()