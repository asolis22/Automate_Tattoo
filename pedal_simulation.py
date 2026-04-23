from pynput import keyboard

class PedalController:
    def __init__(self, mode="sim"):
        self.mode = mode
        self.state = False  # False = OFF, True = ON

    def pedal_down(self):
        if not self.state:
            self.state = True
            if self.mode == "sim":
                print("[SIM] Pedal DOWN → circuit CLOSED (gun ON)")
            else:
                # Placeholder for real GPIO
                pass

    def pedal_up(self):
        if self.state:
            self.state = False
            if self.mode == "sim":
                print("[SIM] Pedal UP → circuit OPEN (gun OFF)")
            else:
                # Placeholder for real GPIO
                pass


controller = PedalController(mode="sim")

pressed = False

def on_press(key):
    global pressed
    try:
        if key.char == '1' and not pressed:
            pressed = True
            controller.pedal_down()
    except AttributeError:
        pass

def on_release(key):
    global pressed
    try:
        if key.char == '1':
            pressed = False
            controller.pedal_up()
    except AttributeError:
        pass

print("Press and hold '1' to simulate pedal. Ctrl+C to exit.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

    