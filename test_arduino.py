import serial
import time
 # test working
PORT = "COM3"  # change to your Arduino port

arduino = serial.Serial(PORT, 9600, timeout=1)
time.sleep(2)  # Arduino resets when Python connects

print("Arduino:", arduino.readline().decode(errors="ignore").strip())

print("Sending ALERT")
arduino.write(b"ALERT\n")
time.sleep(3)

print("Sending NORMAL")
arduino.write(b"NORMAL\n")
time.sleep(1)

arduino.close()