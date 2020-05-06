import numpy as np
import serial
import time

waitTime = 0.1

# frequency: 256~65535
songTable = [
    [261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    256, 256, 256, 256, 256, 256, 256, 256],

    [440, 440, 660, 660, 660, 494, 524, 494, 440,
    660, 784, 880, 784, 660, 740, 588, 660, 
    880, 880, 880, 784, 660, 660, 588, 524, 494,
    440, 660, 588, 524, 494, 440, 392, 440,
    256, 256, 256, 256, 256, 256,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256],

    [330, 330, 349, 392, 392, 349, 330, 294, 262, 262, 294, 330, 330, 294, 294,
    330, 330, 349, 392, 392, 349, 330, 294, 262, 262, 294, 330, 294, 262, 262,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
]
length = [
    [257, 257, 257, 257, 257 ,257 ,258,
    257, 257, 257, 257, 257 ,257 ,258,
    257, 257, 257, 257, 257 ,257 ,258,
    257, 257, 257, 257, 257 ,257 ,258,
    257, 257, 257, 257, 257 ,257 ,258,
    257, 257, 257, 257, 257 ,257 ,258,
    256, 256, 256, 256, 256, 256, 256, 256],

    [260, 258, 258, 259, 257, 259, 257, 258, 266,
    258, 258, 260, 258, 258, 258, 258, 271,
    258, 260, 258, 260, 258, 258, 258, 258, 266,
    260, 258, 260, 258, 258, 258, 258, 271,
    256, 256, 256, 256, 256, 256,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256],

    [257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 258,
    257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 258,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
]

# output formatter
formatter = lambda x: "%d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("song\n0: Twinkle, Twinkle, Little Star\n1: Scarborough Fair\n2: Ode to Joy")
sel = input("Select a song: ")
sel = int(sel)
#print("Sending signal ...")
#print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
time.sleep(waitTime)
for data in songTable[sel]:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
for data in length[sel]:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Sended")