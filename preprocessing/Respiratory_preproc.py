# RESPIRATORY ICHBI DATASET
# sr 4-10-44.1 kHz 
# 16bit


import glob, os
from pydub import AudioSegment
import pydub.scipy_effects
import csv


os.chdir("./")

csvFile = open('./export/Reference.csv', 'w', newline='')
writer = csv.writer(csvFile)
writer.writerow(['name', 'class_label']) 


for file in glob.glob("*.txt"):
    print("file corrente: ",file)
    riga = 0
    f = open(file, "r") 

    for x in f:
        splittedX = x.split()
        start = splittedX[0]
        stop = splittedX[1]
        crackles = splittedX[2] #presence=1, absence=0
        wheezes =splittedX[3] #presence=1, absence=0
        filename = './export/' + file[:-4] + '_' + str(riga)+ '.txt'

        t1 = float(start) * 1000 #Works in milliseconds
        t2 = float(stop) * 1000
        wavename = file[:-4] + '.wav'
        t1=round(t1,1)
        t2=round(t2,1)
        newAudio = AudioSegment.from_wav(wavename)
        newAudio = newAudio[t1:t2]
        wavename = filename[:-4] + '.wav'
        newAudio = newAudio.band_pass_filter(100, 1800, order=12) 
        newAudio.export(wavename, format="wav", parameters=["-ar", "4000"])

        c = int(crackles)
        w = int(wheezes)
        if c==0 and w==0:            # 0-0
            writer.writerow([wavename[9:], "0"]) #normal

        if c==0 and w==1:            # 0-1
            writer.writerow([wavename[9:], "1"]) #wheezes

        if c==1 and w==0:            # 1-0
            writer.writerow([wavename[9:], "2"]) #crackles

        if c==1 and w==1:            # 1-1
            writer.writerow([wavename[9:], "3"]) #crackles & wheezes
 

        riga += 1

    riga = 0
    f.close()
csvFile.close()
