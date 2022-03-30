# HEARTBEAT PHYSIONET DATASET
# sr 2000 Hz 
# 16bit
# -1 = NORMAL
#  1 = ABNORMAL
# 1 = good quality
# 0 = poor quality


import glob, os
from pydub import AudioSegment
import pydub.scipy_effects
import csv



os.chdir("./")

csvFile = open('export/Reference_total.csv', 'w', newline='')
writer = csv.writer(csvFile)
writer.writerow(['name', 'class_label'])


for direc in glob.iglob('/PHYSIONET_hb_dataset/**', recursive=True):
    if os.path.isdir(direc): # filter dirs
        print(direc)
        if direc == "/PHYSIONET_hb_dataset/" or direc == "/PHYSIONET_hb_dataset/export":
            continue
        else:
            csvFileRead = open(direc+'/REFERENCE-SQI.csv', newline='')
            reader = csv.reader(csvFileRead)

            for row in reader:
                file_name = row[0]
                label = row[1]
                quality = row[2]
                print("file corrente: ",file_name, " label: ",label, " quality: ",quality)
                if int(quality) == 1:
                    file_name_wav = file_name + '.wav'
                    audio = AudioSegment.from_wav(direc+'/'+file_name_wav)
                    slices = audio[::2000] #cut every 2 seconds
                    count = 0
                    for fetta in slices:
                        fetta = fetta.band_pass_filter(25, 400, order=12) 
                        wavename = file_name + '_' + str(count) + '.wav'
                        fetta.export('export/'+wavename, format="wav")
                        count += 1
                        writer.writerow([wavename, label])


csvFileRead.close()
csvFile.close()
