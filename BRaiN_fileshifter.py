
##### moves files before rerun

import shutil
import glob


wav_input = glob.glob("PIPI/*.wav")
wav_input.extend(glob.glob("PIPY/*.wav"))
wav_input.extend(glob.glob("UNID/*.wav"))
wav_input.extend(glob.glob("SILENCE/*.wav"))
wav_input.extend(glob.glob("UNCLASS/*.wav"))

for wav in wav_input:
    shutil.move(wav, "DATA")
    