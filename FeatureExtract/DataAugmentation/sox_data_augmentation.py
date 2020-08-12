
import sox
import subprocess
import random
import os



volume = random.uniform(0.5,1.5)


file_path = './Data/'
target_file_path = './DataAugmentation/'
file_names = os.listdir(file_path)

for audio_index, audio_file in enumerate(file_names):
	input_filename = audio_file	
	output_filename = audio_file+'DataAug.wav'
	
	args = ['sox', '-v', str(volume), input_filename, ' ',output_filename]
	command = ' '.join(args)	

	process_handle = subprocess.Popen(
	command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)