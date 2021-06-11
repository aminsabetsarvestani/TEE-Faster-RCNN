import imageio
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path
import csv

sampling_margin ={'park':[250,600],'lakeSide':[1000,6500],'library':[600,4900], 'diningRoom':[700,3700],
					'corridor':[500,5400], 'turbulence0':[1000,2999],'turbulence1':[1200,2599],
					'turbulence2':[500,2499],'turbulence3':[800,1499],'peopleInShade':[250,1199],
					'cubicle':[1100,7400],'copyMachine':[500,3400],'busStation':[300,1250],
					'backdoor':[400,2000],'streetCornerAtNight':[800,2999],'turnpike_0_5fps':[800,1149],
					'winterDriveway':[1000,2500],'sofa':[500,2750],'overpass':[1000,3000],'fountain02':[500,1499],
					'fountain01':[400,1184],'fall':[1000,4000],'canoe':[800,1189],'traffic':[900,1570],
					'badminton':[800,1150],'pedestrians':[300,1099],'office':[570,2050],'highway':[470,1700],
					'PETS2006':[300,1200],'wetSnow':[500,1999],'snowFall':[800,3649],'skating':[800,2349],
					'blizzard':[900,3949],'zoomInZoomOut':[500,818],'intermittentPan':[1200,2349],'continuousPan':[600,1149]
					
}


iteration = 20
IoU_cutoff = 0.7
tran_test_proportion = 1
directories = os.walk('.')
video_list = list()
for dirpath, dirs, files in directories:
	if "groundtruth" in dirpath:
		video_list.append(dirpath+"/*.png")
#video_list = ["badWeather/snowFall/groundtruth/*.png"]

global_min_video_lengh = 100000
extracted_frames = list()
for vid in video_list:
	image_list = glob.glob(vid)
	image_list.sort()
	picked_video = image_list[0].split('/')[2]
	if picked_video in sampling_margin.keys():
		sampling_range = sampling_margin[picked_video]
		sampling_range
		Unchanged_frames = list()
		Changed_frames = list()
		for i in range(iteration):
			for first_frame in range(sampling_range[0],sampling_range[1]):
				#Calculate the IoU between objects.
				end_sampling = first_frame + 50
				if end_sampling>sampling_range[1]-1:
					end_sampling = sampling_range[1]-1
				im1 = imageio.imread(image_list[first_frame])
				second_frame = random.randint(first_frame,end_sampling)
				data1 = np.array(im1)
				data1 = np.where(data1==170, 0, data1)
				data1 = np.where(data1==85, 0, data1)
				data1 = np.where(data1==255, 1, data1)

				im2 = imageio.imread(image_list[second_frame])
				data2 = np.array(im2)
				data2 = np.where(data2==170, 0, data2)
				data2 = np.where(data2==85, 0, data2)
				data2 = np.where(data2==255, 1, data2)
				

				diff = data2 - data1
				diff = np.where(diff!=0, 1, 0)

				added = data2 + data1
				added = np.where(added!=0, 1, 0)
				total_data = np.where(added!=0, 1, 1)

				counter1 = np.sum(diff)
				counter2 = np.sum(added)
				info_propr = counter2/np.sum(total_data)

				if counter2 == 0:
					counter2 =1
				IoU = 1- counter1/counter2
				if info_propr<0.002:
					IoU = 1		
	
				splited_1 = image_list[first_frame].split('/')
				splited_2 = image_list[second_frame].split('/')
				numer1 = splited_1[4].split('.')[0].split('gt')[1]
				numer2 = splited_2[4].split('.')[0].split('gt')[1]
				name1 = splited_1[2]+ '_in' + numer1
				name2 = splited_2[2]+ '_in' + numer2
				if IoU < IoU_cutoff:
					IoU = 0
					Changed_frames.append([name1,name2,IoU])
				else:
					IoU = 1
					Unchanged_frames.append([name1,name2,IoU])
		
			random.shuffle(Changed_frames)
			random.shuffle(Unchanged_frames)
			min_length = min(len(Changed_frames),len(Unchanged_frames))
			Changed_frames = Changed_frames[0:min_length]
			Unchanged_frames = Unchanged_frames[0:min_length]
			extracted_frames.append(Changed_frames)
			extracted_frames.append(Unchanged_frames)
			#minimum video lenght
			if min_length < global_min_video_lengh:
				global_min_video_lengh = min_length
final_video_samples = list()
print('Global Min',global_min_video_lengh)
for samples in extracted_frames:
	samples = samples[0:global_min_video_lengh]
	final_video_samples += samples
random.shuffle(final_video_samples)
print('len extracted_frames', len(final_video_samples))
split_point = round((len(final_video_samples)/10))*tran_test_proportion
print('----Splitting point-----',split_point)
test_set = final_video_samples[0:split_point]
training_set = final_video_samples[split_point:(len(final_video_samples))]

print('len train',len(training_set))
print('len test',len(test_set))

remove_file_training = len(training_set)%128
remove_file_test = len(test_set)%128

print('remove train',remove_file_training)
print('remove test',remove_file_test)


training_set = training_set[0:(len(training_set)-remove_file_training)]
test_set = test_set[0:(len(test_set)-remove_file_test)]

print('--Testset---->', len(test_set))
print('--Traintset---->', len(training_set))


with open("train.csv","w") as f:
	writer = csv.writer(f)
	writer.writerows(training_set)

with open("eval.csv","w") as f:
	writer = csv.writer(f)
	writer.writerows(test_set)
	
