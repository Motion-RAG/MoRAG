import os
import openai
import json
import sys
import numpy as np
import time

from openai import OpenAI

client = OpenAI(api_key="")

prompt = """The instructions for this task is to describe the below body part position and movements in a sentence using simple language.\n1) Torso \n2) Hands \n3) Legs \nHere are some examples of the question and answer pairs for this task: \nQuestion: Describe the below body parts position and movements involved in the action "a person squats down and puts their hands above their head" in a sentence using simple language. \nAnswer: \n1) Torso: The person bends their upper body forward, bringing their chest closer to their thighs while squatting down. \n2) Hands: The hands are raised above the head, reaching upward in a stretching motion. \n3) Legs: The person is squatting down, bending their knees and lowering their body toward the ground. \nQuestion: Describe the below body parts position and movements involved in the action "a person is swimming in the water" in a sentence using simple language. \nAnswer: \n1) Torso: The person's torso is floating horizontally in the water, moving smoothly from side to side as they swim. \n2) Hands: Their hands are pushing through the water, alternating in a circular motion, propelling them forward. \n3) Legs: The legs are kicking gently, helping to keep the body afloat and moving in a coordinated rhythm with the arms. \nQuestion: Describe the below body parts position and movements involved in the action [ACTION] in a sentence using simple language.\n1) Torso \n2) Hands \n3) Legs"""

def GPT_Completion(texts):
	response = client.completions.create(
	  model = "gpt-3.5-turbo-instruct",
	  prompt=texts,
	  max_tokens = 256
	)	
	gpt_texts = []
	for i in range(len(response.choices)):
		gpt_texts.append(response.choices[i].text)
	
	return gpt_texts	

all_texts = []
annotations = json.load(open('datasets/annotations.json'))

keys = list(annotations.keys())

for key in keys:
	# if(key[0]=="M"): #To avoid generation for mirrored text descriptions
		# continue
	dico = annotations[key]
	for lst in dico["annotations"]:
		all_texts.append(lst["text"])
all_texts = list(set(all_texts))

gpt_texts_dict = {}

for i in range((len(all_texts)//20)+1): # Doing it batch wise to avoid concurrent calls to api
	texts = []
	texts_batch = all_texts[i*20:(i+1)*20]
	if(len(texts_batch)==0):
		continue
	for j in texts_batch:
		texts.append(prompt.replace("[ACTION]",str(j)))
	
	gpt_batch = GPT_Completion(texts)
	
	for j in range(len(texts_batch)):
		gpt_texts_dict[texts_batch[j]]=gpt_batch[j]
	
	with open('gpt_texts.json', 'w') as fp:
		json.dump(gpt_texts_dict, fp, indent = 4)
		fp.close()

	time.sleep(5)
