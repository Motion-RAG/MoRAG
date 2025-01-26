import os
import json

gpt_texts = json.load(open('gpt_texts.json'))
keys = list(gpt_texts.keys())

torso = {}
hands = {}
legs = {}

issue_json = {}
issue_keys = []

for i in range(len(keys)):
	key = keys[i]
	text = gpt_texts[key]
	try:
		torso[key]=(text.split("1) Torso:")[1].split("2) Hands:")[0].strip())
		hands[key]=(text.split("2) Hands:")[1].split("3) Legs:")[0].strip())
		legs[key]=(text.split("3) Legs:")[1].strip())
	except:
		issue_json[key]=text
		gpt_texts.pop(key)

# print(len(gpt_texts))
# print(len(issue_json))

with open('torso.json', 'w') as fp:
	json.dump(torso, fp, indent = 4)
	fp.close()

with open('hands.json', 'w') as fp:
	json.dump(hands, fp, indent = 4)
	fp.close()

with open('legs.json', 'w') as fp:
	json.dump(legs, fp, indent = 4)
	fp.close()

with open('issue_gpt.json', 'w') as fp:
	json.dump(issue_json, fp, indent = 4)
	fp.close()	
