import numpy as np
import torch
import joblib

from openai import OpenAI

from demo.model import TMR_text_encoder
from demo.load import load_unit_embeddings, load_splits, load_json
from sinc.tools.easyconvert import matrix_to, axis_angle_to
from sinc.transforms.smpl import RotTransDatastruct

import pickle
import argparse
import os
import datetime
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from tqdm import tqdm

import clip
import time

from rots_to_smpl_conversion.rots_to_smpl import convert_rots_to_h3d

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

openai_api_key = "API-KEY"
client = OpenAI(api_key=openai_api_key)
prompt = """The instructions for this task is to describe the below body part position and movements in a sentence using simple language.\n1) Torso \n2) Hands \n3) Legs \nHere are some examples of the question and answer pairs for this task: \nQuestion: Describe the below body parts position and movements involved in the action "a person squats down and puts their hands above their head" in a sentence using simple language. \nAnswer: \n1) Torso: The person bends their upper body forward, bringing their chest closer to their thighs while squatting down. \n2) Hands: The hands are raised above the head, reaching upward in a stretching motion. \n3) Legs: The person is squatting down, bending their knees and lowering their body toward the ground. \nQuestion: Describe the below body parts position and movements involved in the action "a person is swimming in the water" in a sentence using simple language. \nAnswer: \n1) Torso: The person's torso is floating horizontally in the water, moving smoothly from side to side as they swim. \n2) Hands: Their hands are pushing through the water, alternating in a circular motion, propelling them forward. \n3) Legs: The legs are kicking gently, helping to keep the body afloat and moving in a coordinated rhythm with the arms. \nQuestion: Describe the below body parts position and movements involved in the action [ACTION] in a sentence using simple language.\n1) Torso \n2) Hands \n3) Legs"""

crop_size = 196
DATASET = "humanml3d"
assert DATASET == "humanml3d"

amass_tar_path = "amass_data/amass.pth.tar"

parser = argparse.ArgumentParser()
parser.add_argument("--text", default="A person is walking forward.")
parser.add_argument("--nretrieval", default="1")
args = parser.parse_args()
query_text = args.text

TORSO_PATH = f"MoRAG_Torso/torso_outputs/humanml3d_guoh3dfeats"
HANDS_PATH = f"MoRAG_Hands/hands_outputs/humanml3d_guoh3dfeats"
LEGS_PATH = f"MoRAG_Legs/legs_outputs/humanml3d_guoh3dfeats"

splits_choice = "all" #["Unseen", "all"]
text_category = "existing" #["existing", "zero-shot"]

mapper = {'ACCAD': 'ACCAD',
 'BioMotionLab_NTroje': 'BMLrub',
 'CMU': 'CMU',
 'BMLmovi': 'BMLmovi',
 'EKUT': 'EKUT',
 'DFaust_67': 'DFaust67',
 'HumanEva': 'HumanEva',
 'Eyes_Japan_Dataset': 'EyesJapanDataset',
 'KIT': 'KIT',
 'MPI_HDM05': 'MPIHDM05',
 'MPI_Limits': 'MPILimits',
 'MPI_mosh': 'MPImosh',
 'SFU': 'SFU',
 'SSM_synced': 'SSMsynced',
 'TCD_handMocap': 'TCDhandMocap',
 'TotalCapture': 'TotalCapture',
 'Transitions_mocap': 'Transitionsmocap',
 'DanceDB': 'DanceDB',
 'BMLhandball': 'BMLhandball'}

smplh_joints = [
	"pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
	"spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
	"neck", "left_collar", "right_collar", "head", "left_shoulder",
	"right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
	"left_index1", "left_index2", "left_index3", "left_middle1",
	"left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
	"left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
	"left_thumb2", "left_thumb3", "right_index1", "right_index2",
	"right_index3", "right_middle1", "right_middle2", "right_middle3",
	"right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
	"right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
	"right_thumb3", "nose", "right_eye", "left_eye", "right_ear", "left_ear",
	"left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
	"right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
	"left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
	"right_ring", "right_pinky"
]

smpl_bps = {
	"torso" : ['spine1', 'spine2', 'spine3', 'neck', 'head'],
	"hands": ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist',
	"left_index1", "left_index2", "left_index3", "left_middle1",
	"left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
	"left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
	"left_thumb2", "left_thumb3", "right_index1", "right_index2",
	"right_index3", "right_middle1", "right_middle2", "right_middle3",
	"right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
	"right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
	"right_thumb3"],
	"legs" : ['left_hip', 'left_knee', 'left_ankle', 'left_foot', 'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'pelvis']
}

torso_joints = [smplh_joints.index(x) for x in smpl_bps["torso"]]
hands_joints = [smplh_joints.index(x) for x in smpl_bps["hands"]]
legs_joints = [smplh_joints.index(x) for x in smpl_bps["legs"]]

def read_split(split):
	split_file = os.path.join("MoRAG_Hands/datasets/annotations/", "splits", split + ".txt")
	id_list = []
	with cs.open(split_file, "r") as f:
		for line in f.readlines():
			id_list.append(line.strip())
	return id_list

def load_annotations(name="annotations.json"):
	json_path = os.path.join("MoRAG_Hands/datasets/annotations/", name)
	with open(json_path, "rb") as ff:
		return orjson.loads(ff.read())

def amass_name_getter(fname):
	amass2babel_name = ""
	fname_splits = fname.split("/")
	for i in range(1,len(fname_splits)):
		if(i==1):
			amass2babel_name+=mapper[fname_splits[i]]
		amass2babel_name+="/"
		amass2babel_name+=fname_splits[i]
	return amass2babel_name

def encode_text(text):
	with torch.no_grad():
		text = clip.tokenize([text], truncate=True).to(device)
		x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

		x = x + clip_model.positional_embedding.type(clip_model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = clip_model.transformer(x)
		x = clip_model.ln_final(x).type(clip_model.dtype)

	# B, T, D
	xf_out = x.permute(1, 0, 2)
	return xf_out

def smpl_data_to_matrix_and_trans(data, nohands=True):
	trans = data['trans']
	nframes = len(trans)
	try:
		axis_angle_poses = data['poses']
		axis_angle_poses = data['poses'].reshape(nframes, -1, 3)
	except:
		breakpoint()

	if nohands:
		axis_angle_poses = axis_angle_poses[:, :22]

	matrix_poses = axis_angle_to("matrix", axis_angle_poses)

	return RotTransDatastruct(rots=matrix_poses, trans=trans)

def GPT_Completion(texts):
	response = client.completions.create(
	  model = "gpt-3.5-turbo-instruct",
	  prompt=texts,
	  max_tokens = 256
	)   
	gpt3_texts = []
	for i in range(len(response.choices)):
		gpt3_texts.append(response.choices[i].text)
	
	return gpt3_texts

def get_rots_trans(data,frames):
	poses = torch.from_numpy(data['poses'][frames]).float()
	trans = torch.from_numpy(data['trans'][frames]).float()

	smpl_data = {
		"poses": poses,
		"trans": trans
	}

	smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=False)
	return smpl_data.rots, smpl_data.trans

def humanml3d_keyid_to_babel_rendered_url(keyid):
	# Don't show the mirrored version of HumanMl3D
	if "M" in keyid:
		return None

	dico = h3d_index[keyid]
	path = dico["path"]

	# HumanAct12 motions are not rendered online
	# so we skip them for now
	if "humanact12" in path:
		return None

	# This motion is not rendered in BABEL
	# so we skip them for now
	if path not in amass_to_babel:
		return None

	babel_id = amass_to_babel[path].zfill(6)
	url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"

	# For the demo, we retrieve from the first annotation only
	ann = dico["annotations"][0]
	start = ann["start"]
	end = ann["end"]
	text = ann["text"]

	data = {
		"url": url,
		"start": start,
		"end": end,
		"text": text,
		"keyid": keyid,
		"babel_id": babel_id,
		"path": path,
	}

	return data

def merge(torso_datas, hands_datas, legs_datas, text):
	merged_rots = {}
	merged_trans = {}

	torso_rots = {}
	torso_trans = {}
	hands_rots = {}
	hands_trans = {}
	legs_rots = {}
	legs_trans = {}

	for rank in range(len(torso_datas)):
		torso_fname = torso_datas[rank]['path']
		hands_fname = hands_datas[rank]['path']
		legs_fname = legs_datas[rank]['path']

		torso_data = amass_data[amass_name_idx_map[torso_fname]]
		hands_data = amass_data[amass_name_idx_map[hands_fname]]
		legs_data = amass_data[amass_name_idx_map[legs_fname]]

		torso_frames = len(torso_data['poses'])
		hands_frames = len(hands_data['poses'])
		legs_frames = len(legs_data['poses'])
		frames = min([torso_frames,hands_frames,legs_frames])
		frames = np.arange(0,frames)

		motion_data_rots = {} 
		motion_data_trans = {} 

		motion_data_rots["torso"], motion_data_trans["torso"] = get_rots_trans(torso_data, frames)
		motion_data_rots["hands"], motion_data_trans["hands"] = get_rots_trans(hands_data, frames)
		motion_data_rots["legs"], motion_data_trans["legs"] = get_rots_trans(legs_data, frames) 

		compositioned_rots = torch.zeros_like(motion_data_rots["torso"])
		compositioned_rots[:, torso_joints] = motion_data_rots["torso"][:, torso_joints]
		compositioned_rots[:, hands_joints] = motion_data_rots["hands"][:, hands_joints]
		compositioned_rots[:, legs_joints] = motion_data_rots["legs"][:, legs_joints]
		compositioned_trans = motion_data_trans["legs"]

		merged_rots[rank] = compositioned_rots
		merged_trans[rank] = compositioned_trans

		# BELOW ARE PART-SPECIFIC RETRIEVAL DATA (Can be used for analysis purpose)
		# torso_rots[rank] = motion_data_rots["torso"]
		# torso_trans[rank] = motion_data_trans["torso"]

		# hands_rots[rank] = motion_data_rots["hands"]
		# hands_trans[rank] = motion_data_trans["hands"]

		# legs_rots[rank] = motion_data_rots["legs"]
		# legs_trans[rank] = motion_data_trans["legs"]				

	return merged_rots, merged_trans

	# BELOW DICT IS AS PER REMODIFFUSE RETRIEVAL FORMAT - 263 dimension format (Can be used for training remodiffuse)

	# merged_h3d = convert_rots_to_h3d(merged_rots[0],merged_trans[0])
	# length = len(merged_h3d)
	# if length >= crop_size:
	# 	idx = random.randint(0, length - crop_size)
	# 	merged_h3d = merged_h3d[idx: idx + crop_size]
	# 	length = crop_size
	# else:
	# 	padding_length = crop_size - length
	# 	D = merged_h3d.shape[1:]
	# 	padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
	# 	merged_h3d = np.concatenate([merged_h3d, padding_zeros], axis=0)

	# assert len(merged_h3d) == crop_size

	# torso_clip_feats = encode_text(torso_datas[0]['text'])
	# hands_clip_feats = encode_text(hands_datas[0]['text'])
	# legs_clip_feats = encode_text(legs_datas[0]['text'])

	# return_dict = {}
	# return_dict["captions"]=text
	# return_dict["motions"]=merged_h3d
	# return_dict["m_lengths"]=length
	# return_dict["clip_seq_features"]={}
	# return_dict["clip_seq_features"]["torso"]=torso_clip_feats
	# return_dict["clip_seq_features"]["hands"]=hands_clip_feats
	# return_dict["clip_seq_features"]["legs"]=legs_clip_feats
	# return_dict["retrieved_keyids"]={}
	# return_dict["retrieved_keyids"]["torso"]=torso_datas[0]['keyid']
	# return_dict["retrieved_keyids"]["hands"]=hands_datas[0]['keyid']
	# return_dict["retrieved_keyids"]["legs"]=legs_datas[0]['keyid']
	# return_dict["retrieved_text"]={}
	# return_dict["retrieved_text"]["torso"]=torso_datas[0]['text']
	# return_dict["retrieved_text"]["hands"]=hands_datas[0]['text']
	# return_dict["retrieved_text"]["legs"]=legs_datas[0]['text']	

	# return return_dict

def retrieve_helper(
	model,
	unit_motion_embs,
	all_keyids,
	text,
	keyids_index,
	index_keyids,    
	split="test",
	nmax=8,
	body_part="original",
	text_category="existing" 
):
	keyids = [x for x in all_keyids[split] if x in keyids_index]
	index = [keyids_index[x] for x in keyids]

	unit_embs = unit_motion_embs[index]

	scores = model.compute_scores(text, unit_embs=unit_embs)

	keyids = np.array(keyids)
	sorted_idxs = np.argsort(-scores)
	best_keyids = keyids[sorted_idxs]
	best_scores = scores[sorted_idxs]

	datas = []
	for keyid, score in zip(best_keyids, best_scores):
		if len(datas) == nmax:
			break

		data = humanml3d_keyid_to_babel_rendered_url(keyid)
		if data is None:
			continue
		data["score"] = round(float(score), 2)
		data["input_text"] = text
		data["gpt_text"]=None

		if(body_part=="original"):
			data["gpt_text"] = None
		elif(body_part=="torso"):
			data["gpt_text"] = torso_json[data["text"]]
		elif(body_part=="hands"):
			data["gpt_text"] = hands_json[data["text"]]            
		elif(body_part=="legs"):
			data["gpt_text"] = legs_json[data["text"]]            
		datas.append(data)

	return datas

def retrieve_function(
	*,
	model,
	unit_motion_embs,
	all_keyids,
	text,
	keyids_index,
	index_keyids,
	torso_model,
	torso_unit_motion_embs,
	torso_keyids_index,
	torso_index_keyids,
	hands_model,
	hands_unit_motion_embs,
	hands_keyids_index,
	hands_index_keyids,    
	legs_model,
	legs_unit_motion_embs,
	legs_keyids_index,
	legs_index_keyids,    
	torso_input,
	hands_input,
	legs_input,
	split="test",
	nmax=8,
	text_category="existing",
):

	if(text_category == "existing"):
		torso_text = torso_json[text]
		hands_text = hands_json[text]
		legs_text = legs_json[text]
	elif(text_category == "zero-shot"):
		torso_text = torso_input
		hands_text = hands_input
		legs_text = legs_input
		if(len(torso_text)==0 or len(hands_text)==0 or len(legs_text)==0):
			req_prompt = prompt.replace("[ACTION]",str(text))
			gpt_texts = GPT_Completion([req_prompt])[0]

			torso_text = gpt_texts.split("1) Torso:")[1].split("2) Hands:")[0].strip()
			hands_text = gpt_texts.split("2) Hands:")[1].split("3) Legs:")[0].strip()
			legs_text = gpt_texts.split("3) Legs:")[1].strip()

	torso_datas = retrieve_helper(torso_model,torso_unit_motion_embs,all_keyids,torso_text,torso_keyids_index,torso_index_keyids,split=split,nmax=nmax,body_part="torso",text_category=text_category)
	hands_datas = retrieve_helper(hands_model,hands_unit_motion_embs,all_keyids,hands_text,hands_keyids_index,hands_index_keyids,split=split,nmax=nmax,body_part="hands",text_category=text_category)
	legs_datas = retrieve_helper(legs_model,legs_unit_motion_embs,all_keyids,legs_text,legs_keyids_index,legs_index_keyids,split=split,nmax=nmax,body_part="legs",text_category=text_category)

	return merge(torso_datas, hands_datas, legs_datas, text)


torso_model = TMR_text_encoder(TORSO_PATH).to(device)
hands_model = TMR_text_encoder(HANDS_PATH).to(device)
legs_model = TMR_text_encoder(LEGS_PATH).to(device)

torso_unit_motion_embs, torso_keyids_index, torso_index_keyids = load_unit_embeddings(
	TORSO_PATH, DATASET, device
)
hands_unit_motion_embs, hands_keyids_index, hands_index_keyids = load_unit_embeddings(
	HANDS_PATH, DATASET, device
)
legs_unit_motion_embs, legs_keyids_index, legs_index_keyids = load_unit_embeddings(
	LEGS_PATH, DATASET, device
)

all_keyids = load_splits(DATASET, splits=["test", "all"])

h3d_index = load_json(f"MoRAG_Hands/datasets/annotations/annotations.json")
torso_json = load_json(f"MoRAG_Torso/datasets/annotations/torso.json")
hands_json = load_json(f"MoRAG_Hands/datasets/annotations/hands.json")
legs_json = load_json(f"MoRAG_Legs/datasets/annotations/legs.json")
amass_to_babel = load_json("demo/amass_to_babel.json")

amass_data = joblib.load(amass_tar_path)
amass_enumerator = enumerate(amass_data)

amass_id_idx_map = {}
amass_name_idx_map = {}

for i, sample in amass_enumerator:
	amass_name_idx_map[sample['fname'].split("UNZIPPED_datasets/")[-1].split(".")[0]] = i

if (splits_choice=="Unseen"):
	split = "test"
else:
	split = "all"

# retrieval_data = {}
# retrieval_data["captions"]=[]
# retrieval_data["motions"]=[]
# retrieval_data["m_lengths"]=[]
# retrieval_data["torso_keyids"]=[]
# retrieval_data["hands_keyids"]=[]
# retrieval_data["legs_keyids"]=[]
# retrieval_data["torso_text"]=[]
# retrieval_data["hands_text"]=[]
# retrieval_data["legs_text"]=[]


def morag_retrieve(text, nvids=1):
	torso_input = ""
	hands_input = ""
	legs_input = ""

	merged_rots, merged_trans = retrieve_function(
		model=model, unit_motion_embs=unit_motion_embs, all_keyids=all_keyids, keyids_index=keyids_index, index_keyids=index_keyids, text=text, 
		torso_model=torso_model, torso_unit_motion_embs=torso_unit_motion_embs, torso_keyids_index=torso_keyids_index, torso_index_keyids=torso_index_keyids,
		hands_model=hands_model, hands_unit_motion_embs=hands_unit_motion_embs, hands_keyids_index=hands_keyids_index, hands_index_keyids=hands_index_keyids,
		legs_model=legs_model, legs_unit_motion_embs=legs_unit_motion_embs, legs_keyids_index=legs_keyids_index, legs_index_keyids=legs_index_keyids,
		torso_input=torso_input, hands_input=hands_input, legs_input=legs_input,
		split=split, nmax=nvids, text_category=text_category)

	return merged_rots, merged_trans

	if not os.path.exists(morag_retrieved):
	    os.makedirs(morag_retrieved)
	    
	with open('morag_retrieved/rots.pkl', 'wb') as file: 
	    pickle.dump(merged_rots, file) 

	with open('morag_retrieved/trans.pkl', 'wb') as file: 
	    pickle.dump(merged_trans, file) 

	# retrieval_data["captions"].append(output["captions"])
	# retrieval_data["motions"].extend(output["motions"][None,:,:].tolist())
	# retrieval_data["m_lengths"].append(output["m_lengths"])
	# retrieval_data["torso_keyids"].append(output["retrieved_keyids"]["torso"])
	# retrieval_data["hands_keyids"].append(output["retrieved_keyids"]["hands"])
	# retrieval_data["legs_keyids"].append(output["retrieved_keyids"]["legs"])
	# retrieval_data["torso_text"].append(output["retrieved_text"]["torso"])
	# retrieval_data["hands_text"].append(output["retrieved_text"]["hands"])
	# retrieval_data["legs_text"].append(output["retrieved_text"]["legs"])

if __name__ == "__main__":
	morag_retrieve(args.text, args.nretrieval)