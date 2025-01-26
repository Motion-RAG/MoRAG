<div align="center">

# MoRAG - Multi-Fusion Retrieval Augmented Generation for Human Motion


<a href="https://www.linkedin.com/in/sai-shashank-54288219b"><strong>Sai Shashank Kalakonda</strong></a>
·
<a href="https://shubhmaheshwari.github.io/"><strong>Shubh Maheshwari</strong></a>
·
<a href="https://ravika.github.io/"><strong>Ravi Kiran Sarvadevabhatla</strong></a>


[![WACV2025](https://img.shields.io/badge/WACV-2025-9065CA.svg?logo=WACV)](https://wacv2025.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-TMR-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2409.12140)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>


## Description

Please visit our [**webpage**](https://motion-rag.github.io/) for more details.

### Bibtex
If you find this code useful in your research, please cite:

```bibtex
@InProceedings{MoRAG,
    title     = {MoRAG - Multi-Fusion Retrieval Augmented Generation for Human Motion},
    author    = {Kalakonda, Sai Shashank and Maheshwari, Shubh and Sarvadevabhatla, Ravi Kiran},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision ({WACV})},
    year      = 2025
}
```

You can also put a star :star:, if the code is useful to you.

## Installation :construction_worker:

<details><summary>Create environment</summary>
&emsp;

Create a python virtual environnement:
```bash
python -m venv ~/.venv/morag_env
source ~/.venv/morag_env/bin/activate
```

Install [PyTorch](https://pytorch.org/get-started/locally/)
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install remaining packages:
```
python -m pip install -r requirements.txt
```

</details>

<details><summary>Set up the datasets</summary>

### Introduction
Each MoRAG_[part] folder contains the data setup files, and you can use any of them for the required motion data setup. However, setting up text embeddings must be done separately for each body part, as we generate part-specific embeddings using GPT-generated text descriptions unique to each body part.

If you are currious about the motion data details, refer this file: [DATASETS.md](DATASETS.md).

### Motion data setup

#### Get the motion data
Please follow the instructions of the ``raw_pose_processing.ipynb`` of the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) repo, to get the ``pose_data`` folder.
Then copy or symlink the pose_data folder in ``datasets/motions/``:
```bash
ln -s /path/to/HumanML3D/pose_data datasets/motions/pose_data
```

#### Compute the motion features
Run the following command, to compute the HumanML3D Guo features on the whole AMASS (+HumanAct12) dataset.

```bash
python -m prepare.compute_guoh3dfeats
```

It should process the features (+ mirrored version) and saved them in ``datasets/motions/guoh3dfeats``.


### Text data setup

#### Generate GPT text descriptions
To generate GPT text descriptions for all text annotations associated to the motion sequences in the HumanML3D dataset, run the following commands. Pre-generated GPT descriptions are available in the ``datasets`` folder within each MoRAG_[part] folder.

```bash
python utils/gpt_generate.py
python utils/gpt_segregate.py
```

``gpt_generate.py`` generates GPT texts for all text annotations, and ``gpt_segregate.py`` organizes the generated descriptions into separate ``[part].json`` files based on part details. Copy these ``[part].json`` files to the respective ``datasets`` folder in each MoRAG_[part] directory.  

**Note:** Some text instances might cause errors while running ``gpt_segregate.py`` due to format differences. These problematic instances are stored in ``issue_gpt.json``. Regenerate GPT texts for these annotations.

#### Compute GPT text embeddings (Independently for each body part)
Run this command in each MoRAG_[part] folder to compute the sentence and token embeddings for the GPT-generated texts required for MoRAG.

```bash
python -m prepare.text_embeddings data=humanml3d
```

This will save:
- the token embeddings of ``distilbert`` in ``datasets/annotations/humanml3d/[part]_token_embeddings``
- the sentence embeddings of ``all-mpnet-base-v2`` in ``datasets/annotations/humanml3d/[part]_sent_embeddings``

</details>

## Training :rocket:

Run the following command in each MoRAG_[part] folder to train the part-specific MoRAG model.

```bash
python train.py
```

<details><summary>Details</summary>
&emsp;

It will train part-specific MoRAG on HumanML3D and store the outputs in `[part]_outputs/humanml3d_guoh3dfeats`, which will be referred to as `RUN_DIR`.

</details>

<details><summary>Extracting weights</summary>
&emsp;

After training, run the following command, to extract the weights from the checkpoint:

```bash
python extract.py run_dir=RUN_DIR
```

It will take the last checkpoint by default. This should create the folder ``RUN_DIR/last_weights`` and populate it with the files: ``motion_decoder.pt``, ``motion_encoder.pt`` and ``text_encoder.pt``.
This process makes loading models faster, it does not depends on the file structure anymore, and each module can be loaded independently. This is already done for pretrained models.

</details>

## Encode the whole motion dataset

Run the below command to encode the dataset for the retrieval process.

```bash
python encode_dataset.py run_dir=RUN_DIR
```

## Pretrained models :dvd:

All the pretrained models for each MoRAG_[part], along with the encoded dataset latent files, can be downloaded from this [link](https://zenodo.org/records/14741949).


## Retrieval

As spatial composition relies on part-specific retrieved sequences, rotation and translation data are required. For this, we use AMASS data. Follow the instructions in [AMASS data setup](https://github.com/atnikos/teach?tab=readme-ov-file#data) to set up the AMASS dataset. Once processing is complete, copy the `amass.pth.tar` file into the `amass_data` folder, which will be accessed during MoRAG retrieval.

Ensure the following variables are correctly set in `morag_retrieval.py`:
- **OpenAI API key** (`openai_api_key`)
- **AMASS data path** (`amass_tar_path`)
- **Encoded part-specific latent data paths using MoRAG-trained models**:
  - `TORSO_PATH`
  - `HANDS_PATH`
  - `LEGS_PATH`

Finally, run the command below to perform part-specific motion retrieval and compose it into a full-body motion sequence for a given text description. The composed/merged rotation and translation data for `nretrieval` sequences will be saved in `morag_retrieved/rots.pkl` and `morag_retrieved/trans.pkl`.

```bash
python morag_retrieval.py text="A person is walking forward." nretrieval=1
```


## Visualization

Refer to `rots_to_smpl_conversion/rots_to_smpl.py` for the visualization helper function, which is based on the [MDM](https://github.com/GuyTevet/motion-diffusion-model) visualization setup.

## Acknowledgments

Many parts of this code are based on the official implementations of [TMR](https://github.com/Mathux/TMR), [SINC](https://github.com/atnikos/sinc), [TEACH](https://github.com/atnikos/teach), [MDM](https://github.com/GuyTevet/motion-diffusion-model) and [ReMoDiffuse](https://github.com/mingyuan-zhang/ReMoDiffuse). We extend our gratitude to the respective authors for making their code publicly available.

This template was adapted from the [TMR](https://github.com/Mathux/TMR) GitHub repository.

## License :books:
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including PyTorch, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.