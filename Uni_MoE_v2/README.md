
# Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts
[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL), [Baotian Hu](https://faculty.hitsz.edu.cn/hubaotian), [Longyue Wang](http://www.longyuewang.com/), [Wanqi Zhong](URL), [Lin Ma](https://forestlinma.com/), [Wenhan Luo](https://whluo.github.io/), [Min Zhang](https://faculty.hitsz.edu.cn/MinZhang)
</h4>
This is the repo of Uni-MoE-v2

Uni-MoE-v2 is our updated edition ofMoE-based unified multimodal model and can handle diverse modalities including audio, speech, image, text, and video. This new framework support multi-GPUs training and inferencing which speed up the optimization process and the scale of our model.

## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/model.png" height="100%" width="75%"/></div>

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the Uni-MoE folder
```bash
git clone https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.git
cd UMOE-Scaling-Unified-Multimodal-LLMs/Uni_MoE_v2
```

2. Install Package
```Shell
conda create -n unimoe_v2 python==3.9.16
conda activate unimoe
pip install -r env.txt
conda install mpi4py
pip install tutel git+https://github.com/microsoft/tutel@56dbd664341cf6485c9fa292955f77d3ac918a65
pip install flash-attn==2.5.6
pip install VideoFileClip
```

3. Replace all the absolute pathnames '/path/to/' with your specific path to the Uni-MoE file

## ⚡️ Uni-MOE Weights

To use our new version model, all weights should be downloaded, the base link is not released yet.

After downloading all of them, organize the weights as follows in 'Uni_MoE/checkpoint' folder:
```
└── checkpoint
    ├── Uni-MoE-speech-base-8
    ├── Uni-MoE-speech-8-e2
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [Fine-tuned BEATs_iter3+ (AS2M)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) |
| Uni-MoE-audio-base-model | [Uni-MoE/Uni-MoE-audio-base](https://huggingface.co/VictorJsy/Uni-MoE-audio-base/tree/main) |
| Uni-MoE-audio-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-audio-e2](https://huggingface.co/VictorJsy/Uni-MoE-audio-e2/tree/main) |
| Uni-MoE-speech-base-model | [Uni-MoE/Uni-MoE-speech-base](https://huggingface.co/VictorJsy/Uni-MoE-speech-base/tree/main) |
| Uni-MoE-speech-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-speech-e2](https://huggingface.co/VictorJsy/Uni-MoE-speech-e2/tree/main) |
| Uni-MoE-speech-interval-base-model | [Uni-MoE/Uni-MoE-speech-base-interval](url) |
| Uni-MoE-speech-fine-tuned-chekpoint-v1.5 | [Uni-MoE/Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) |

* Uni-MoE-speech refers to the MOE-Task2 and Uni-MoE-audio refers to the MOE-Task3 in our paper.

## 🗝️ Dataset

### Training Data
| DataSet  | Type |
|----------|-----------|
| [LLaVA-Instruct-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) | [imgae(train2014)](http://images.cocodataset.org/zips/train2014.zip)(todo) |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | [imgae(train2014)](http://images.cocodataset.org/zips/train2014.zip) |
| [Video-Instruct-Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT) | [video(from youtube)](https://www.youtube.com/) |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/json_files) | [audio](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files) |
| [AudioCaps](https://audiocaps.github.io/) | [audio(Cap)](https://audiocaps.github.io/) |
| [ClothoAQA](https://zenodo.org/records/6473207)  | [audio(QA)](https://zenodo.org/records/6473207) |
| [ClothoV1](https://zenodo.org/records/3490684) | [audio(Cap)](https://zenodo.org/records/3490684) |
| [MELD](https://affective-meld.github.io/) | [audio(Music)](https://affective-meld.github.io/) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | [Speech(TTS)](https://www.cs.cmu.edu/~glai1/data/race/) |
| [LibriSpeech](https://www.openslr.org/12) | [Speech(Long)](https://www.openslr.org/12) |

We use TTS technical to convert long text to speech to construct long speech understanding data.

Overall, all training tasks (*16 comparative experiments covering models with single-expert and MoE configurations*) are as follows:

| Training Tasks        | Data Types                                      | Data Size | Epochs | Trainable Modules        | Pretraining tasks           |
|-----------------------|-------------------------------------------------|-----------|--------|--------------------------|-----------------------------|
| Audio-Language Pretraining | WaveCaps*, Audiocap*, MELD, ClothoV1       | 194K      | 2      | Audio Q-former, Audio projection layer | -                           |
| Speech-Language Pretraining | Common Voice (Short Speech)                                | 1.7M      | 2      | Speech Q-former, Speech projection layer | -                           |
| Single-Modality-Expert-Task1 | LLaVA-Instruction-150K(I-A)                  | 150K      | 1      | LoRA, Speech projection layer | Speech-pretrain-task         |
| Single-Modality-Expert-Task2 | LLaVA-Instruction-150K(T-I)                  | 150K      | 1      | LoRA, Image projection layer | Speech-pretrain-task         |
| Single-Modality-Expert-Task3 | LLaVA-Instruction-150K(I-A)                  | 150K      | 1      | LoRA, Speech Q-former, Speech and Image projection layer         | Speech-pretrain-task         |
| Single-Modality-Expert-Task4 | LLaVA-Instruction-150K(I-A), RACE(T-A), LibriSpeech | 271K | 1 | LoRA, Speech & Image projection | Speech-pretrain-task         |
| Single-Modality-Expert-Task5 | LLaVA-Instruction-150K(T-I), RACE(T-A), LibriSpeech | 271K | 1 | LoRA, Speech & Image projection | Speech-pretrain-task         |
| Single-Modality-Expert-Task6 | LLaVA-Instruction-150K(I-A), LLaVA-Instruction-150K(T-I), RACE(T-A), LibriSpeech | 421K | 1 | LoRA, Speech & Image projection | Speech-pretrain-task         |
| Single-Modality-Expert-Task7 | RACE(T-A), LibriSpeech, RACE(T-A)-MC                       | 209K      | 1      | LoRA, Speech projection layer | Speech-pretrain-task         |
| Single-Modality-Expert-Task8 | WaveCaps*, Audiocap*, MELD, ClothoAQA, ClothoV1 | 203K    | 1      | LoRA, Audio projection layer | Audio-pretrain-task          |
| MoE-Task1               | LLaVA-Instruction-Dataset(T-I), LLaVA-Instruction-150K(I-A), RACE(T-A), LibriSpeech, RACE(T-A)-MC | 509K | 3 | LoRA, Router, speech & image projection layer | LLava-V1.5-LoRA, Single-Modality-Expert-Tasks 2/3/7 |
| MoE-Task1-short-speech  | LLaVA-Instruction-Dataset(T-I), LLaVA-Instruction-150K(I-A) | 300K | 3 | LoRA, Router, speech & image projection layer | LLava-V1.5-LoRA, Single-Modality-Expert-Tasks 2/3/7 |
| MoE-Task2               | Video-Instruction-150K, LLaVA-Instruction-Dataset(T-I), RACE(T-A), LibriSpeech, RACE(T-A)-MC | 459K | 2 | LoRA, Router, speech & image projection layer | Llava-v1.5-LoRA, Single-Modality-Expert-Tasks 2/3/7 |
| MoE-Task3               | Video-Instruction-150K, LLaVA-Instruction-Dataset(T-I), WaveCaps*, Audiocap*, MELD,  ClothoAQA, ClothoV1 | 453K | 2 | LoRA, Router, audio & image projection layer | LLava-V1.5-LoRA, Single-Modality-Expert-Tasks 2/3/8 |
| Pure-MoE-Task1          | Video-Instruction-Dataset, LLaVA-Instruction-Dataset(T-I), WaveCaps*, Audiocap*, MELD, ClothoAQA, ClothoV1 | 453K | 2 | LoRA, Router, audio & image projection layer | LLava-V1.5-LoRA            |
| Pure-MoE-Task2          | Video-Instruction-Dataset, LLaVA-Instruction-Dataset(T-I), WaveCaps*, Audiocap*, MELD, ClothoAQA, ClothoV1 | 453K | 2 | LoRA, Router, audio & image projection layer | -            |

``*`` refers to the fact that the dataset we use is only a subset. ``MC`` represents the multi-choice setting. ``I-A`` means image-audio pairs, which convert the question into the corresponding speech.  ``T-I`` shows the original text-image pairs. ``T-A`` indicates the contextual paragraph of the RACE dataset is transferred into the long speech. ``Pretraining task`` represents the tasks included in the previous training stage.



### Evaluation Data
| DataSet  | Input Type |
|----------|----------|
| [AOKVQA](https://allenai.org/project/a-okvqa/home) | Text-Image |
| [OKVQA](https://okvqa.allenai.org/) | Text-Image |
| [VQAv2](https://visualqa.org/) | Text-Image |
| [ClothoAQA](https://zenodo.org/records/6473207) | Text-Audio |
| [ClothoV1](https://zenodo.org/records/3490684) | Text-Audio |
| [ClothoV2](https://zenodo.org/records/3490684) | Text-Audio |
| [MMBench](https://mmbench.opencompass.org.cn/home) | Text-Image |
| [MMBench-Audio](https://mmbench.opencompass.org.cn/home) | Text-Image-Speech(Long) |
| [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main) | Text-Speech(Long) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | Text-Speech(Long) |
| [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |Text-Video-Audio |
| [Activitynet-QA](https://github.com/MILVLG/activitynet-qa) |Text-Video-Audio |

#### College Entrance English Examination Listening Part

We build a real speech understanding dataset to check the practical long speech recognition capabilities: [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main)
It comprises 150 questions related to long audio segments with an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds.


## 🌈 How to infer and deploy your demo

1. Make sure that all the weights are downloaded and the running environment is set correctly.
2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_speech.sh) using ```bash inference_audio.sh``` ```bash inference_speech.sh``` or run the following commands to inference:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/inference_all.py
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/inference_all.py
```


## 🌈 How to train and evaluate on datasets

Training:
1. Make sure that all the weights are downloaded and the environment is set correctly, especially for the base model.
2. Our training data can be downloaded from [UMOE-Speech-453k.json](url) and [UMOE-Cap-453k.json](url).
3. Relevant vision and audio files: [Dataset](#Training-Data)
4. Run training scripts: [`finetune_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_audio.sh) or [`finetune_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech.sh) using ```bash finetune_audio.sh``` ```bash finetune_speech.sh```, remember to modify the training set with your own preference.
5. For multiple GPUs training, run training scripts: [`finetune_speech_dp.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh) using ```bash finetune_speech_dp.sh```, remember to modify the training set with your own preference.

Evaluation:
1. Prepare the evaluation set using the form as [`samples.json`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/data_sample/samples.json).
2. Run evaluation scripts: [`eval_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_audio.sh) or [`eval_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_speech.sh) using ```bash eval_audio.sh``` ```bash eval_speech.sh``` or run the following commands to eval:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/eval.py\
 --data_path /path/to/clotho.json\
 --data_type clothov1\
 --output test.json
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/eval.py\
 --data_path /path/to/vqa_eval.json\
 --data_type vqa\
 --output test.json
```
We recommend using 80GB GPU RAM to run all experiments.

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{li2024umoe,
      title={Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts}, 
      author={},
      publisher={},
      year={2024},
}

```