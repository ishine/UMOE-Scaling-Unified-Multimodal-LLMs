<h1 align="center">Uni-MoE 2.0: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://idealistxy.github.io/Uni-MoE-v2.github.io/"><img src="https://img.shields.io/badge/📰 -Website-228B22" style="margin-right: 5px;"></a>
  <a href="https://arxiv.org/abs/2405.11273"><img src="https://img.shields.io/badge/📄-Paper-8A2BE2" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/collections/HIT-TMG/lychee-uni-moe-20"><img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg" style="margin-right: 5px;"></a>
</div>


<p>
    <strong>Uni-MoE 2.0</strong> is a significant evolution of our original Uni-MoE 1.0 model. The previous version explored the use of Mixture of Experts (MoE) for unified multimodal language modeling, demonstrating its effectiveness across diverse modalities such as text, audio, speech, images, and video.
</p>
<p>
    Uni-MoE 2.0 builds on this foundation, rebuilt from scratch on the more powerful Qwen2.5-7B core, and introduces key architectural and training paradigms. Major improvements include a unified speech encoder, context-aware MoE-TTS, deep cross-modal alignment via 3D RoPE, and advanced MoE fusion strategies with a refined training recipe.
</p>






## Architecture

<img src="assets/images/architecture.png" alt="Architecture of Uni-MoE-2.0" style="max-width: 100%; width: 1000px; height: auto; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(123, 179, 255, 0.15);" align="center">
<div align="center">
<strong>Fig. 1</strong> The Uni-MoE 2.0 architecture processes multimodal data through a unified tokenization strategy. Audio is tokenized in 30-second clips, augmented with generation tokens for voice control in the context-aware MoE-TTS module, while images are encoded using a sliding window technique. Image Generation Tokens bridge the model to a task-aware diffusion transformer for end-to-end generation tasks. The model's comprehension is powered by Omni-Modality 3D RoPE, which aligns inputs across time, and a sophisticated MoE layer. This MoE layer dynamically routes information using diverse experts, with stability ensured by null experts (for token skipping) and modality-specific routed experts (A, V, T indicate audio, visual, and textual expert pretrained on corresponding data). In contrast, compact shared experts (only 1/8 size of routed experts) enable efficient cross-modal knowledge transfer.
</div>

## Results
<img src="assets/images/results.png" alt="Main Results of Uni-MoE 2.0" style="max-width: 100%; width: 1000px; height: auto; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(123, 179, 255, 0.15);" align="center">
<div align="center">
<strong>Fig. 2</strong> The performance of Uni-MoE 2.0 and previous SOTA omnimodal large models.
</div>

## Getting Started

### 1. Clone this repository and navigate to the Uni-MoE 2.0 folder
```bash
git clone https://github.com/HITsz-TMG/Uni-MoE.git
cd Uni-MoE-2
```
### 2. Set up environment
Install the evaluation environment according to the requirements.
```bash
conda create -n uni_moe_2 python=3.11
conda activate uni_moe_2
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install flash-attn==2.6.0.post1 --no-build-isolation
pip install clip==1.0@git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
```
## Uni-MoE 2.0 Weights
We have released the weights of five versions of Uni-MoE 2.0 on Hugging Face, as shown in below tables.
| Model | Capalicity|
| --------  | -----------|
| [ Uni-MoE 2.0-Omni](https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Omni) | All-Modality Understanding, Speech Generation, Image Generation and More| 
| [ Uni-MoE 2.0-Base](https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Base) | All-Modality Understanding| 
| [ Uni-MoE 2.0-Thinking](https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Thinking) | Omni Long-form Reasoning Capabilities. |
| [ Uni-MoE 2.0-Image](https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Image) | Powerful Image Generation, Image Editting, Low-Level Abality|
| [ Uni-MoE 2.0-TTS](https://huggingface.co/HIT-TMG/Uni-MoE-TTS) | Speech Generation|

## Cookbooks
We are preparing [cookbooks]() for many capabilities, including multi-images understanding, omni video understanding, audio generation, image editting and more. Welcome to learn more!

| Cookbook | Description |
| -------- | ----------- |
| [Image Understanding](Uni-MoE-2/examples/cookbook_image.py) | Image Understanding, Multi-Images Understanding |
| [Video Understanding](Uni-MoE-2/examples/cookbook_video.py) | Video Understanding, Omni Video Understanding |
| [Audio Tasks](Uni-MoE-2/examples/cookbook_audio.py) | Audio Understanding, ASR, TTS, Long ASR/TTS, Speech Chat, Vision+Speech Chat |
| [Visual Generation](Uni-MoE-2/examples/cookbook_visual_gen.py) | Image Generation, Image Editing, Controllable Image Generation, Low-level Image Restoration |
| [Visual Reasoning](Uni-MoE-2/examples/cookbook_thinking.py) | Long-CoT Visual Reasoning |


## Example Usage
We also provide a simple example on the usage of this repo. 
```python
import torch
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.qwen_vl_utils import process_mm_info
from uni_moe.model import deepspeed_moe_inference_utils

processor = Qwen2VLProcessor.from_pretrained("HIT-TMG/Uni-MoE-2.0-Omni")

model = GrinQwen2VLOutForConditionalGeneration.from_pretrained("HIT-TMG/Uni-MoE-2.0-Omni", torch_dtype=torch.bfloat16).cuda()

processor.data_args = model.config
processor.image_processor = model.vision_tower.image_processor
processor.audio_processor = model.audio_tower.audio_processor

messages = [{
    "role": "user", 
    "content": [
            {"type": "text", "text": "<audio>\n<image>\nAnswer the question in the audio."},
            {"type": "audio", "audio": "examples/assets/audio/quick_start.mp3"},
            {"type": "image", "image": "examples/assets/image/quick_start.jpg"}
        ]
}]

texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
texts = texts.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>","<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>","<|vision_start|><|video_pad|><|vision_end|>")
image_inputs, video_inputs, audio_inputs = process_mm_info(messages)

inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    audios=audio_inputs,
    padding=True,
    return_tensors="pt",
)
inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

inputs = inputs.to(device=model.device)

output_ids = model.generate(
    **inputs,
    use_cache=True,
    pad_token_id=processor.tokenizer.eos_token_id,
    max_new_tokens=4096,
    temperature=1.0,
    do_sample=True
)

text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(text)

```

## Citation

```
Please cite the repo if you use the model or code in this repo.

@misc{li2024unimoe,
    title={Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts}, 
    author={Yunxin Li and Shenyuan Jiang and Baotian Hu and Longyue Wang and Wanqi Zhong and Wenhan Luo and Lin Ma and Min Zhang},
    year={2024},
    eprint={2405.11273},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}

```
