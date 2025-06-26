# :sparkles:[CVPR 2025 Highlight] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation

[CVPR 2025] Official repository of "FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation".

[[Project Page]](https://ivl.cs.brown.edu/research/foundhand.html) [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_FoundHand_Large-Scale_Domain-Specific_Learning_for_Controllable_Hand_Image_Generation_CVPR_2025_paper.pdf) 

<p> <strong>Authors</strong>:
    <a href="https://arthurchen0518.github.io/">Kefan Chen<sup>*</sup></a>
    ·
    <a href="https://chaerinmin.github.io/">Chaerin Min<sup>*</sup></a>   
    ·
	<a href="https://lg-zhang.github.io/">Linguang Zhang</a> 	 
    ·
	<a href="https://shreyashampali.github.io/">Shreyas Hampali</a>          
    ·
	<a href="https://scholar.google.co.uk/citations?user=9HoiYnYAAAAJ&hl=en">Cem Keskin</a> 
    ·
    <a href="https://cs.brown.edu/people/ssrinath/">Srinath Sridhar</a>
</p>

<img src="./assets/teaser.jpg" alt="[Teaser Figure]" style="zoom:80%;" />


## Installation

1. Create a virtual environment and install necessary dependencies

```shell
conda create  -n foundhand python=3.9
conda activate foundhand
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0  --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.3.0
pip install timm==1.0.7 tqdm opencv-python scikit-image matplotlib tensorboard
```

3. Download pretrained [FoundHand](https://drive.google.com/file/d/1AR08bpX5Ync7VykXq-S7ww8YIbRihVMD/view?usp=sharing), [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and [SD-VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt) models and place them under `./weights/`.

## Demo Notebooks
```shell
./demos/FixHand.ipynb       # Fix malformed AI-generated hand.
./demos/Image2Image.ipynb   # Gesture transfer and domain transfer.
./demos/Image2Video.ipynb   # Video generation given the first frame and hand motion sequence.
./demos/NVS.ipynb	    # Novel view synthesis.
```

## Checklist

- [x] Release model weights and code.
- [x] Release demo notebooks.
- [ ] Release FoundHand-10M data.
- [ ] Release training code.
- [ ] Release inference code.

## Acknowledgement

Part of this work was done during Kefan (Arthur) Chen’s internship at Meta Reality Lab. This work was additionally supported by NSF CAREER grant #2143576, NASA grant #80NSSC23M0075, and an Amazon Cloud Credits Award.


## Citation


```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Kefan and Min, Chaerin and Zhang, Linguang and Hampali, Shreyas and Keskin, Cem and Sridhar, Srinath},
    title     = {FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17448-17460}
}
```
