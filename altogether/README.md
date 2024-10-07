# Altogether: Image Captioning via Re-aligning Alt-text

Altogehter is a captioner that transforms Internet-scale alt-texts into dense captions. It does not caption images from scratch and generative naive and well-known visual concepts that provide little value to an average user (e.g., "a dog is walking in the park" offer minimal utility to most users). Instead, it completes Internet-scale alt-texts into dense captions while preserving information in alt-texts that describing the image that most annotators cannot annotate.

![Altogether](altogether.png)


```bibtex
@inproceedings{xu2024altogether,
   title={Altogether: Image Captioning via Re-aligning Alt-text},
   author={Hu Xu, Po-Yao Huang, Xiaoqing Ellen Tan, Ching-Feng Yeh, Jacob Kahn, Christine Jou, Gargi Ghosh, Omer Levy, Luke Zettlemoyer, Wen-tau Yih, Shang-Wen Li, Saining Xie and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:xxxx.xxxxx},
   year={2024}
}
```


## Training

Config `config/altogether.py` to the proper path.

Single GPU Testing

```bash
python src/training/main.py altogether
```

2 Nodes training via SLURM 

```bash
python submit.py altogether  # --resume epoch_pt.pt  # for fine-tuning from existing alt-texts pretraining.
```

## Inference

```bash
python altogether/infer.py altogether:epoch_ft.pt <your_wds_path> <output_path>
```


## License

The majority of Altogether is licensed under CC-BY-NC, portions of the project are available under separate license terms: CLIPCap is licensed MIT and open_clip is licensed under the https://github.com/mlfoundations/open_clip license.

## Acknowledgement
We gratefully acknowledge [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption) and the [OpenCLIP](https://github.com/mlfoundations/open_clip) team for initial CLIP codebase.
