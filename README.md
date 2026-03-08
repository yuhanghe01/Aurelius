 <p align="center"><a href="./"><img src=./imgs/aurelius_contribution.jpg width="50%"></a></p>

<p align="center">
<a href="https://openreview.net/pdf?id=LAYCYiIgZ1"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
<a href="https://yuhanghe01.github.io/Aurelius-Proj/"><img src="https://img.shields.io/badge/website-visit-blue?logo=github" alt="Project Website"></a>
<img src="https://img.shields.io/github/forks/yuhanghe01/Aurelius-Proj" alt="GitHub forks">
<img src="https://img.shields.io/github/stars/yuhanghe01/Aurelius" alt="GitHub stars">
<a href="https://huggingface.co/datasets/yuhanghe01/Aurelius"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="HuggingFace Data"></a>
</p>

## Aurelius: Relation Aware Text-to-Audio Generation at Scale

[Yuhang He<sup>1</sup>](https://yuhanghe01.github.io/),
[He Liang<sup>2</sup>](https://www.cs.ox.ac.uk/people/he.liang/),
[Yash Jain<sup>1</sup>](https://scholar.google.com/citations?user=Fr6QHDsAAAAJ&hl=en),
[Andrew Markham<sup>2</sup>](https://www.cs.ox.ac.uk/people/andrew.markham/),
[Vibhav Vineet<sup>1</sup>](https://vibhav-vineet.github.io//)
<br>
1. Microsoft Research
2. Department of Computer Science, University of Oxford. Oxford. UK.

### Aurelius Summary

 <p align="center"><a href="./"><img src=./imgs/audioeventset_audiorelset_illu.jpg width="99%"></a></p>

Aurelius aims to enable relation aware text-to-audio (TTA) generation at scale. It achieves so by introducing **AudioEventSet**: a 110-class audio event corpus maximumly covering 7 main audio classes that are 
commonly heard in our daily life, and **AudioRelSet**: a 100-relation audio relation corpus comprehensively evaluating TTA's relation aware generation capability from different perspectives. Events in **AudioEventSet** are distinctive, high-quality and mutually disambiguous. Relation in **AudioRelSet** reflects the relation both exists in real world and is neatly descriptive by text.

### AudioEventSet

**AudioEventSet** is provided in [AudioEventSet.json](./AudioEventSet.json), each event is provided in the following format:

```python
{
    "id": "horse_neighing",
    "name": "horse neighing",
    "child_ids": [],
    "synonyms": [
        "horse neigh",
        "neighing horse"
    ],
    "label_id": 22,
    "seed_audio_dir": "seed_audios/animal/wild_ground_animal/horse_neighing"
},
```

where `synonyms` provides multiple textual description of the same audio event. The instantiated seed audio can be found in `seed_audio_dir`, and each audio event is associated with multiple seed audio samples. 

### AudioRelSet

**AudioRelSet** is provided in [AudioRelSet.json](./AudioRelSet.json), each relation is provided in the following format:

```python
{
    "arity": 2,
    "explanation": "far",
    "id": "Farness",
    "name": "Farness",
    "child_ids": [],
    "text_prompt_template": [
        "generate {A} audio first, then the same {B} audio, ensuring that {A} audio is farther away from the listener than {B} audio",
        "sequentially generate {A} audio and another {B} audio, where {B} audio is closer to the listener than {A} audio",
        "at the beginning, generate {A} audio that is spatially distant, then continue to generate the same {B} audio that is spatially close",
        "first generate {A} audio that is farther away from the listener, then generate the same {B} audio that is closer",
        "create a spatially far {A} audio, followed by the same {B} audio that is spatially much closer"
    ],
    "event_rule_code": "AA",
    "event_max_timelen_code": "55",
    "child_label": [
        12
    ],
    "parent_label": [
        1
    ]
}
```

where `event_rule_code` and `event_max_timelen_code` are used to guide <text,audio> pair generation.

### <text,audio> Pair Generation

<text,audio> pair generation code is provided in [TextAudioPairGen](./TextAudioPairGen/), in which you have to first download seed audios from HuggingFace, edit `config.yaml` and run,

```python
    python main.py
```

### Evaluation

The evaluation (both general evaluation and relation aware evaluation) is provided in [Evaluation](./Evaluation/), in which you have to download pre-trained PANNs model and edit `config.yaml` accordingly. The evaluation is ran by,

```python
    python main.py
```

### Benchmarking Results

Detailed benchmarking result can be found in the paper. We report the benchmarking on TangoFlux model through three strategies: zero-shot, finetuning and training from scratch. The quantitative result is shown in the following table. The training data has 36,000 <text,audio> pairs, while the evaluation data has 10,000 <text,audio> pairs.

| Train Strategy | Model | #Param | FAD ↓ | KL ↓ | FD ↓ | mAPre (%) ↑ | mARel (%) ↑ | mAPar (%) ↑ | mAMSR (%) ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Zero-Shot | TangoFlux| 576M | 6.01 | 26.73 | 30.00 | 12.38 | 3.34 | 7.28 | 1.77 |
| finetuning | TangoFlux | 576M | 1.29 | 9.68 | 16.44 | 28.57 | 8.02 | 20.84 | 5.58 |
| from scratch | TangoFlux | 576M | 1.64 | 17.82 | 11.72 | 16.68 | 3.82 | 12.01 | 2.58 |

From this table, we can clearly observe that, while finetuning strategy gives the best-performing result, its target audio event presence accuracy is still below 30% (mAPre) and relation accuracy rate is below 10% (mARel). It in turns shows the incapability of existing TTA model in handling relation aware text-to-audio generation and necessity for the research community to further push the boundary of relation aware TTA by extensively exploiting our introduced Aurelius framework.

### Cite Aurelius

```bibtex
@inproceedings{yuhheAurelius2026,
title={Aurelius: Relation Aware Text-to-Audio Generation At Scale},
author={He, Yuhang and Liang, He and Jain, Yash and Markham, Andrew and Vineet, Vibhav},
booktitle={International Conference on Learning Representations (ICLR)},
year={2026}}
```

### Contact

Email: yuhanghe[at]microsoft.com
