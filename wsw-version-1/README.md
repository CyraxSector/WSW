# <i> Who Says What (WSW) </i>: A Novel Model for Utterance-Aware Speaker Identification in Text-Based Multi-Party Conversations
This repository contains the source code of the initial version of Who Says What (WSW), a novel PLM which models who says what in an MPC to understand equipping discourse parsing in deep semantic structures and contextualized representations of utterances and interlocutors. To our knowledge, this is the first attempt to use the relative semantic distance of utterances in MPCs for designing self-supervised tasks for graphical-based MPC understanding. These self-supervised tasks are used to train a Bidirectional Encoder Representations from Transformer (BERT) PLM in a multi-task framework, to model contextual representations of utterances and interlocutors of an MPC. WSW models two self-supervised tasks including <i> speaker/addressee identification </i> and <i> response utterance selection </i> on top of pre-trained language models (PLMs) to enhance the PLM’s ability to understand MPCs. <br>

## Dependencies
Python 3.8 <br>
Tensorflow 2.10.0

## Download
- Download the [BERT released by the Google research](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), 
  and move to path: ./uncased_L-12_H-768_A-12 <br>
  
- Download the [Hu et al. (2019) dataset](https://drive.google.com/file/d/1qSw9X22oGGbuRtfaOAf3Z7ficn6mZgi9/view?usp=sharing) used in the original paper,
  and move to path: ```./data/ijcai2019/``` <br>

- Download the [Ouchi and Tsuboi (2016) dataset](https://drive.google.com/file/d/1nMiH6dGZfWBoOGbIvyBJp8oxhD8PWSNc/view?usp=sharing) used in the original paper,
  and move to path: ```./data/emnlp2016/``` <br>

- Download the [Asher et al. (2016) dataset](https://www.irit.fr/STAC/corpus.html) used in the original paper,
  and move to path: ```./data/stac2016/``` <br>

- Download the [Molweni (2020) dataset](https://github.com/HIT-SCIR/Molweni/tree/main/DP) used in the original paper,
  and move to path: ```./data/molweni2020/``` <br>
  
Unzip the dataset and run the following commands. <br>
  ```
  cd data/emnlp2016/
  python data_preprocess.py
  ```

## Pre-training Phase - Self-supervised Tasks Formation
Pre-training phase, the initial phase of WSW, is essential for generating the respective logic for self-supervised tasks formation. Self-supervised tasks formation can be identified as utterance structure modelling and utterance semantic modelling. Utterance structure modelling consists of two self-supervised tasks: <i> Speaker Utterance Identification (SUI) </i> and <i> Exact Speaker Recognition (ESR)</i>. <i> Root Utterance Node Detection (RUND) </i> task is designed to accomplish the utterance semantic modelling. 
```
python pretraining_data_kernel.py 
```
Running the pre-training process.
```
python pretraining_runner.py 
```
The pre-trained model will be saved to the path ```./uncased_L-12_H-768_A-12_pretrained```<br> 
Modify the filenames in this folder to make it the same as those in Google's BERT.

## Fine-tuning Phase - Downstream Tasks Formation
Two downstream tasks such as <i> Reply Utterance Selection (RUS) </i> and <i> Speaker Identification (SI) </i> are performed based on the three self-supervised tasks. <i> RUS </i> maps with the self-supervised task of <i> SUI</i>. The objective of <i> RUS </i> is to determine the semantic relevance of the context and a given reply-to utterance. <i> SI </i> maps with the self-supervised task of <i> ESR </i> determining the exact speaker of any given utterance of an MPC.

### <u> Fine-tuning downstream task <i> SI. </i> </u> <br>

Create the fine-tuning data for the downstream task <i> SI. </i>
```
python finetuning_speaker_identification_data.py 
```
Running the fine-tuning process for the downstream task <i> SI. </i>
```
python finetuning_speaker_identification_runner.py
```
Running the testing process for the downstream task <i> SI. </i>
```
python finetuning_speaker_identification_tester.py
```

### <u> Fine-tuning downstream task <i> RUS. </i> </u> <br>

Create the fine-tuning data for the downstream task <i> RUS. </i>
```
python finetuning_reply_utterance_selection_data.py 
```
Running the fine-tuning process for the downstream task <i> RUS. </i>
```
python finetuning_reply_utterance_selection_runner.py
```
Running the testing process for the downstream task <i> RUS. </i>
```
python finetuning_reply_utterance_selection_tester.py
```

## Cite
If you think of using the code, please cite our paper:
**"Who Says What (WSW): A Novel Model for Utterance-Aware Speaker Identification in Text-Based Multi-Party Conversations"**
Y.H.P.P. Priyadarshana (Prasan Yapa), Zilu Liang, and Ian Piumarta. _WEBIST 2023_

```
 title = "Who Says What (WSW): A Novel Model for Utterance-Aware Speaker Identification in Text-Based Multi-Party Conversations",
 author = "Y.H.P.P. Priyadarshana and 
           Zilu Liang and
           Ian Piumarta",
 booktitle = "Proceedings of the 19th International Conference on Web Information Systems and Technologies",
 month = Nov,
 year = "2023",
 address = "Rome, Italy",
 publisher = "SCITEPRESS"
```

## Acknowledgments
Some code of this project are referenced from [MPC-BERT](https://github.com/JasonForJoy/MPC-BERT). We thank their open source materials for the contribution in our study. <br>
Thanking Wenpeng Hu et al., for providing the processed Hu et al. (2019) GSN dataset used in their [paper](https://www.ijcai.org/proceedings/2019/0696.pdf). <br>
Thanking Ran Le for providing the processed Ouchi and Tsuboi (2016) ARS dataset used in their [paper](https://www.aclweb.org/anthology/D19-1199.pdf). <br>
Thanking Nicholas Asher and Julie Hunter for providing the processed Asher et al., (2016) STAC dataset used in their [paper](https://hal.science/hal-02124399/). <br>
Thanking Jiaqi Li et al., for providing the processed Li et al., (2020) Molweni dataset used in their [paper](https://aclanthology.org/2020.coling-main.238/).

## Inquiry
Feel free to contact us (2022md05@kuas.ac.jp) for any issues or inquiry.
