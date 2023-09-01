# WSW
Who Says What (WSW) is a pre-trained language model (PLM) which is designed to model who says what in multi-party conversations (MPCs) to understand equipping the discourse parsing in deep semantic structures and contextualized representations of utterances and interlocutors. To our knowledge, this is the first attempt to use the relative semantic distance of utterances in MPCs to design self-supervised tasks for MPC utterance structure modelling and MPC utterance semantic modelling. We designed multiple self-supervised tasks on utterance structure modelling and utterance semantic modelling. Utterance structure modelling consists of two self-supervised tasks: Speaker Utterance Identification (SUI) and Exact Speaker Recognition (ESR). Root Utterance Node Detection (RUND) is designed to accomplish the utterance semantic modelling. Two downstream tasks such as Reply Utterance Selection (RUS) and Speaker Identification (SI) are performed based on the three self-supervised tasks. We conducted few experiments to evaluate the two downstream tasks. Experimental results show that WSW outperforms state-of-the-art (SOTA) models by large margins and achieves new SOTA performance on both the downstream tasks at four benchmarks datasets.
 
 # System Design 
![Multi-Party Conversational PLM Modelling drawio](https://github.com/CyraxSector/WSW/assets/4902204/9f601374-8bf3-426e-ac25-127bcdad35b3)

 # Experimental Results
 Evaluation results of RUS in terms of recall (Rn@k). Non-PLMs and PLMs are shown in the 1st and 2nd row, respectively while ablation results are shown in the last row. 
![RUS](https://github.com/CyraxSector/WSW/assets/4902204/a46e7eaf-aca4-484b-b1b2-98109180241b)
