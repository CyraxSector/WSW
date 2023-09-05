# <i> Who Says What (WSW) </i>
Who Says What (WSW) is a pre-trained language model (PLM) which is designed to model who says what in multi-party conversations (MPCs) to understand equipping the discourse parsing in deep semantic structures and contextualized representations of utterances and interlocutors. To our knowledge, this is the first attempt to use the relative semantic distance of utterances in MPCs to design self-supervised tasks for MPC utterance structure modelling and MPC utterance semantic modelling. 

![Multi-Party Utterance Structure drawio](https://github.com/CyraxSector/WSW/assets/4902204/8e2005f8-7a0c-40c5-967c-5933526609c1)

We designed multiple self-supervised tasks on utterance structure modelling and utterance semantic modelling. Utterance structure modelling consists of two self-supervised tasks: Speaker Utterance Identification (SUI) and Exact Speaker Recognition (ESR). Root Utterance Node Detection (RUND) is designed to accomplish the utterance semantic modelling. Two downstream tasks such as Reply Utterance Selection (RUS) and Speaker Identification (SI) are performed based on the three self-supervised tasks. We conducted few experiments to evaluate the two downstream tasks. Experimental results show that WSW outperforms state-of-the-art (SOTA) models by large margins and achieves new SOTA performance on both the downstream tasks at four benchmarks datasets.
 
 # System Design 
![Multi-Party Conversational PLM Modelling drawio](https://github.com/CyraxSector/WSW/assets/4902204/9f601374-8bf3-426e-ac25-127bcdad35b3)

 # Experimental Results
 Evaluation results of RUS in terms of Recall (Rn@k). Non-PLMs and PLMs are shown in the 1st and 2nd row, respectively while ablation results are shown in the last row. 
 
![RUS](https://github.com/CyraxSector/WSW/assets/4902204/bc7c9be2-7793-49a7-b9cb-6e778e841d47)

Evaluation results of SI in terms of Precision (P@1). PLMs are shown in the 1st row while ablation results are shown in the last row.

![SI](https://github.com/CyraxSector/WSW/assets/4902204/fb68a409-4656-44ad-b17a-b132efe26caa)

Evaluation results of SI in terms of Recall (Rn@k). Non-PLMs and PLMs are shown in the 1st and 2nd row, respectively while ablation results are shown in the last row.

![SI](https://github.com/CyraxSector/WSW/assets/4902204/395b26ab-14d0-45dc-a0f4-68dc958a7c05)

Evaluation results of SI in terms of F1 score.

![SI](https://github.com/CyraxSector/WSW/assets/4902204/4be13b5b-a209-46fc-96f4-6065ea11a397)
