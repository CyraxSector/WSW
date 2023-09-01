# WSW
Who Says What (WSW) is a pre-trained language model (PLM) which is designed to model who says what in multi-party conversations (MPCs) to understand equipping the discourse parsing in deep semantic structures and contextualized representations of utterances and interlocutors. To our knowledge, this is the first attempt to use the relative semantic distance of utterances in MPCs to design self-supervised tasks for MPC utterance structure modelling and MPC utterance semantic modelling. 

![Multi-Party Utterance Structure drawio](https://github.com/CyraxSector/WSW/assets/4902204/8e2005f8-7a0c-40c5-967c-5933526609c1)

We designed multiple self-supervised tasks on utterance structure modelling and utterance semantic modelling. Utterance structure modelling consists of two self-supervised tasks: Speaker Utterance Identification (SUI) and Exact Speaker Recognition (ESR). Root Utterance Node Detection (RUND) is designed to accomplish the utterance semantic modelling. Two downstream tasks such as Reply Utterance Selection (RUS) and Speaker Identification (SI) are performed based on the three self-supervised tasks. We conducted few experiments to evaluate the two downstream tasks. Experimental results show that WSW outperforms state-of-the-art (SOTA) models by large margins and achieves new SOTA performance on both the downstream tasks at four benchmarks datasets.
 
 # System Design 
![Multi-Party Conversational PLM Modelling drawio](https://github.com/CyraxSector/WSW/assets/4902204/9f601374-8bf3-426e-ac25-127bcdad35b3)

 # Experimental Results
 Evaluation results of RUS in terms of Recall (Rn@k). Non-PLMs and PLMs are shown in the 1st and 2nd row, respectively while ablation results are shown in the last row. 
 
![RUS](https://github.com/CyraxSector/WSW/assets/4902204/bc7c9be2-7793-49a7-b9cb-6e778e841d47)

Evaluation results of SI in terms of Precision (P@1). PLMs are shown in the 1st row while ablation results are shown in the last row.

![SI](https://github.com/CyraxSector/WSW/assets/4902204/6e9acda8-b964-4274-99d0-43ec13bc5ff0)

Evaluation results of SI in terms of Recall (Rn@k). Non-PLMs and PLMs are shown in the 1st and 2nd row, respectively while ablation results are shown in the last row.

![SI 2](https://github.com/CyraxSector/WSW/assets/4902204/0d6b5951-8e53-4e62-b0b2-354d705e827d)

Evaluation results of SI in terms of F1 score.

![SI](https://github.com/CyraxSector/WSW/assets/4902204/2723bd19-fcf8-42d9-8cb8-52869537e759)
