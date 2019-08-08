# APANet

The codes in the attachment are based on the NetVLAD training and testing pipeline. So you should have the fundamental codes of
NetVLAD and the Place recognition datasets. [https://www.di.ens.fr/willow/research/netvlad/](https://www.di.ens.fr/willow/research/netvlad/)

Based on the NetVLAD codes, We rewrite the addPCA.m, addLayers.m to perform PCA-pw and add the APA module, you should overwrite the original files. 
APA module are implemented in the SPP_2468.m and layerL2Attention.m (layerL2nanAttention.m for cascaded attention block).

You can download the trained APANet models here: [https://drive.google.com/open?id=1kOVGjEhJRTkLsOYPFmzp2Ge8rcPttlia](https://drive.google.com/open?id=1kOVGjEhJRTkLsOYPFmzp2Ge8rcPttlia)
