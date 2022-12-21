# enzyminer
A machine learning program to predict promiscuity of esterases 
It will produce the following files:
* positive.fasta: a fasta file with the positive sequences
* negative.fasta: a fasta file with the negative sequences
* CSV files of the features generated by Possum and iFeature

The different sequences are ranked based on the applicability domain or how similar is to the training set, since if they are too different you would be predicting on something that the classifier has not been trained on.

> Note that Possum requires perl and python (compatible with python3). The package requirements can be found inside the Possum_toolkit's folder.  
> EP-pred also requires the NCBI Blast+ program, so it can generate the PSSM profiles  
> To generate the profiles, NCBI Blast+ needs a protein database, in this case, Uniref50 was used.   
> The generate_feature.py can turn a protein database into a Blast+ database if the appropriate flags are used and then use it to generate the PSSM profiles  
> Ifeature is another required package and can be download in https://github.com/Superzchen/iFeature/  
> 
> If you use EP-pred please cite all 3 papers:   
> * Xiang, R. et al. EP-Pred: A machine learning tool for bioprospecting promiscuous ester hydrolases. "Biomolecules", 2022, vol. 12, núm. 10, 1529. 10.3390/biom12101529  
> * Wang J, Yang B et al. POSSUM: a bioinformatics toolkit for generating numerical sequence feature descriptors based on PSSM profiles. Bioinformatics 2017;33(17):2756-2758. DOI: 10.1093/bioinformatics/btx302.  
> * Zhen Chen, Pei Zhao, Fuyi Li, André Leier, Tatiana T Marquez-Lago, Yanan Wang, Geoffrey I Webb, A Ian Smith, Roger J Daly*, Kuo-Chen Chou*, Jiangning Song*, iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences. Bioinformatics, 2018, 34(14): 2499–2502. https://doi.org/10.1093/bioinformatics/bty140  


