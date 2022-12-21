# enzyminer
A machine learning program to predict promiscuity of esterases 
It will produce the following files:
* positive.fasta: a fasta file with the positive sequences
* negative.fasta: a fasta file with the negative sequences
* CSV files of the features generated by Possum and iFeature

The different sequences are ranked based on the applicability domain or how similar is to the training set, since if they are too different you would be predicting on something that the classifier has not been trained on.

> Note that Possum requires perl and python. The package requirements can be found inside the Possum_toolkit's folder.  
> EP-pred also requires the NCBI Blast+ program, so it can generate the PSSM profiles  
> To generate the profiles, NCBI Blast+ needs a protein database, in this case, Uniref50 was used.   
> The generate_feature.py can turn a protein database into a Blast+ database if the appropriate flags are used and then use it to generate the PSSM profiles  


