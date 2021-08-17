Maxwell Sime 2021
Submitted for my final year Research Project in Text Classification

SCC.java
==========
Seperated Classifier Chain.
Trains 2 Classifier Chains in parallel: 1 chain on all features and labels, 1 chain on only labels
Requires the use of J48(C48) base-classifier. Dependant on WEKA & MEKA libraries.

SCCNode.java
==========
Node class for use within SCC to simplify transformations, training, and testing.

SCC-1.0.zip
==========
Compiled file ready for use within the MEKA library installer.