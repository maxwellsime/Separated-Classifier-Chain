# Separated Classifier Chain
Submitted for my final year Research Project in Multi-Label Text Classification.
A novel variant of the Classifier Chain algorithm developed by [Jesse Read, et al.](https://www.cs.waikato.ac.nz/~eibe/pubs/chains.pdf) in 2009.

Requires the use of J48 Classifier, programmed specifically for the text classification software [WEKA](https://www.cs.waikato.ac.nz/ml/weka/) extension [MEKA](http://waikato.github.io/meka/).

# SCC.java
Separated Classifier Chain.
Trains 2 Classifier Chains in parallel: 1 chain on all features and labels, 1 chain on only labels.
Dependant on WEKA & MEKA libraries.

# SCCNode.java
Node class for use within SCC.java. Simplifies transformations, training, and testing.

# SCC-1.0.zip
Compiled file ready for use within the MEKA library installer.

# mjs84.pdf
Finished writeup for my Research Project, final mark: 74.

###### Maxwell Sime 2021
