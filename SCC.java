package meka.classifiers.multilabel;

import meka.classifiers.multilabel.*;
import meka.core.A;
import weka.core.*;
import java.util.*;


/**
 * NovelCC, technically parallel CC. For use with J48 classifier
 * Adapts function names from CC.java, written for the ML program MEKA
 * @author	Maxwell	Sime
 * @version April 2021
 */
public class SCC extends ProblemTransformationMethod{

	private static final long serialVersionUID = -4115294965331340629L;

	protected SCCNode nodes[] = null;

	protected SCCNode nodes2[] = null;

	protected Random m_R = null;

	protected int m_Chain[] = null;

	public void printChain(int[] c){
		System.out.print("[");
		for(int j : c){
			System.out.print(j + ", ");
		}
		System.out.print("]\n");
	}

	public void printChain2(double[] c){
		System.out.print("[");
		for(double j : c){
			System.out.printf("%5f", j);
		}
		System.out.print("]\n");
	}

	/**
	 * Prepare a Chain.<br>
	 * - set chain order based on predictive probabilites labels have on each other
	 * @param L		number of labels
	 * @param D		data file instance
	 */
	protected void prepareChain(int L, Instances D) throws Exception {
		int chain[] = new int[]{};
		int rem[] = A.make_sequence(L);		// remaining labels
		int temp;


		if (getDebug()) System.out.print(":- Chain Order: ");
		for(int x =0; x < L; x++){
			temp = orderChain(D, L, chain, rem);
			chain = A.append(chain, temp);
			if(getDebug()){
				System.out.print("new chain # ");
				printChain(chain);
			}

			// build new remainder chain, meka functions do not work
			int[] tempRem = new int[rem.length-1];
			int i = 0;
			for(int j = 0; j < rem.length; j++){
				if(rem[j] == temp)j++;
				if(j != rem.length){
					tempRem[i] = rem[j];
					i++;
				}
			}
			rem = tempRem;
			if(getDebug()){
				System.out.print("new rem # ");
				printChain(rem);
			}

			if(rem.length == 1){	// chain complete
				chain = A.append(chain, rem[0]);
				break;
			}
		}

		// set chain
		prepareChain(chain);
	}

	public int orderChain(Instances D, int L, int[] c, int[] r) throws Exception{
		double p[] = new double[L]; 	// probabilities
		int pa[] = new int[]{};			// parents
		
		if(getDebug()){
			System.out.print("chain c == ");
			printChain(c);
			System.out.println("chain r == ");
			printChain(r);
		}
		
		for(int i : c){		// append old parent data
			pa = A.append(pa, i);
		}

		for(int i : r){				// calculate probabilities
			double sum = 0.0d;
			for(int j : r){
				if(i != j){			// stop training node on itself
					int[] t_i = A.append(pa, i);
					if(getDebug()){
						System.out.println("temporary index"+ i +"==============");
						printChain(t_i);
						System.out.print(i + " " + j + "'s turn \n");
					}
		
					//create SCCNode
					SCCNode n = new SCCNode(j, null, t_i);
					Instances D_ = n.build2(D, m_Classifier, true);

					sum += n.distributionSum(D_, L);
				}
			}
			p[i] = sum/ ((double)L);
		}

		//printChain2(p);

		// build chain
		int highest = -1;
		for(int i : r){
			if ((highest == -1) || (p[i] > p[highest])){
				highest = i;
			}
		}
		if (getDebug()) System.out.println("Best Pred Performance: " + highest + " = " + p[highest]);
		return highest;
	}

	/**
	 * Prepare a Chain. Set the specified 'chain'.
	 * It must contain all indices [0,...,L-1] (but in any order)
	 * @param chain		a specified chain
	 */
	public void prepareChain(int chain[]) {
		m_Chain = Arrays.copyOf(chain,chain.length);
		if(getDebug()) 
			System.out.println("Chain s="+Arrays.toString(m_Chain));
	}

	public int[] retrieveChain() {
		return m_Chain;
	}

	@Override
	public void buildClassifier(Instances D) throws Exception {
		testCapabilities(D);

		int L = D.classIndex();

		if(m_Chain == null){
			prepareChain(L, D);
		}
		if (getDebug()) System.out.printf("Class Index = %d\n", L);

		/*
		 * make a classifier node for each label, taking the parents of all previous nodes
		 */
		if(getDebug()) System.out.print(":- Chain (");
		nodes = new SCCNode[L];
		nodes2 = new SCCNode[L];
		int pa[] = new int[]{};
		
		for(int j : m_Chain) {
			if (getDebug()) System.out.print(" "+D.attribute(j).name()+"\n");

			// build first chain (just parents + class)
			if(nodes.length != 0){	// empty nodes = first item
				if(getDebug()) System.out.print("Building first chain");
				nodes2[j] = new SCCNode(j, null, pa);
				nodes2[j].build2(D, m_Classifier);
			}
			
			nodes[j] = new SCCNode(j, null, pa);

			// build second chain (base CC)
			if(getDebug())System.out.print("Building second chain");
			nodes[j].build(D, m_Classifier);

			pa = A.append(pa,j);
		}
		if (getDebug()) System.out.println(" ) -:");

		// to store posterior probabilities (confidences)
		// maybe create a second confidences array and work on that?
		// work on processing confidence array correctly for SCC?
		confidences = new double[L];
	}

	protected double confidences[] = null;

	/**
	 * GetConfidences - get the posterior probabilities of the previous prediction (after calling distributionForInstance(x)).
	 */
	public double[] getConfidences() {
		return confidences;
	}

	@Override
	public double[] distributionForInstance(Instance x) throws Exception {
		int L = x.classIndex();
		double y[] = new double[L];

		for(int j : m_Chain) {
			double y1[];
			double y2[];

			// classify first chain
			y1 = nodes[j].classify((Instance)x.copy(), y);

			if(j == m_Chain[0]){				// catching first loop
				y[j] = Utils.maxIndex(y1);
			}else{
				// classify second chain
				y2 = nodes2[j].classify((Instance)x.copy(),y);
				
				int y1_i = Utils.maxIndex(y1);
				int y2_i = Utils.maxIndex(y2);

				if(y1_i != y2_i){				// chains have different classification
					if(y1[y1_i] >= y2[y2_i]){
						y[j] = y1_i;			// choose first chain answer
						//System.out.printf("chain 1 %d (%f) is better than %d(%f) \n",y1_i, y1[y1_i], y2_i, y2[y2_i]);
					}else{
						y[j] = y2_i;			// choose second chain answer
						//System.out.printf("chain 2 %d (%f) is better than %d(%f) \n", y2_i, y2[y2_i], y1_i, y1[y1_i]);
					}
				}
			}
		}

		return y;
	}

	public static void main(String args[]) {
		ProblemTransformationMethod.evaluation(new SCC(), args);
	}
}