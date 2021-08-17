package meka.classifiers.multilabel;

import meka.classifiers.multilabel.*;
import meka.core.A;
import meka.core.F;
import weka.core.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;

import java.io.FileReader;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import javax.management.Attribute;

/**
 * SCCNode, for use with SCC.java
 * Adapts function names and main() tests from CC.java, written for the ML program MEKA
 * @author	Maxwell	Sime
 * @version April 2021
 */
public class SCCNode implements Serializable {

	private int j = -1;
	private int d = -1;
	private int inX[] = null;
	private int paY[] = null;
	private Instances T = null;
	private Instance t_ = null;
	private Classifier h = null;
	private int map[] = null;

	/**
	 * SCCNode - A Node 'j', taking inputs from all parents inX and paY.
	 * @param	j		the label index of this node
	 * @param	inX		attribute indices going into this node
	 * @param	paY		label indices going into this node
	 */
	public SCCNode(int j, int inX[], int paY[]) {
		this.j = j;
		this.inX = inX;
		this.paY = paY; 
	}

	/** getParentsY - get the parents (indices) of this node */
	public int[] getParentsY() {
		return paY;
	}

	/**
	 * Transform - transform dataset D for this node.
	 * this.j defines the current node index, e.g., 3
	 * this.paY[] defines parents,            e.g., [1,4]
	 * we should remove the rest,             e.g., [0,2,5,...,L-1]
	 * @return dataset we should remove all variables from D EXCEPT current node.
	 */
	public Instances transform(Instances D) throws Exception {
		int L = D.classIndex();
		d = D.numAttributes() - L;
		int keep[] = A.append(this.paY,j);		// keep all parents and self!
		Arrays.sort(keep);
		int remv[] = A.invert(keep,L); 	// i.e., remove the rest < L
		Arrays.sort(remv);
		map = new int[L];
		for(int j = 0; j < L; j++) {
			map[j] = Arrays.binarySearch(keep,j);
		}
		Instances D_ = F.remove(new Instances(D),remv, false); 
		D_.setClassIndex(map[this.j]);
		return D_;
	}
	
	/**
	 * Transform2 - transform dataset D for this node's 2nd chain.
	 * this.j defines the current node index, e.g., 3
	 * this.paY[] defines parents,            e.g., [1,4]
	 * we should remove the rest,             e.g., [0,2,5,...,L-1]
	 * @return dataset we should remove all variables and attributes from D EXCEPT current node, and parents.
	 */
	public Instances transform2(Instances D) throws Exception {
		/*int L = D.numAttributes();
		d = D.numAttributes() - L;
		int keep[] = A.append(this.paY,j);		// keep all parents and self!
		Arrays.sort(keep);
		int remv[] = A.invert(keep,L); 	// i.e., remove the rest < L
		Arrays.sort(remv);
		map = new int[L];
		for(int j = 0; j < L; j++) {
			map[j] = Arrays.binarySearch(keep,j);
		}
		Instances D_ = F.remove(new Instances(D),remv, false); 
		D_.setClassIndex(map[this.j]);
		return D_;*/

		int L = D.numAttributes(); 		
		int keep[] = A.append(this.paY, this.j);	// keep all parents and self
		Arrays.sort(keep);					// required for binarysearch
		map = new int[L];
		for(int j = 0; j < L; j++){
			map[j] = Arrays.binarySearch(keep, j);
		}
		Instances D_ = F.remove(new Instances(D), keep, true);		// removes all attributes but keep
		D_.setClassIndex(map[this.j]);
		
		return D_;
	}

	/**
	 * Build - Create transformation for this node, and train classifier of type H upon it.
	 * The dataset should have class as index 'j', and remove all indices less than L *not* in paY.
	 */
	public void build(Instances D, Classifier H) throws Exception {
		// transform data
		T = transform(D);
		// build SLC 'h'
		h = AbstractClassifier.makeCopy(H);
		h.buildClassifier(T);
		// save templates
		//t_ = new SparseInstance(T.numAttributes());
		//t_.setDataset(T);
		//t_.setClassMissing();								// [?,x,x,x]
		T.clear();
	}
	/**
	 * Build2 - Create transformation for this node, and train classifier of type H upon it.
	 * The dataset should have class as index 'j', and remove all indices excluding paY and j.
	 */
	public void build2(Instances D, Classifier H) throws Exception {
		// transform data
		T = transform2(D);
		// build SLC 'h'
		h = AbstractClassifier.makeCopy(H);
		h.buildClassifier(T);
		// save templates
		//t_ = new SparseInstance(T.numAttributes());
		//t_.setDataset(T);
		//t_.setClassMissing();								// [?,x,x,x]
		T.clear();
	}

	/**
	 * Build2 however returns transformed D_
	 * @param D
	 * @param H
	 * @throws Exception
	 */
	public Instances build2(Instances D, Classifier H, Boolean exists) throws Exception {
		// transform data
		T = transform2(D);
		// build SLC 'h'
		h = AbstractClassifier.makeCopy(H);
		h.buildClassifier(T);
		// save templates
		//t_ = new SparseInstance(T.numAttributes());
		//t_.setDataset(T);
		//t_.setClassMissing();								// [?,x,x,x]
		return T;
	}

	/**
	 * Sum of distribution for this node given instances D, transformed using transform2
	 * Used as part of chain order calculation in SCC
	 * @param D
	 * @return sum of distributionForInstance
	 */
	public double distributionSum(Instances D, int L) throws Exception{
		double[] y = new double[L];
		double[] temp;
		double sum = 0;

		for (Instance l : D){
			temp = h.distributionForInstance(l);
			sum += temp[Utils.maxIndex(temp)];
		}
		sum = sum/((double)D.numInstances());

		return sum;
	}
	
	/**
	 * The distribution this this node, given input x.
	 * @return p( y_j = k | x , y_pred ) for k in {0,1}
	 */
	public double[] distribution(Instance x, double ypred[]) throws Exception {
		Instance x_ = transform(x,ypred);
		return h.distributionForInstance(x_);
	}

	/** Same as #distribution(Instance, double[]), but the Instance is pre-transformed with ypred inside. */
	public double[] distributionT(Instance x_) throws Exception {
		return h.distributionForInstance(x_);
	}

	/**
	 * Sample the distribution given by #distribution(Instance, double[]).
	 * @return y_j ~ p( y_i | x , y_pred )
	 */ 
	public double sample(Instance x, double ypred[], Random r) throws Exception {
		double p[] = distribution(x, ypred);
		return A.samplePMF(p,r);
	}

	/**
	 * Transform - turn [y1,y2,y3,x1,x2] into [y1,x1,x2].
	 * @return transformed Instance
	 */
	public Instance transform(Instance x) throws Exception {
		x = (Instance)x.copy();
		int L = x.classIndex();					// the class' index (this.j)
		x.setDataset(null);
		for(int j = 0; j < L - 1; j++) {	// L - L_c would be 4
			x.deleteAttributeAt(0);
		}
		x.setDataset(T);
		x.setClassMissing();
		return x;
	}

	/**
	 * Transform - turn [y1,y2,y3,x1,x2] into [y1,y2,x1,x2].
	 * @return transformed Instance
	 */
	public Instance transform(Instance x, double ypred[]) throws Exception {
		x = (Instance)x.copy();
		int L = x.classIndex();					
		int L_c = (paY.length + 1);				
		x.setDataset(null);
		for(int j = 0; j < (L - L_c); j++) {
			x.deleteAttributeAt(0);
		}
		for(int pa : paY) {
			//System.out.println("x_["+map[pa]+"] <- "+ypred[pa]);
			x.setValue(map[pa],ypred[pa]);
		}
		x.setDataset(T);
		x.setClassMissing();
		return x;
	}

	public void updateTransform(Instance t_, double ypred[]) throws Exception {
		for(int pa : this.paY) {
			t_.setValue(this.map[pa],ypred[pa]);
		}
	}

	/**
	 * Transform x then return probabilies array of choices
	 * Returns array for use in heuristic
	 * @return p( y_j = k | x , y_pred ) for k in {0,1}
	 */
	public double[] classify(Instance x) throws Exception {
		Instance x_ = transform(x);
		return h.distributionForInstance(x_);
	}

	/**
	 * Transform x then return probabilies array of choices
	 * Returns array for use in heuristic
	 * @return p( y_j = k | x , y_pred ) for k in {0,1}
	 */
	public double[] classify(Instance x, double ypred[]) throws Exception {
		Instance x_ = transform(x,ypred);
		return h.distributionForInstance(x_);
	}

	/**
	 * Transform.
	 * @param	D		original Instances
	 * @param	c		to be the class Attribute
	 * @param	pa_c	the parent indices of c
	 * @return	new Instances T
	 */
	public static Instances transform(Instances D, int c, int pa_c[]) throws Exception {
		int L = D.classIndex();
		int keep[] = A.append(pa_c,c);			// keep all parents and self!
		Arrays.sort(keep);
		int remv[] = A.invert(keep,L); 	// i.e., remove the rest < L
		Arrays.sort(remv);
		Instances T = F.remove(new Instances(D),remv, false); 
		int map[] = new int[L];
		for(int j = 0; j < L; j++) {
			map[j] = Arrays.binarySearch(keep,j);
		}
		T.setClassIndex(map[c]);
		return T;
	}

	/**
	 * Returns the underlying classifier.
	 *
	 * @return the classifier
	 */
	public Classifier getClassifier() {
		return h;
	}

	/**
	 * Main - run some tests.
	 * main from MEKA CCNode.java
	 */
	public static void main(String args[]) throws Exception {
		Instances D = new Instances(new FileReader(args[0]));
		Instance x = D.lastInstance();
		D.remove(D.numInstances()-1);
		int L = Integer.parseInt(args[1]);
		D.setClassIndex(L);
		double y[] = new double[L];
		Random r = new Random();
		int s[] = new int[]{1,0,2};
		int PA_J[][] = new int[][]{
			{},{},{0,1},
		};

		// MUST GO IN TREE ORDER !!
		for(int j : s) {
			int pa_j[] = PA_J[j];
			System.out.println("PARENTS = "+Arrays.toString(pa_j));
			//MLUtils.randomize(pa_j,r);
			System.out.println("**** TRAINING ***");
			SCCNode n = new SCCNode(j,null,pa_j);
			n.build(D,new SMO());
			System.out.println("============== D_"+j+" / class = "+n.T.classIndex()+" =");
			System.out.println(""+n.T);
			System.out.println("**** TESTING ****");
			Instance x_ = n.transform(x,y);
			System.out.println(""+x_);
			y[j] = 1;
		}
	}

}

