package ccnovel;

import meka.classifiers.multilabel.*;
import meka.core.A;
import weka.core.*;
import java.util.*;


/**
 * NovelCC, technically parallel BR
 * @author	Maxwell	Sime
 * @version January 2021
 */
public class CCn extends ProblemTransformationMethod{

	private static final long serialVersionUID = -4115294965331340629L;

	protected CnNode nodes[] = null;

	protected Random m_R = null;

	protected int m_Chain[] = null;

	/**
	 * Prepare a Chain.<br>
	 * - set chain order based on predictive probabilites labels have on each other
	 * @param L		number of labels
	 * @param D		data file instance
	 */
	protected void prepareChain(int L, Instances D) throws Exception {
		int chain[] = retrieveChain();
		double p[] = new double[L]; //probabilities
		double temp;

		if (getDebug()) System.out.print(":- Chain Probabilities: ");
		for(int i = 0; i < L; i++){
			int[] t_i = {i};
			double[] y_pred = {i};
			double sum = 0;
			for(int j = 0; j < L; j++){
				if(i == j){
					//stops training a node on itself
					j++;
				}

				//create CnNode
				CnNode n = new CnNode(j, null, t_i);
				n.build2(D, m_Classifier);
				//get probabity and save value
				for(Instance l : D){
					temp = n.distribution(l, y_pred)[0];
					sum += temp;	
				}
			}
			
			//average p using sum
			p[i] = sum/L;
			if (getDebug()) System.out.print(i + " = " + p[i] + ", ");
		}

		if (getDebug()) System.out.print(":- Chain Order: ");
		//build chain
		for(int i = 0; i < L; i++){
			int highest = 0;	//index of current best value
			for(int j = 0; j < L; j++){
				if(p[j] > p[highest]){
					highest = j;
				}
			}
			p[highest] = 0d;
			chain[i] = highest;
			if (getDebug()) System.out.print(i + " = " + chain[i] + ", ");
		}

		// set it
		prepareChain(chain);
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

		prepareChain(L, D);

		/*
		 * make a classifier node for each label, taking the parents of all previous nodes
		 */
		if(getDebug()) System.out.print(":- Chain (");
		nodes = new CnNode[L];
		int pa[] = new int[]{};
		
		for(int j : m_Chain) {
			if (getDebug()) 
			System.out.print(" "+D.attribute(j).name());

			//build first chain (just parents)
			if(nodes.length != 0){	// empty nodes = first item
				nodes[j] = new CnNode(j, null, pa);
				nodes[j].build2(D, m_Classifier);
			}else{
				nodes[j] = new CnNode(j, null, pa);
			}

			//build second chain (just attributes)
			nodes[j].build(D, m_Classifier);
			pa = A.append(pa,j);
		}
		if (getDebug()) System.out.println(" ) -:");

		// to store posterior probabilities (confidences)
		// maybe create a second confidences array and work on that?
		// work on processing confidence array correctly for CCn?
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
			// h_j : x,pa_j -> y_j
			y[j] = nodes[j].classify((Instance)x.copy(),y); 
		}

		return y;
	}

	/**
	 * SampleForInstance.
	 * predict y[j] stochastically rather than deterministically (as with distributionForInstance(Instance x)).
	 * @param	x	test Instance
	 * @param	r	Random 			&lt;- TODO probably can use this.m_R instead
	 */
	public double[] sampleForInstance(Instance x, Random r) throws Exception {
		int L = x.classIndex();
		double y[] = new double[L];

		for(int j : m_Chain) {
			double p[] = nodes[j].distribution(x, y);
			y[j] = A.samplePMF(p,r);
			confidences[j] = p[(int)y[j]];
		}

		return y;
	}


	/**
	 * GetTransformTemplates - pre-transform the instance x, to make things faster.
	 * @return	the templates
	 */
	public Instance[] getTransformTemplates(Instance x) throws Exception {
		int L = x.classIndex();
		Instance t_[] = new Instance[L];
		double ypred[] = new double[L];
		for(int j : m_Chain) {
			t_[j] = this.nodes[j].transform(x,ypred);
		}
		return t_;
	}

	/**
	 * SampleForInstance - given an Instance template for each label, and a Random.
	 * @param	t_	Instance templates (pre-transformed) using #getTransformTemplates(x)
	 */
	public double[] sampleForInstanceFast(Instance t_[], Random r) throws Exception {

		int L = t_.length;
		double y[] = new double[L];

		for(int j : m_Chain) {
			double p[] = nodes[j].distribution(t_[j],y);               // e.g., [0.4, 0.6]
			y[j] = A.samplePMF(p,r);                                   // e.g., 0
			confidences[j] = p[(int)y[j]];                             // e.g., 0.4
			nodes[j].updateTransform(t_[j],y); 						   // need to update the transform #SampleForInstance(x,r)
		}

		return y;
	}

	/**
	 * TransformInstances - this function is DEPRECATED.
	 * this function preloads the instances with the correct class labels ... to make the chain much faster,
	 * but CnNode does not yet have this functionality ... need to do something about this!
	 */
	public Instance[] transformInstance(Instance x) throws Exception {
		return null;
		/*
		//System.out.println("CHAIN : "+Arrays.toString(this.getChain()));
		int L = x.classIndex();
		Instance x_copy[] = new Instance[L];
		root.transform(x,x_copy);
		return x_copy;
		*/
	}

	/**
	 * ProbabilityForInstance - Force our way down the imposed 'path'. 
	 * <br>
	 * TODO rename distributionForPath ? and simplify like distributionForInstance ?
	 * <br>
	 * For example p (y=1010|x) = [0.9,0.8,0.1,0.2]. If the product = 1, this is probably the correct path!
	 * @param	x		test Instance
	 * @param	path	the path we want to go down
	 * @return	the probabilities associated with this path: [p(Y_1==path[1]|x),...,p(Y_L==path[L]|x)]
	 */
	public double[] probabilityForInstance(Instance x, double path[]) throws Exception {
		int L = x.classIndex();
		double p[] = new double[L];

		for(int j : m_Chain) {
			// h_j : x,pa_j -> y_j
			double d[] = nodes[j].distribution((Instance)x.copy(),path);  // <-- posterior distribution
			int k = (int)Math.round(path[j]);                             // <-- value of interest
			p[j] = d[k];                                                  // <-- p(y_j==k) i.e., 'confidence'
			//y[j] = path[j];
		}

		return p;
	}

	public static void main(String args[]) {
		ProblemTransformationMethod.evaluation(new CCn(), args);
	}
}