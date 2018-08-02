import java.util.*;

/**
 * This class represents a segmentation related to a video file
 * used in shot boundary detection task programs of TREC 2001.
 *
 * @author
 * This software was produced by NIST, an agency of the
 * U.S. government, and by statute is not subject to copyright in the
 * United States. Recipients of this software assume all
 * responsibilities associated with its operation, modification and
 * maintenance.
 *
 */

public class Segmentation{
	/**
     * The video name the segmentation is related to.
     *
     * @see #getVideoName()
     */
    private String videoName;
    /**
     * This object is containing all the transitions of the segmentation.
     *
     * @see #getElements()
     */
    private TreeSet transitionSet;
    private Vector dropped;
    
    /**
     * The number of frames of the video.
     *
     * @see #getElements()
     */
    private int totalFrameNumber = 0;
	
    /**
     * Constructs Segmentation object with an ordering comparator for its set of transitions
     *
     * @param     videoName The video name the segmentation is related to.
     * @param     orderingComparator The comparator used to order transitions in the set of transitions
     */
    public Segmentation(String videoName, Comparator orderingComparator){
	this(videoName, orderingComparator, 0);
	//this.videoName = videoName;
	//this.transitionSet = new TreeSet(orderingComparator);
	//this.dropped = new Vector();
    }
    /**
     * Constructs Segmentation object with an ordering comparator for
     * its set of transitions.
     *
     * @param     videoName The video name the segmentation is related to.
     * @param     orderingComparator The comparator used to order transitions in the set of trnasitions.
     * @param totalFrameNumber The number of frames of the video.  
     */
    public Segmentation(String videoName, Comparator orderingComparator, int totalFrameNumber){
	this.totalFrameNumber = totalFrameNumber;
	this.videoName = videoName;
	this.transitionSet = new TreeSet(orderingComparator);
	this.dropped = new Vector();
    }
    /**
     * Process the comparison between the current assumed reference
     * segmentation and the given system segmentation.
     *
     * @param     sysId The system Id of the given system segmentation.
     * @param     systemSegmentation The system segmentation to be compared with.
     * @param     shortGradualThreshold The threshold below which short gradual 
     *            transition can match an abrupt tansition.
     * @return    an object representing all the results (measures...) of the comparison.
     * @exception TransitionDefinitionException if the definition of a
     * transition is not correct (pre-frame >= post-frame).  
     */
    public comparisonResult processComparison(String sysId, 
					      Segmentation systemSegmentation
					      )	throws TransitionDefinitionException{

	//**********************************unchanged

	comparisonResult result = new comparisonResult(sysId, this.videoName, 
						       this.totalFrameNumber);
	
	int insertedTransCount = 0;
	int deletedTransCount = 0;		
	int insertedTransCountCut = 0;
	int deletedTransCountCut = 0;
	int insertedTransCountGradual = 0;
	int deletedTransCountGradual = 0;
	int frameNumCount = 0;
	int frameRecallDenCount = 0;
	int framePrecisionDenCount = 0;
	
	String nl = System.getProperty("line.separator");
	String insertionsSection = nl+"<insertion>"+nl+nl;
	String cutMatchSection = nl;
	String gradualMatchSection = nl;
	String deletionsSection = nl+"<deletion>"+nl+nl;
	
	//***********************************  unchanged
	
	// For each transition within the Reference Segmentation (this)
	// detection of insertion or match. 
	// 1-Ensure that the transition are in time sequence
	// ensured only if the comparator for the TreeSet is the appropriate one
	// 2-Verify that cut are considered as 2 frames transition before overlapping test  !!!!!!!!!!
	// verified cause as is: cut overlappings are enabled cause of ref-cut expansion
	
	Iterator referenceIterator = this.transitionSet.iterator();
	while(referenceIterator.hasNext()){
	    
	    Transition aReferenceTransition = (Transition) referenceIterator.next();
	    // Expanding the reference transition
	    if(aReferenceTransition.isCut()){
		if(aReferenceTransition.getPre() < 5)
		    aReferenceTransition.reset(0, aReferenceTransition.getPost() + 5);
		else
		    aReferenceTransition.reset(aReferenceTransition.getPre() - 5, aReferenceTransition.getPost() + 5);
	    }
	    Iterator systemIterator = systemSegmentation.getElements();
	    
	    Transition bestMatch = null;
	    double bestMatchFrameRecall = 0.0;
	    double bestMatchFramePrecision = 0.0;
	    int bestOverlap = 0;
	    
	    while(systemIterator.hasNext()){
		
		Transition aSystemTransition = (Transition) systemIterator.next();
		// Expanding the systemTransition if it is a CUT
		// dont forget to reset it to normal values after the loop			    
		if(aSystemTransition.isCut()){
		    if(aSystemTransition.getPre() == 0){
			aSystemTransition.reset(0, aSystemTransition.getPost() + 1);				
		    }
		    else
			aSystemTransition.reset(aSystemTransition.getPre() - 1, aSystemTransition.getPost() + 1);
		}
		Transition intersection = null;
		if(aSystemTransition.matched == false && 
		   (intersection = aSystemTransition.intersection(aReferenceTransition)) != null &&
		   aSystemTransition.sameType(aReferenceTransition)){				    
		    
		    int intersectionLength = intersection.length();
		    if(intersectionLength > bestOverlap){
			
			//Re-init previous bestMatch matched value
			if(bestMatch != null)
			    bestMatch.matched = false;

			bestMatch = aSystemTransition;
			aSystemTransition.matched = true;
			aReferenceTransition.matched = true;
			bestOverlap = intersectionLength;
			bestMatchFrameRecall = (double) bestOverlap / (double) aReferenceTransition.length();
			bestMatchFramePrecision = (double) bestOverlap / (double) bestMatch.length();
			
		    }
		    else if(intersectionLength > 0 && intersectionLength == bestOverlap){
			
			double candidateFramePrecision = (double) intersectionLength / (double) aSystemTransition.length();    
			if(candidateFramePrecision > bestMatchFramePrecision){
			    
			    //Re-init previous bestMatch matched value
			    if(bestMatch != null)
				bestMatch.matched = false;
		    
			    bestMatch = aSystemTransition;
			    aSystemTransition.matched = true;
			    aReferenceTransition.matched = true;
			    bestOverlap = intersection.length();
			}
			//if still equal : you arrived after so nothing
		    }
		}
		// Reseting the system transition size 
		// if it is a CUT (previously expanded)
		if(aSystemTransition.isCut()){
		    if(aSystemTransition.getPre() == 0){
			aSystemTransition.reset(aSystemTransition.getPost() - 2, aSystemTransition.getPost() - 1);		
		    }
		    else aSystemTransition.reset(aSystemTransition.getPre() + 1, aSystemTransition.getPost() - 1);
		}
	    }//End LOOP over system transitions
	    
	    // Reseting the reference transition size 
	    // if it is a CUT (previously expanded)
	    if(aReferenceTransition.isCut()){
		if(aReferenceTransition.getPre() == 0)
		    aReferenceTransition.reset(aReferenceTransition.getPost() - 6, aReferenceTransition.getPost() - 5);
		else
		    aReferenceTransition.reset(aReferenceTransition.getPre() + 5, aReferenceTransition.getPost() - 5);
	    }
	    // Persistance for aReferenceTransition
	    // either deletion or match
	    if(!aReferenceTransition.matched){
		
		deletionsSection += aReferenceTransition.refToString();
		deletedTransCount++;
		if(aReferenceTransition.isCut() || aReferenceTransition.isShortGradual())
		    deletedTransCountCut++;
		else
		    deletedTransCountGradual++;
	    }
	    else{
		if((aReferenceTransition.isCut() && bestMatch.isCut()) ||
		   (aReferenceTransition.isCut() && bestMatch.isShortGradual()) ||
		   (aReferenceTransition.isShortGradual() && bestMatch.isCut()) ||
		   (aReferenceTransition.isShortGradual() && bestMatch.isShortGradual())){				
		    cutMatchSection += nl + "<match>" + nl + aReferenceTransition.refToString() +
			bestMatch.sysToString() +"</match>";
		}
		else{				
		    frameNumCount += bestMatch.intersection(aReferenceTransition).length();
		    frameRecallDenCount += aReferenceTransition.length();
		    framePrecisionDenCount += bestMatch.length();
		    
		    gradualMatchSection += nl + "<match fr=\"" + 
			comparisonResult.trimPrecision(new Double(bestMatchFrameRecall))+
			"\" fp=\"" + 
			comparisonResult.trimPrecision(new Double(bestMatchFramePrecision)) + "\">" + nl +
			aReferenceTransition.refToString() +
			bestMatch.sysToString() + 				    
			"</match>";
		}
	    }
	    // RE INIT matching info
	    aReferenceTransition.matched = false;
	}
	// Enumeration of system transitions to fill in the insertion section
	Iterator systemIterator = systemSegmentation.getElements();
	while(systemIterator.hasNext()){
	    Transition aSystemTransition = (Transition) systemIterator.next();
	    if(!aSystemTransition.matched){
		insertionsSection += aSystemTransition.sysToString();
		insertedTransCount++;
		if(aSystemTransition.isCut() || aSystemTransition.isShortGradual()) insertedTransCountCut++;
		else insertedTransCountGradual++;
	    }
	}		
	insertionsSection += nl+"</insertion>"+nl;
	deletionsSection += nl+"</deletion>"+nl;
	
	result.setResults(systemSegmentation.droppedToString(),
			  this.getTransitionCount(),
			  insertedTransCount,
			  deletedTransCount, 
			  cutMatchSection,
			  this.getCutCount(),
			  insertedTransCountCut,
			  deletedTransCountCut,
			  gradualMatchSection,
			  insertedTransCountGradual,
			  deletedTransCountGradual,
			  frameNumCount,
			  frameRecallDenCount,
			  framePrecisionDenCount,
			  insertionsSection,
			  deletionsSection);
	return result;
    }

    /**
     * Returns the number of Cut transitions within the segmentation.
     *
     * @return     the number of Cut transitions within the segmentation.
     */	
    public int getCutCount(){
	int toReturn = 0;
	Iterator i = this.transitionSet.iterator();
	while(i.hasNext()){
	    Transition t = (Transition) i.next();
	    if(t.isCut() || t.isShortGradual()) toReturn++;
	}
	return toReturn;
    }

    /**
     * Returns an iterator on the transitions within the segmentation.
     *
     * @return     an iterator on the transitions within the segmentation.
     */	
     public Iterator getElements(){
	 return this.transitionSet.iterator();
     }
    /**
     * Add a given transition to the segmentation.
     *
     * @param     aTransition the transition to be added to the segmentation.
     */
    public void add(Transition aTransition){ 
	boolean foundInDropped = false;
	Enumeration enumDropped = this.dropped.elements();
	while(enumDropped.hasMoreElements() && !foundInDropped){
	    Transition t = (Transition) enumDropped.nextElement();
	    if(t.intersection(aTransition) != null)
		foundInDropped = true;
	}
	if(!transitionSet.contains(aTransition) && !foundInDropped){
	    transitionSet.add(aTransition);
	}
	else{
	    this.dropped.add(aTransition);
	    //On the way to find the removed transition that intersects
	    //Only justification to keep TreeSet structure is for the ordering in iteration
	    //in processing comparison so in order to get ordered output files
	    //Again: iteration over the treeSet ! after the ones in contains() and add() !!
	    Transition movedToDropped = null;
	    Iterator setIterator = this.transitionSet.iterator();boolean found = false;
	    while(setIterator.hasNext() && !found){
		Transition t = (Transition) setIterator.next();
		if(t.intersection(aTransition) != null){
		    movedToDropped = t;
		    found = true;
		}
	    }
	    if(movedToDropped != null){ this.dropped.add(movedToDropped);}
	    this.transitionSet.remove(aTransition);	    
	}
    }
    /**
     * Returns the video name the segmentation is related to.
     *
     * @return     the video name the segmentation is related to.
     */	
    public String getVideoName(){ 
	return videoName;
    }
    /**
     * Returns the number of transitions within the segmentation.
     *
     * @return     the number of transitions within the segmentation.
     */	
    public int getTransitionCount(){ 
	return transitionSet.size();
    }
    public int getDroppedCount(){
	return this.dropped.size();
    }
    public String droppedToString(){

	String nl = System.getProperty("line.separator");
	StringBuffer toReturn = new StringBuffer();
	Enumeration enumDropped = this.dropped.elements();
	while(enumDropped.hasMoreElements()){
	    Transition s = (Transition) enumDropped.nextElement();
	    toReturn.append(s.toString("droppedTrans"));
	}
	return toReturn.toString();
    }
    /**
     * Returns an Xml string representing the segmentation in a pre-defined format.
     *
     * @return     an Xml string representing the segmentation.
     */	
    public String toString(){
	String toReturn = "<seg src=\""+videoName+"\">"+System.getProperty("line.separator");
	Iterator e = transitionSet.iterator();
	while(e.hasNext()){
	    Transition s = (Transition) e.next();
	    toReturn += s.toString();
	}
	toReturn += "</seg>"+System.getProperty("line.separator");
	return toReturn;
    }
}//END CLASS
