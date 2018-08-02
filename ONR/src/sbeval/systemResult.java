import java.util.*;
import java.io.*;

/**
 * This class represents the results for one system identified by its ID.
 * One system has several segmentation results corresponding to the run of this system on a video.
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
public class systemResult{

    /**
     * The segmentations for this system are stored in this Hashtable. 
     * The key is the name of the video the segmentation is done for.
     *     
     */
    private Hashtable segmentations = new Hashtable();
    /**
     * The system ID of the results.
     *     
     */
    private String sysId;
    /**
     * The number of frames of all the gradual-gradual intersections  
     *
     */
    public int gradualFrameCount = 0;
    /**
     * The number of frames of the matching reference gradual transitions  
     *
     */
    public int gradualRecallFrameCount = 0;
    /**
     * The number of frames of the matching system transitions
     *
     */
    public int gradualPrecisionFrameCount = 0; 
    /**
     * The number of reference matching transitions across all segmentations
     *
     */
    public int referenceMatchingCount = 0;

    /**
     * The number of reference matching cut transitions across all 
     * segmentations
     * po
     */
    public int referenceCutMatchingCount = 0;

    /**
     * The number of reference matching gradual transitions across all 
     * segmentations
     * po
     */
    public int referenceGradMatchingCount = 0;

    /**
     * The number of system matching transitions across all segmentations
     * 
     */
    public int systemMatchingCount = 0;

    /**
     * The number of system matching cut transitions across all segmentations
     * po
     */
    public int systemCutMatchingCount = 0;

    /**
     * The number of system matching gradual transitions across all 
     * segmentations
     * po
     */
    public int systemGradMatchingCount = 0;

    /**
     * The number of system transitions across all segmentations
     *
     */
    public int systemTransCount = 0;

    /**
     * The number of system cut transitions across all segmentations
     * po
     */
    public int systemCutTransCount = 0;

    /**
     * The number of system gradual transitions across all segmentations
     * po
     */
    public int systemGradTransCount = 0;

    /**
     * The number of reference transitions across all segmentations
     * 
     */
    public int referenceTransCount = 0;

    /** 
     * The number of reference cut transitions across all segmentations
     * po
     */
    public int referenceCutTransCount = 0;

    /**
     * The number of reference gradual transitions across all segmentations
     * po
     */
    public int referenceGradTransCount = 0;

    /**
     * Instanciate a systemResult object with only the given system ID.
     * 
     */	
    private int droppedCount = 0;

    public systemResult(String sysId){
	this.sysId = sysId;
    }
    /**
     * Add the given segmentation to the system results.
     * 
     */	
    public void addSegmentation(Segmentation s){
	if(s.getVideoName() == null){
	    System.out.println("PROBLEM");
	}
	segmentations.put(s.getVideoName(), s);
    }
    /**
     * Returns an enumeration of segmentations.
     * 
     */	
    public Enumeration getSegmentations(){
	return segmentations.elements();	
    }
    /**
     * Returns the system ID.
     * 
     */
    public String getSysId(){
	return this.sysId; 
    }
    /**
     * Returns a Vector containing video names of the segmentations.
     * 
     */	
    public Vector getSegmentationsVideoNames(){
	Vector toReturn = new Vector();
	Enumeration e = segmentations.elements();
	while(e.hasMoreElements()){		
	    Segmentation s = (Segmentation) e.nextElement();	
	    toReturn.add(s.getVideoName());
	}
	return toReturn;
    }
    /**
     * Process the comparison between the reference segmentations and each system segmentation.
     *     
     */
    public void processComparison(Hashtable referenceSegmentations) throws Exception{
	
	System.out.println(this.sysId);
	//For each Segmentation
	Enumeration n = this.getSegmentations();
	while(n.hasMoreElements()){
	    Segmentation aSystemSeg = (Segmentation) n.nextElement();
	    this.droppedCount += aSystemSeg.getDroppedCount();

	    Segmentation aRefSeg = (Segmentation) referenceSegmentations.get(aSystemSeg.getVideoName());
	    //We compare the two segmentations
	    //Creation of a comparisonResult
	    if(aRefSeg != null){
		comparisonResult aResult = aRefSeg.processComparison(this.getSysId(), aSystemSeg);
		this.gradualFrameCount += aResult.gradualFrameCount;
		this.gradualRecallFrameCount += aResult.gradualRecallFrameCount;
		this.gradualPrecisionFrameCount += aResult.gradualPrecisionFrameCount;
		
		this.referenceMatchingCount += aResult.referenceMatchingCount;
		this.referenceCutMatchingCount += aResult.referenceCutMatchingCount;
		this.referenceGradMatchingCount += aResult.referenceGradMatchingCount;

		this.systemMatchingCount += aResult.systemMatchingCount;
		this.systemCutMatchingCount += aResult.systemCutMatchingCount;
		this.systemGradMatchingCount += aResult.systemGradMatchingCount;

		this.systemTransCount += aSystemSeg.getTransitionCount();
		this.systemCutTransCount += aSystemSeg.getCutCount();
		this.systemGradTransCount += aSystemSeg.getTransitionCount() - aSystemSeg.getCutCount();

		this.referenceTransCount += aRefSeg.getTransitionCount();
		this.referenceCutTransCount += aRefSeg.getCutCount();
		this.referenceGradTransCount += aRefSeg.getTransitionCount() - aRefSeg.getCutCount();
	    }
	    else{
		System.out.println("    Warning: Unprocessed segmentation - No reference found for: "+
				   aSystemSeg.getVideoName());
	    }
	}
    }
    /**
     * Returns an Xml String of the system results including all segmentations.
     * 
     */		
    public String toString(){
	String toReturn = "<shotBoundaryResult sysId="+sysId+">"+System.getProperty("line.separator");
	Enumeration e = segmentations.elements();
	while(e.hasMoreElements()){
	    Segmentation s = (Segmentation) e.nextElement();
	    toReturn += s.toString();
	}
	toReturn += "</shotBoundaryResult>"+System.getProperty("line.separator");
	return toReturn;
    }
    /**
     * Write into a file the system level results
     *
     */
    public void save(){
    
	File sysIdDir = new File(System.getProperty("user.dir")+System.getProperty("file.separator")+
				 "ComparisonResult"+System.getProperty("file.separator")+
				 sysId+System.getProperty("file.separator"));
	sysIdDir.mkdir();
	File outputXmlFile = new File(sysIdDir, sysId+".xml");
	try{
	    FileWriter out = new FileWriter(outputXmlFile);
	    String nl = System.getProperty("line.separator");
	    out.write("<?xml version=\"1.0\" encoding='ISO-8859-1'?>"+nl);
	    out.write("<!DOCTYPE systemComparisonResult>"+nl);
	    out.write("<systemComparisonResult sysId=\""+this.sysId+"\""+nl);
	    out.write("                        droppedTransCount=\""+this.droppedCount+"\""+nl);
	    out.write("                        meanRecall=\""+this.referenceMatchingCount+"/"+this.referenceTransCount+"\""+nl);
	    out.write("                        meanPrecision=\""+this.systemMatchingCount+"/"+this.systemTransCount+"\""+nl);

	    out.write("                        meanCutRecall=\""+this.referenceCutMatchingCount+"/"+this.referenceCutTransCount+"\""+nl);
	    out.write("                        meanCutPrecision=\""+this.systemCutMatchingCount+"/"+this.systemCutTransCount+"\""+nl);
	    out.write("                        meanGradRecall=\""+this.referenceGradMatchingCount+"/"+this.referenceGradTransCount+"\""+nl);
	    out.write("                        meanGradPrecision=\""+this.systemGradMatchingCount+"/"+this.systemGradTransCount+"\""+nl);

	    out.write("                        meanGradFrameRecall=\""+this.gradualFrameCount+"/"+this.gradualRecallFrameCount+"\""+nl);
	    out.write("                        meanGradFramePrecision=\""+this.gradualFrameCount+"/"+this.gradualPrecisionFrameCount+"\" />");
	    out.close();
	}
	catch(IOException i){
	    System.out.println("A problem occured while preparing to write results");
	}
    }
}
