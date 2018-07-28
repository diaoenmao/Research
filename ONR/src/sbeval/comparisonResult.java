import java.io.*;

/**
 * This class represents the result of a comparison between a system result for one video and the reference.
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
public class comparisonResult {

    /**
     * The output Xml file.
     *     
     */
    private File outputXmlFile = null;
    /**
     * The file writer.
     *     
     */
    private FileWriter out = null;
	
    /**
     * The system ID for the one the comparison result is created.
     *     
     */
    public String sysId = null;
    /**
     * The video name for the one the comparison result is created.
     *     
     */
    public String videoName = null;
    /**
     * The total frame count in the related video.
     *     
     */
    public double totalFrameCount = 0;
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
     * The number of matching reference transition
     *
     */
    public int referenceMatchingCount = 0;

    /**
     * The number of matching cut reference transition
     * po
     */
    public int referenceCutMatchingCount = 0;

    /**
     * The number of matching gradual reference transition
     * po
     */
    public int referenceGradMatchingCount = 0;

    /**
     * The number of matching system transitions
     *
     */
    public int systemMatchingCount = 0; 

    /**
     * The number of matching cut system transitions
     * po
     */
    public int systemCutMatchingCount = 0; 

    /**
     * The number of matching gradual system transitions
     * po
     */
    public int systemGradMatchingCount = 0; 

    /**
     * Constructs comparisonResult object with the given parameters.
     * It creates a directory and a file for the storage of the results.
     *
     * @param     sysId The system ID for the one the comparison's been done.
     * @param     videoName The video name.
     * @param     totalFrameCount The total frame count in the given video.
     */
	public comparisonResult(String sysId, String videoName, int totalFrameCount){
		
		File sysIdDir = new File(System.getProperty("user.dir")+System.getProperty("file.separator")+
								 "ComparisonResult"+System.getProperty("file.separator")+
								 sysId+System.getProperty("file.separator"));
		sysIdDir.mkdir();
		this.outputXmlFile = new File(sysIdDir, sysId+"_"+videoName.substring(0, videoName.indexOf("."))+".xml");
		try{
			this.out = new FileWriter(outputXmlFile);
		}
		catch(IOException i){
			System.out.println("A problem occured while preparing to write results");
		}
		this.totalFrameCount = (double) totalFrameCount;
		this.sysId = sysId;
		this.videoName = videoName;
	}
	
    /**
     * Given the parameter, this method writes in the file the results.
     *
     * @param     refTransCount The transitions count in the reference for the related video.
     * @param     insertedTransCount The inserted transitions count for the comparison.
     * @param     deletedTransCount The deleted transitions count for the comparison.
     * @param     cutMatchSection A ready to write XML formatted section for the CUT matching transition results.
     * @param     cutTransCount The CUT transitions count in the reference for the related video.
     * @param     insertedTransCountCut The inserted CUT transitions count for the comparison.
     * @param     deletedTransCountCut The deleted CUT transitions count for the comparison.
     * @param     gradualMatchSection A ready to write XML formatted section for the GRADUAL matching transition results.
     * @param     insertedTransCountGradual The inserted GRADUAL transitions count for the comparison.
     * @param     deletedTransCountGradual The deleted GRADUAL transitions count for the comparison.
     * @param     insertionsSection A ready to write XML formatted section for the insertion results.
     * @param     deletionsSection A ready to write XML formatted section for the deletion results.
     */
    public void setResults(String droppedSection,
			   int refTransCount,
			   int insertedTransCount, 
			   int deletedTransCount,
			   String cutMatchSection,
			   int cutTransCount,
			   int insertedTransCountCut,
			   int deletedTransCountCut,
			   String gradualMatchSection,
			   int insertedTransCountGradual,
			   int deletedTransCountGradual,
			   int frameNumCount,
			   int frameRecallDenCount,
			   int framePrecisionDenCount,
			   String insertionsSection,
			   String deletionsSection){
	
	try{
	    String nl = System.getProperty("line.separator");
	    this.referenceMatchingCount = refTransCount-deletedTransCount;

	    this.referenceCutMatchingCount = cutTransCount-deletedTransCountCut;
	    this.referenceGradMatchingCount = this.referenceMatchingCount-this.referenceCutMatchingCount;

	    this.systemMatchingCount = refTransCount - deletedTransCount;
	    this.systemCutMatchingCount = cutTransCount - deletedTransCountCut;
	    this.systemGradMatchingCount = this.systemMatchingCount - this.systemCutMatchingCount;
	    //this.systemMatchingCount = refTransCount - deletedTransCount + insertedTransCount;

	    out.write("<?xml version=\"1.0\" encoding='ISO-8859-1'?>"+nl);
	    out.write("<!DOCTYPE comparisonResult>"+nl);
	    out.write("<comparisonResult sysId=\""+this.sysId+"\" "+nl);
	    out.write("                  src=\""+this.videoName+"\" "+nl);
	    out.write("                  totalFrameCount=\""+(int)totalFrameCount+"\" "+nl);
	    
	    getXmlAttributes((double)refTransCount, (double)insertedTransCount, (double)deletedTransCount);
	    out.write(">");
	    //DROPPED section
	    out.write(nl+"<droppedTransitions>"+nl);
	    //System.out.println(droppedSection);
	    out.write(droppedSection);
	    out.write("</droppedTransitions>"+nl);

	    //CUT section
	    out.write(nl+"<cutMatching      shortGradualThreshold=\""+Transition.getShortGradualThreshold()+"\""+nl);			
	    getXmlAttributes(cutTransCount, insertedTransCountCut, deletedTransCountCut);
	    out.write(">");
	    out.write(cutMatchSection);
	    out.write(nl+"</cutMatching>"+nl);
	    
	    //GRADUAL section
	    out.write(nl+"<gradualMatching"+nl);
	    getXmlAttributes((double)refTransCount - cutTransCount, 
			     (double)insertedTransCountGradual, (double)deletedTransCountGradual);
	    out.write("                  meanFrameRecall=\""+frameNumCount+"/"+frameRecallDenCount+"\" "+nl);
	    out.write("                  meanFramePrecision=\""+frameNumCount+"/"+framePrecisionDenCount+"\" >"+nl);
	    
	    this.gradualFrameCount = frameNumCount;
	    this.gradualRecallFrameCount = frameRecallDenCount;
	    this.gradualPrecisionFrameCount = framePrecisionDenCount;

	    out.write(gradualMatchSection);
	    out.write(nl+"</gradualMatching>"+nl);
	    
	    
	    //INSERTION section
	    out.write(nl);
	    out.write(insertionsSection);
	    
	    //DELETION section
	    out.write(nl);
	    out.write(deletionsSection);
	    
	    out.write(nl+"</comparisonResult>");
	    out.close();
	} catch(IOException i){
	    i.printStackTrace();
	    System.out.println("A problem occured while writing results");
	}
    }	
	
    /**
     * Private method that computes and writes in file a part of header results for a section.
     * For example the attributes of the cut mathcing section's Xml element:
     * <cutMatching      gradualConfusion="true"
     *                   confusionParameter="0.199999"
     *                   referenceTransCount="127" 
     *                   insertedTransCount="34" 
     *                   deletedTransCount="1" 
     *                   correctionRate="0.992" 
     *                   deletionRate="0.007" 
     *                   insertionRate="0.267" 
     *                   errorRate="0.275" 
     *                   qualityIndex="0.902" 
     *                   correctionProbability="0.995" 
     *                   recall="0.992" 
     *                   precision="0.787" />
     * It starts writing attributes begining at "referenceTransCount" Xml attribute.
     * 
     * @param     transitionCount The inserted transitions count for the comparison.
     * @param     insertedTransCount The deleted transitions count for the comparison.
     * @param     deletedTransCount The deleted transitions count for the comparison.
     */
    private void getXmlAttributes(double transitionCount, 
				  double insertedTransCount, 
				  double deletedTransCount){
	
	try{
	    String nl = System.getProperty("line.separator");
	    
	    out.write("                  referenceTransCount=\""+(int)transitionCount+"\" "+nl);
	    out.write("                  insertedTransCount=\""+(int)insertedTransCount+"\" "+nl);
	    out.write("                  deletedTransCount=\""+(int)deletedTransCount+"\" "+nl);
	    if(transitionCount != 0.0){
		out.write("                  correctionRate=\""+
			  trimPrecision(new Double((transitionCount-deletedTransCount)/transitionCount))
			  +"\" "+nl);
		out.write("                  deletionRate=\""+
			  trimPrecision(new Double(deletedTransCount/transitionCount))+
			  "\" "+nl);
		out.write("                  insertionRate=\""+
			  trimPrecision(new Double(insertedTransCount/transitionCount))+
			  "\" "+nl);
		out.write("                  errorRate=\""+
			  trimPrecision(new Double((deletedTransCount+insertedTransCount)/transitionCount))+
			  "\" "+nl);
		out.write("                  qualityIndex=\""+
			  trimPrecision(new Double((transitionCount-deletedTransCount-(insertedTransCount/3))/transitionCount))+
			  "\" "+nl);
		out.write("                  correctionProbability=\""+
			  trimPrecision(new Double(1-(insertedTransCount/
						      (totalFrameCount-transitionCount) + deletedTransCount/transitionCount)/2))+
			  "\" "+nl);
		out.write("                  recall=\""+
			  trimPrecision(new Double((transitionCount-deletedTransCount)/transitionCount))+
			  "\" "+nl);
		if((transitionCount-deletedTransCount+insertedTransCount != 0) &&
		   (transitionCount-deletedTransCount != 0))
		    out.write("                  precision=\""+ 			      
			      trimPrecision(new Double((transitionCount-deletedTransCount)/
						       (transitionCount-deletedTransCount+insertedTransCount)))+
			      "\""+nl);
		else
		    out.write("                  precision=\"0.0\""+nl);
	    }
	    else{
		out.write("                  correctionRate=\"0.0\" "+nl);
		out.write("                  deletionRate=\"0.0\" "+nl);
		out.write("                  insertionRate=\"0.0\" "+nl);
		out.write("                  errorRate=\"0.0\" "+nl);
		out.write("                  qualityIndex=\"0.0\" "+nl);
		out.write("                  correctionProbability=\"0.0\" "+nl);
		out.write("                  recall=\"0.0\" "+nl);
		if(transitionCount-deletedTransCount+insertedTransCount != 0)
		    out.write("                  precision=\""+
			      trimPrecision(new Double((transitionCount-deletedTransCount)/
						       (transitionCount-deletedTransCount+insertedTransCount)))+
			      nl);
		else
		    out.write("                  precision=\"0.0\""+nl);
	    }
	} catch(IOException i){
	    i.printStackTrace();
	    System.out.println("A problem occured while writing results");
	}
    }
    /**
     * Private method that only trim precision of a double until the first 3 decimals and returns it as a String.
     * 
     */
    public static String trimPrecision(Double d){
	
		String toReturn = d.toString();
		if(toReturn.length() >= 6)
			toReturn = toReturn.substring(0,5);
		return toReturn;		
	}
}
