import java.lang.Math;

/**
 * This class represents a shot transition.
 * A shot boundary or transition is defined by its pre frame and its post frame. 
 * These frames are respectively the last and first frame before and after the 
 * beginning of the transition. The transition could be a short one 
 * (CUT) with its post right after the pre frame or 
 * a gradual one with at least one frame within the transition effect.
 *
 * @author
 * This software was produced by NIST, an agency of the
 * U.S. government, and by statute is not subject to copyright in the
 * United States. Recipients of this software assume all
 * responsibilities associated with its operation, modification and
 * maintenance.
 *
 */

public class Transition implements Cloneable{

    private final static String[] types = {"CUT", "GRAD"};
    private static int shortGradualThreshold = 5;
    /**
     * The pre frame: the last frame of a previous shot without any content of the transition effect.
     *
     * @see #getPre()
     */
    public int pre;
    /**
     * The post frame: the first frame of the following shot without any content of the transition effect.
     *
     * @see #getPost()
     */
    public int post;
    /**
     * The type of transition: could take the value "cut" and anything else stands for a gradual transition.
     *
     */
    public String type;
    /**
     * Indicates wether the transition has matched with another one during a comparison.
     *
     */
    public boolean matched = false;
    public Object clone(){

	int apre = pre;
	int apost = post;
	//String type = type
	try{
	    Transition aClone = new Transition(apre, apost, type);
	return aClone;
	}catch(TransitionDefinitionException e){
	    System.out.println("CLONING PROBLEM");
	    return null;
	}
    }
    /**
     * Constructs Transition object with the given parameters.
     *
     * @param     pre The pre frame of the transition.
     * @param     post The post frame of the transition.
     * @param     type The type of trnasition.
     * @exception TransitionDefinitionException Thrown if the pre frame is not strictly less than the post frame.
     */
    public Transition(int pre, int post, String type)throws TransitionDefinitionException{
	if((pre >-1) && (post >-1) && (pre < post)){
	    this.pre = pre;
	    this.post = post;
	    if(type.compareToIgnoreCase("CUT") == 0)
		if(pre != post-1)
		    throw new TransitionDefinitionException("Transition definition error: CUT definition not respected");
		else
		    this.type = "CUT";
	    else
		this.type = type;
	}
	else throw new TransitionDefinitionException("Transition definition error: pre and post not well ordered");
    }
    /**
     * Set the threshold for short gradual and cut confusion
     *
     */
    public static void setShortGradualThreshold(int value){

	Transition.shortGradualThreshold = value;
    }
    /**
     * Get the threshold for short gradual and cut confusion
     *
     */
    public static int getShortGradualThreshold(){

	return shortGradualThreshold;
    }
    /**
     * Return true if the transition is a short gradual ie length less than
     * the threshold.
     *
     */
    public boolean isShortGradual(){

	return(this.type.compareToIgnoreCase("CUT") != 0 &&
	       this.length() <= Transition.shortGradualThreshold);
    }
    /**
     * Returns true if the transition is a CUT
     *
     */
    public boolean isCut(){

	return (this.type.compareToIgnoreCase("CUT") == 0);
    }
    /**
     * Returns true if the transition is a CUT
     *
     */
    public boolean isGradual(){

	return (this.type.compareToIgnoreCase("CUT") != 0 &&
		this.length() > Transition.shortGradualThreshold);
    }  
    /**
     * Gets the pre frame.
     *
     */
    public int getPre(){
	return this.pre;
    }
    /**
     * Gets the post frame.
     *
     */
    public int getPost(){
	return this.post;
    }
    /**
     * Gets the type.
     *
     */
    public String getType(){
	return this.type;
    }
    /**
     * Returns true if the given transition has the same type as the current instance 
     * regarding the setted confusions between types.
     *
     */
    public boolean sameType(Transition t){
	return ((this.isCut() && t.isCut()) ||
		(this.isGradual() && t.isGradual()) ||
		(this.isCut() && t.isShortGradual()) ||
		(this.isShortGradual() && t.isCut()) ||
		(this.isShortGradual() && t.isShortGradual()));
    }
    /**
     * Sets the pre frame value.
     *
     */
    public void setPre(int newPre) throws TransitionDefinitionException{
	if((newPre >-1) && (newPre < post)){
	    this.pre = newPre;
	}
	else throw new TransitionDefinitionException("Transition definition error: pre and post not well ordered");
    }
    /**
     * Sets the post frame value.
     *
     */
    public void setPost(int newPost) throws TransitionDefinitionException{
	if(newPost > pre){
	    this.post = newPost;
	}
	else throw new TransitionDefinitionException("Transition definition error: pre and post not well ordered");
    }
    /**
     * Sets the post frame value.
     *
     */
    public void reset(int newPre, int newPost) throws TransitionDefinitionException{
	if((newPre >-1) && (newPre < newPost)){
	    this.pre = newPre;
	    this.post = newPost;
	}
	else throw new TransitionDefinitionException("Transition definition error: pre and post not well ordered "+
						     newPre+" "+newPost);
    }
    /**
     * Returns the length of the transition (post - pre - 1).
     *
     */
    public int length(){
	return (this.post - this.pre -1);		
    }
    /**
     * Returns a transition representing the intersection with the given transition.
     *
     */
    public Transition intersection(Transition t){
	
	try{
	    return (new Transition(Math.max(this.pre, t.pre), 
				   Math.min(this.post, t.post), "INT"));
	}
	catch(Exception e){
	    return null;			
	}
    }
    /**
     * Returns true if there is an intersection with the given transition.
     *
     */
    public boolean intersected(Transition t){
	return (Math.max(this.pre, t.pre) < Math.min(this.post, t.post));
    }	
    /**
     * Returns an Xml style string of transition.
     *
     */
    public String toString(){
	return "<trans type=\""+this.type+"\" pre=\""+this.pre+"\" post=\""+this.post+"\"/>"+System.getProperty("line.separator");	
    }
    /**
     * Returns an Xml style string of transition.
     *
     */
    public String toString(String tag){
	return "<"+tag+" type=\""+this.type+"\" pre=\""+this.pre+"\" post=\""+this.post+"\"/>"+System.getProperty("line.separator");	
    }
    /**
     * Returns an Xml style string of transition with the Xml element name equal to "sysTrans".
     *
     */
    public String sysToString(){
	return "<sysTrans type=\""+this.type+"\" pre=\""+this.pre+"\" post=\""+this.post+"\"/>"+System.getProperty("line.separator");	
    }
    /**
     * Returns an Xml style string of transition with the Xml element name euqal to "refTrans".
     *
     */
	public String refToString(){
		return "<refTrans type=\""+this.type+"\" pre=\""+this.pre+"\" post=\""+this.post+"\"/>"+System.getProperty("line.separator");	
	}
	
}
