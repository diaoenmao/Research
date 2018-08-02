import java.util.*;
import java.io.*;
import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;

/**
 * A class that extends DefaultHandler classes of Java Xml SAX API.
 * This class is used to get Xml data from the Xml parser and 
 * return them in defined formats (Hastable, Segmentation instances...).
 *
 * @author
 * This software was produced by NIST, an agency of the
 * U.S. government, and by statute is not subject to copyright in the
 * United States. Recipients of this software assume all
 * responsibilities associated with its operation, modification and
 * maintenance.
 *
 */


public class shotResultContentHandler extends DefaultHandler {

    static private Writer  out;
    /**
     * The Hashtable used to store the system results,
     * the key is the system ID, and value is a systemResult instance.
     *
     * @see #getSystemResults()
     */
	private Hashtable systemResults = new Hashtable();
	/**
     * Segmentation instance used to get the current Segmentation the parser is working on.
     * The transitions of this segmentation will be added as and when the parser calls startsElement() method
     * and the Xml element "<trans ...>" and its attributes are recognized.
     *
     */
	private Segmentation currentSeg = null;
	/**
     * systemResult instance used to get the current system results the parser is working on.
     * The segmentations of this system result will be added when the parser calls startsElement() method
     * and the Xml element "<seg ...>" and its attributes are recognized.
     *
     */
	private systemResult currentSys = null;
	/**
     * Hashtable that contains the reference file names stored in an Xml file.
     * The key is the video name and value is reference file name.
     */
	private Hashtable referenceFileNames = new Hashtable();
	
	/**
     * Constructs an instance of this class with the given Writer as the output
     * writer used to get information during parsing...
     * 
     */
	public shotResultContentHandler(Writer out){ this.out = out; }
	/**
     * Returns the systemResults parsed and contained in an Hashtable.     
     * 
     */
	public Hashtable getSystemResults(){
		return systemResults;
	}
	/**
     * Returns the reference file namesparsed and contained in an Hashtable.     
     * 
     */
	public Hashtable getReferenceFileNames(){
		return referenceFileNames;	
	}
	
	/**
     * Returns the reference segmentation parsed from one Xml file per segmentation in the reference.
     * Actually currentSeg is returned cause it contains either the current parsed reference 
     * segmentation or a segmentation of a system.
     */
	public Segmentation getRefSegmentation(){
		return currentSeg;	
	}
	/**
     * SAX DocumentHandler method.
     * 
     */
    public void setDocumentLocator(Locator l)
    {
        // Save this to resolve relative URIs or to give diagnostics.
        try {
          //out.write("LOCATOR");
	    out.write("    "+l.getSystemId()+System.getProperty("line.separator"));
	    out.flush();
        } catch (IOException e) {
            // Ignore errors
        }
    }

	/**
     * SAX DocumentHandler method.
     * 
     */
    public void startDocument() throws SAXException{
        //nl();
    }

	/**
     * SAX DocumentHandler method.
     * 
     */
    public void endDocument() throws SAXException {
        //emit("END DOCUMENT");
        try {
            //nl();
            out.flush();
        } catch (IOException e) {
            throw new SAXException("I/O error", e);
        }
    }

    /**
     * SAX DocumentHandler method used to recognized Xml tags and get the information in the required Hashtable
     * or any data structure for later use.
     * 
     */
    public void startElement(String namespaceURI,
                             String lName, // local name
                             String qName, // qualified name
                             Attributes attrs) throws SAXException {
        //indentLevel++;
        String eName = lName; // element name
        if ("".equals(eName)) eName = qName; // namespaceAware = false
        

	//Case
		
	if(eName.equals("refSeg")){
	    String src = null;
	    int totalFNumber = 0;
	    if (attrs != null) {
		for (int i = 0; i < attrs.getLength(); i++) {
		    
		    String aName = attrs.getLocalName(i); // Attr name 
		    if ("".equals(aName)) aName = attrs.getQName(i);
		    if(aName.equals("src")) src = attrs.getValue(i).toLowerCase();
		    if(aName.equals("totalFNum")) totalFNumber = Integer.parseInt(attrs.getValue(i));
		}
	    }
	    currentSeg = new Segmentation(src, IntervalComparator.getInstance(), totalFNumber);
	}
	
	else if(eName.equals("referenceFileName")){
	    String videoName = null;
	    String segName = null;
	    if (attrs != null) {
		for (int i = 0; i < attrs.getLength(); i++) {
		    
		    String aName = attrs.getLocalName(i); // Attr name 
		    if ("".equals(aName)) aName = attrs.getQName(i);
		    if(aName.equals("videoName")) videoName = attrs.getValue(i).toLowerCase();
		    if(aName.equals("segName")) segName = attrs.getValue(i);
		}
	    }
	    referenceFileNames.put(videoName, segName);
	}
	else if(eName.equals("trans")){
	    String type = "";
	    int pre = 0;
	    int post = 0;
	    try{
		if (attrs != null) {
		    for (int i = 0; i < attrs.getLength(); i++) {
			
			String aName = attrs.getLocalName(i); // Attr name 
			if ("".equals(aName)) aName = attrs.getQName(i);
			if(aName.equals("type")) type = attrs.getValue(i);
			if(aName.equals("preFNum")){
			    pre = Integer.parseInt((String) attrs.getValue(i));
			}
			if(aName.equals("postFNum")) post = Integer.parseInt(attrs.getValue(i));
		    }
		}
		//parsing du type cut
		if(pre == (post-1))
		    type = "CUT";
		Transition t = new Transition(pre, post ,type);
		currentSeg.add(t);
	    } catch (TransitionDefinitionException e){
		throw new SAXException("Error in transition definition: "+pre+" "+post);
	    } catch (Exception e){
		throw new SAXException("Error in transition reading or processing");
	    }
	}
	else if(eName.equals("seg")){
	    String fileName = null;
	    if (attrs != null) {
            	for (int i = 0; i < attrs.getLength(); i++) {
		    
		    String aName = attrs.getLocalName(i); // Attr name 
		    if ("".equals(aName)) aName = attrs.getQName(i);
		    if(aName.equals("src")) fileName = attrs.getValue(i).toLowerCase();
		}
	    }
	    currentSeg = new Segmentation(fileName, IntervalComparator.getInstance());
	    currentSys.addSegmentation(currentSeg);
	}
	else if(eName.equals("shotBoundaryResult")){
	    
	    String sysId = null;
	    if (attrs != null) {
            	for (int i = 0; i < attrs.getLength(); i++) {
		    
		    String aName = attrs.getLocalName(i); // Attr name 
		    if ("".equals(aName)) aName = attrs.getQName(i);
		    if(aName.equals("sysId")) sysId = attrs.getValue(i);
		}
	    }
	    currentSys = new systemResult(sysId);
	    systemResults.put(sysId, currentSys);
	}
	else if(eName.equals("shotBoundaryResults")){
	    
	}	
    }
    /**
     * SAX DocumentHandler method.
     * 
     */
    public void endElement(String namespaceURI,
                           String sName, // simple name
                           String qName  // qualified name
			   ) throws SAXException {
    }
    /**
     * SAX DocumentHandler method.
     * 
     */
    public void characters(char buf[], int offset, int len) throws SAXException {
        //nl(); 
        emit("CHARS:   ");
        String s = new String(buf, offset, len);
        if (!s.trim().equals("")) emit(s);
    }
    /**
     * SAX DocumentHandler method.
     * 
     */
    public void ignorableWhitespace(char buf[], int offset, int len) throws SAXException {
        // Ignore it
    }
    /**
     * SAX DocumentHandler method.
     * 
     */
    public void processingInstruction(String target, String data) throws SAXException {
        //nl();
        emit("PROCESS: ");
        emit("<?"+target+" "+data+"?>");
    }
	/**
     * SAX ErrorHandler method. Treats validation errors as fatal.
     * 
     */
    public void error(SAXParseException e) throws SAXParseException {
        throw e;
    }
	/**
     * SAX ErrorHandler method. Dumps warnings.
     * 
     */
    public void warning(SAXParseException err) throws SAXParseException {
        System.out.println("** Warning"
            + ", line " + err.getLineNumber()
            + ", uri " + err.getSystemId());
        System.out.println("   " + err.getMessage());
    }

    //===========================================================
    // Utility Methods ...
    //===========================================================

	/**
     * Wrap I/O exceptions in SAX exceptions, to suit handler signature 
     * requirements.
     * 
     */
    private void emit(String s) throws SAXException {
        try {
            out.write(s);
            out.flush();
        } catch (IOException e) {
            throw new SAXException("I/O error", e);
        }
    }
	/**
     * Start a new line and indent the next line appropriately.
     * 
     */
    private void nl() throws SAXException {
        String lineEnd =  System.getProperty("line.separator");
        try {
            out.write(lineEnd);
            //for (int i=0; i < indentLevel; i++) out.write(indentString);
        } catch (IOException e) {
            throw new SAXException("I/O error", e);
        }
    }
}
