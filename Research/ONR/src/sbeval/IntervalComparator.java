import java.util.*;

/**
 * A class that implements the java.util.Comparator interface to compare two transitions regarding their limits.
 * It compares the transitions regarding their pre-frame and then if needed their post-frame.
 *
 * @author
 * This software was produced by NIST, an agency of the
 * U.S. government, and by statute is not subject to copyright in the
 * United States. Recipients of this software assume all
 * responsibilities associated with its operation, modification and
 * maintenance.
 *
 */

public class IntervalComparator implements Comparator{
    
    /**
     * The unique instance of this class.
     *
     * @see #getInstance()
     */
    private static IntervalComparator myInstance = null;
    /**
     * Private constructor.
     *
     */
    private IntervalComparator(){}
	/**
     * Returns the unique instance of this class.
     *
     */
    public static IntervalComparator getInstance(){
		if(myInstance == null)
	    	myInstance = new IntervalComparator();
		return myInstance;
    }    
	/**
     * Compares its two arguments for ordering them. Uses the pre-frame and if needed the post-frame to check the equality.
     *
     * @return    a negative integer, zero, or a positive integer as the first argument is less than, equal to, or greater than the second.
     * @exception ClassCastException Thrown if one of the arguments is not a transition instance.
     */
	public int compare(Object o1, Object o2) throws ClassCastException{

		if((o1 instanceof Transition) && (o2 instanceof Transition)){

			Transition t1 = (Transition)o1;
			Transition t2 = (Transition)o2;
			if(t1.intersection(t2) != null){
			    return 0;
			}
			else if(t1.pre < t2.pre)
			    return -1;
			else
			    return 1;
			/*if(t1.pre < t2.pre)
				return -1;
			else if(t1.pre == t2.pre){
				if(t1.post == t2.post)
					return 0;
				else if(t1.post < t2.post)
					return -1;
				else
					return 1;
			}
			else return 1;*/
		}
		else throw new ClassCastException("Instance Problem for IntervalComparator");
	}

}
