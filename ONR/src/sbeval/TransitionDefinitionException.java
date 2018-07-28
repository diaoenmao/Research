/**
 * Thrown when attempts to define a transition with uncorrect values.
 *
 * @author
 * This software was produced by NIST, an agency of the
 * U.S. government, and by statute is not subject to copyright in the
 * United States. Recipients of this software assume all
 * responsibilities associated with its operation, modification and
 * maintenance.
 *
 */
public class TransitionDefinitionException extends Exception{

	TransitionDefinitionException(String why){
		super(why);
	}
}