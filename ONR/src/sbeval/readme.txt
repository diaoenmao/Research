
This software was produced by NIST, an agency of the U.S. government,
and by statute is not subject to copyright in the United States.
Recipients of this software assume all responsibilities associated
with its operation, modification and maintenance.

Running the Shot Boundary evaluation software

0) Create a directory called ComparisonResult, this will hold the results.

1) Compile the java files using javac.
   javac ComparisonManager.java (and similarly for the others)
   If javac (or java) is not found, modify PATH and CLASSPATH variables 
   (on a Windows machine, from MyComputer/Properties/Advanced), assuming 
   Java is installed in the system).

2) Locate the directory with the reference (i.e. groundtruth) files, refDir.
   Place the schema files shotBoundaryReferenceFiles.dtd and 
   shotBoundaryReferenceSegmentation.dtd in this directory.

3) Locate the file with the run(s) to be evaluated, runDir.
   Place the schema file shotBoundaryResults.dtd in this directory.

4) Run the program as follows:

   java ComparisonManager runDir/runFile.xml -referenceDir/refDir
   
   For example:
   
   java ComparisonManager svmClassifierRun.xml -referenceDir y:\TRECVID_2004\sbd_test_dvd\sbref04

   Alternatively, a threshold may be specified for short gradual transitions.
   If this threshold is equal to 5, then gradual transitions with less than 5
   frames will be treated as cuts by the ComparisonManager. This usage is:
   
   java ComparisonManager svmClassifierRun.xml -shortGradual5 -referenceDiry:\TRECVID_2004\sbd_test_dvd\sbref04

For any questions please contact Paul Over at over@nist.gov.
