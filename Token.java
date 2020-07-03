class Token {

  
  // NEW VALUES TO PARSE NEWSDATA FILE
  public final static int ERROR = 0;
  public final static int NUM = 1;
  public final static int WORD = 2;
  public final static int APOSTROPHIZED = 3;
  public final static int HYPHENATED = 4;
  public final static int PUNCTUATION = 5;
  public final static int OPEN_TAG = 6;
  public final static int CLOSE_TAG = 7;
  public final static int white = 8;
  public final static int DOC_ID = 9;
  public static int wordsPerLine = 0;
  


  public int m_type;
  public String m_value;
  public int m_line;
  public int m_column;
  
  Token (int type, String value, int line, int column) {
    m_type = type;
    m_value = value;
    m_line = line;
    m_column = column;
  }



  public String toString() {
    switch (m_type) {
 
      case ERROR:
	return "ERROR(" + m_value + ")";

      case NUM:
	return "NUM(" + m_value + ")";

      case WORD:
	int word = testLineLength();
	return 	m_value.toLowerCase();

      case APOSTROPHIZED:
	String[] strApost = canSplitApost(m_value);
	int word1 = testLineLength();
    	if (strApost.length == 1) {
           return  strApost[0].toLowerCase();
	}
	else {
	   for (int i = 0; i < strApost.length - 1; i++) {
	      System.out.print(" " + strApost[i].toLowerCase());
	      int word11 = testLineLength();
	   }
	   return " " + strApost[strApost.length - 1].toLowerCase();
	}

      case HYPHENATED:
	String[] strHyphen = canSplitHyphen(m_value);
	int word2 = testLineLength();
	if (strHyphen.length == 1) {
	   return strHyphen[0].toLowerCase();
	}
	else {
	   for (int i = 0; i < strHyphen.length - 1; i++) {
	      System.out.print(" " + strHyphen[i].toLowerCase());
	      int word22 = testLineLength();
	   }
	   return " " + strHyphen[strHyphen.length - 1].toLowerCase();
        }

      case PUNCTUATION:
	return "PUNCTUATION(" +m_value + ")";

      case OPEN_TAG:
	String mCheck = m_value.substring(1, m_value.length() - 1);
	if (mCheck.equals("DOC")) {
	   return "$" + mCheck;
	}
	else if (mCheck.equals("HEADLINE")) {	
	   return "$TITLE";
	}
	if (mCheck.equals("TEXT")) {
	   return "$BODY";
	}
	return "OPEN-" + mCheck;

      case CLOSE_TAG:
        return "CLOSE-";

      case DOC_ID:
        return m_value;
      default:
        return "()";

    }
  }


  /* test to prevent unnecessary words matched into apostrophized tokens */
  public String[] canSplitApost(String testStr){
	  String[] strArray = testStr.split("'");
	  int flagger = 0;
	  for (int i = 0; i < strArray.length; i++){
	     if (strArray[i].length() > 2)
	        flagger++;
	  }
	  if (flagger == strArray.length)
	     return strArray;
	  else {
	     String[] strNoChange = { testStr };
	     return strNoChange;
	  }
  }


  /* test to prevent unnecessary words matched into hyphenated tokens */
  public String[] canSplitHyphen(String testStr) {
	  String[] strArray = testStr.split("-");
	  int flagger = 0;
	  for (int i = 0; i < strArray.length; i++) {
             if (strArray[i].length() > 2)
	        flagger++;
	  }
	  if (flagger == strArray.length)
	     return strArray;
	  else {
             String[] strNoChange = { testStr };
	     return strNoChange;
	  }
  }

   /* function set to make sure number of words per line is 15 */
   public int testLineLength(){
	  wordsPerLine++;
	  if (wordsPerLine % 15 == 0){
		  System.out.println();
		  return 1;
	  }
	  else
		  return 0;
  } 
		
}

