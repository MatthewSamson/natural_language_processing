/* import packages */
import java.util.Scanner;
import java.util.ArrayList;
import java.util.*;
import java.io.*;

public class Offline{
	
	public static void main(String[] args){
		File my_file = new File("preprocess.txt");
		
		/* treemap data structures */
		TreeMap<String, ArrayList<listObj>> myMap = new TreeMap<String, ArrayList<listObj>>();
		
		/* docids.txt data structures */
		ArrayList<String> docIDList = new ArrayList<String>();
		ArrayList<Integer> docStartLine = new ArrayList<Integer>();
		ArrayList<String> docTitle = new ArrayList<String>();
		
        int lineCount = 0;
        String takeDocID = "";
        String takeTitle = "";
        String word = ""; 
        
        try{
			Scanner scan = new Scanner(my_file);
			while (scan.hasNextLine() == true){
				
				String[] is = scan.nextLine().split(" ");
				
				if (is[0].equals("$DOC")){
					takeDocID = is[1];
					docIDList.add(takeDocID);
					docStartLine.add(lineCount);
				}
				
				if (is[0].equals("$TITLE")){
					takeTitle = scan.nextLine();
					docTitle.add(takeTitle);
				}
				
				if ((!is[0].equals("$DOC")) && (!is[0].equals("$BODY")) && (!is[0].equals("$TITLE"))){
					
					for (int i = 0; i < is.length; i++){
						word = is[i];
						ArrayList<listObj> myList = new ArrayList<listObj>();	
						
						if (!myMap.containsKey(word)){
							listObj obj = new listObj(takeDocID);				
							myList.add(obj);
							myMap.put(word, myList);
						}
						
						else{	
							myList = myMap.get(word);
							
							for (int j = 0; j < myList.size(); j++){
								listObj obj0 = new listObj(takeDocID);
								obj0 = myList.get(j);
								String docName = obj0.getDocID();
								
								if (docName.equals(takeDocID)){
									obj0.incrementFreq();
								}
								
								else{
									listObj newObj = new listObj(takeDocID, 1);
									myList.add(newObj);
									break;
								}
							}
							
						}	
					}
				} 
				lineCount++;
			}	 
		}
        catch (FileNotFoundException e){
           e.printStackTrace();
        }
        writeToDictionary(myMap);
        writeToPostings(myMap);
        writeToDocids(docIDList, docStartLine, docTitle);
	}
	
	
	
	/* functions section */
	
	/* function to write to dictionary.txt */
  
	public static void writeToDictionary(TreeMap<String, ArrayList<listObj>> t1){
		
		int treeSize = t1.size() - 1;
		ArrayList<listObj> docList = new ArrayList<listObj>();
		ArrayList<Integer> termFreq = new ArrayList<Integer>();
		ArrayList<String> treeKeys = new ArrayList<String>();
		
		
		Set<String> keys = t1.keySet();
		
        for(String key: keys){
			treeKeys.add(key);
            docList = t1.get(key);
            termFreq.add(docList.size());
			
		} 
		
		String[] t1Keys = treeKeys.toArray(new String[treeKeys.size()]);
		Integer[] tf = termFreq.toArray(new Integer[termFreq.size()]);
		
		try{
			PrintWriter writer1 = new PrintWriter("dictionary.txt", "UTF-8");
			writer1.println("" + treeSize);
			for (int i = 1; i < t1Keys.length; i++){
				writer1.print("" + t1Keys[i] + "\t");
				writer1.println("" + tf[i]);
			}
			writer1.close();
		}
		
		catch(Exception e){
			e.printStackTrace();
		}
	}

	
	/* function to write to postings.txt */
	public static void writeToPostings (TreeMap<String, ArrayList<listObj>> t2){
		
		ArrayList<listObj> objList = new ArrayList<listObj>();
		ArrayList<Integer> docFreq = new ArrayList<Integer>();
		ArrayList<String> docIDs = new ArrayList<String>();
		ArrayList<String> treeKeys = new ArrayList<String>();
		
		int entryCounter = 1;
		
		Set<String> keys = t2.keySet();
		
        for(String key: keys){
			treeKeys.add(key);
            objList = t2.get(key);
            docFreq.add(objList.size());
            
			for (int x = 0; x < objList.size(); x++){
				listObj myObj = objList.get(x);	
				docIDs.add(myObj.getDocID());
				docFreq.add(myObj.getWordFreq());
				entryCounter++;
			}
		} 
		
		
		entryCounter--;
		String[] IDArray = docIDs.toArray(new String[docIDs.size()]);
		Integer[] df = docFreq.toArray(new Integer[docFreq.size()]);
		
		try{
			PrintWriter writer2 = new PrintWriter("postings.txt", "UTF-8");
			writer2.println("" + entryCounter);
			for (int i = 0; i < docIDs.size(); i++){
				writer2.print("" + IDArray[i] + "\t");
				writer2.println("" + df[i]);
				//System.out.println(tf[i]);
			}
			writer2.close();
		}
		
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
		
	/* function to write to docids.txt */
	public static void writeToDocids(ArrayList<String> s1, ArrayList<Integer> i1, ArrayList<String> s2){
		String[] s1print = s1.toArray(new String[s1.size()]);
		Integer[] i1print = i1.toArray(new Integer[i1.size()]);
		String[] s2print = s2.toArray(new String[s1.size()]);
		
		try{
			PrintWriter writer3 = new PrintWriter("docids.txt", "UTF-8");
			writer3.println("" + s1print.length);
			for (int i = 0; i < s1.size(); i++){
				writer3.print("" + s1print[i] + "\t");
				writer3.print("" + i1print[i] + "\t");
				writer3.println("" + s2print[i]);
			}
			writer3.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
			
	 
}


