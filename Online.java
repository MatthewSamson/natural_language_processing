/* import section */
import java.io.File;
import java.io.*;
import java.util.Scanner;
import java.util.*;
import java.lang.Math.*;

public class Online{
	
	public static void main(String[] args){
		
		/* file read section */
		File dict = new File("dictionary.txt");
		File post = new File("postings.txt");
		File docI = new File("docids.txt");
		
		/* arraylists required */
		ArrayList<String> dictWords = new ArrayList<String>();
		ArrayList<Integer> dicttf = new ArrayList<Integer>();
		ArrayList<String> postID = new ArrayList<String>();
		ArrayList<Integer> postdf = new ArrayList<Integer>();
		ArrayList<String> docIIDs = new ArrayList<String>();
		ArrayList<Integer> docILine = new ArrayList<Integer>();
		ArrayList<String> docITitle= new ArrayList<String>();
		ArrayList<Integer> offsetList = new ArrayList<Integer>();
		
		int totDictOffset = 0;
		int dictEntries = 0;
		int postEntries = 0;
		int docIEntries = 0;
		
		/* reading dictionary.txt */
		try{
			
			Scanner dictScan = new Scanner(dict);
			dictEntries = dictScan.nextInt();
			
			while(dictScan.hasNextLine() == true){
				String[] ob1 = dictScan.nextLine().split(" ");
				dictWords.add(ob1[0]);
				int convToInt1 = Integer.parseInt(ob1[1]);
				dicttf.add(convToInt1);
			}
		
		}
		
		catch (Exception e){
			e.printStackTrace();
		}
		
		try{
			
			Scanner postScan = new Scanner(post);
			postEntries = postScan.nextInt();
			
			while (postScan.hasNextLine() == true){
				String[] ob2 = postScan.nextLine().split(" ");
				postID.add(ob2[0]);
				int convToInt2 = Integer.parseInt(ob2[1]);
				postdf.add(convToInt2);
			}
		}
		catch (Exception e){
			e.printStackTrace();
		}
		
		try{
			
			Scanner docIScan = new Scanner(docI);
			docIEntries = docIScan.nextInt();
			
			while (docIScan.hasNextLine() == true){
				String[] ob3 = docIScan.nextLine().split(" ");
				docIIDs.add(ob3[0]);
				int convToInt3 = Integer.parseInt(ob3[1]);
				docILine.add(convToInt3);
				docITitle.add(ob3[2]);
			}
		}
		catch (Exception e){
			e.printStackTrace();
		}
		
		/* creating the offset list */
		/* adding consecutive word freq gives offset */
		/* subtracting consecutive offset gives document frequency */
		
		for (int i = 0; i < postdf.size(); i++){
			int freq = postdf.get(i);
			totDictOffset += freq;
			offsetList.add(totDictOffset);
		}
		
		
		Scanner sc = new Scanner(System.in);
		String quit = "q";
		ArrayList<Integer> queryTot = new ArrayList();
		ArrayList<Integer> result = new ArrayList();
		Map<String, Integer> myMap = new HashMap<String, Integer>();
		
		while (!sc.nextLine().equals(quit)){
			int total = 0;
			System.out.println("Enter q to quit");
			String query = sc.nextLine();
			
			if (query.equals(quit)){
				System.out.println("Exit");
				break;
			}
			
			/* creating the bag of words */
			int times = 0;
			String[] splitQuery = query.split(" ");
			for (int i = 0; i < splitQuery.length; i++){
				for (int j = i; j < splitQuery.length; j++){
					if (splitQuery[i].equals(splitQuery[j])){
						times++;
					}
				}
				myMap.put(splitQuery[i], times);
				times = 0;
			}
			
			/* calculate query total value for each document*/
			double total = 0;
			int N = dictEntries;
			for (int l = 0; l < (offsetList.length - 1); l++){
				int myDF = offsetList.get(l+1) - offsetList.get(l);
					
				for (int k = 0; k < splitQuery.length; k++){
					int myTF = myMap.get(splitQuery[k]);
					total += myTF * Math.log10(N / myDF);
				}
				result.add(total);
				total = 0;
			}
				
		}
	}
	
}
	
