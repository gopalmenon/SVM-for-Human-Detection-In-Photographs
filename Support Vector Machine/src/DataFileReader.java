import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataFileReader {
	
	public static String WHITESPACE_REGEX = "\\s";


	/**
	 * @param filePath
	 * @return file contents as a list of strings
	 * @throws IOException
	 */
	private static List<String> getDataFileContents(String filePath) throws IOException {
		
		List<String> fileContents = new ArrayList<String>();
		BufferedReader bufferedReader = null;
		
		try {
			bufferedReader = new BufferedReader(new FileReader(filePath));
			String fileLine = bufferedReader.readLine();
			
			while (fileLine != null) {
				
				if (fileLine.trim().length() > 0) {
					fileContents.add(fileLine.trim());
				}

				fileLine = bufferedReader.readLine();
			}
			
			bufferedReader.close();
			
		} catch (IOException e) {
			throw e;
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return fileContents;

	}

	/**
	 * @param filePath
	 * @return data as a list of list of doubles
	 */
	public static List<List<Double>> getData(String filePath) {
		
		List<List<Double>> fileData = new ArrayList<List<Double>>();
				
		try {

			//Get raw data as a list of strings
			List<String> rawData = getDataFileContents(filePath);
			
			List<Double> dataVector = null;
			//Loop through each string in the list
			for (String fileLine : rawData) {
		
				//Split each string into tokens
				String[] dataTokens = fileLine.split(WHITESPACE_REGEX);
				if (dataTokens.length > 0) {
					
					//Parse as doubles and add to data vector
					dataVector = new ArrayList<Double>(dataTokens.length);
					for (String token : dataTokens) {
						dataVector.add(Double.valueOf(Double.parseDouble(token)));
					}
				}
				
				fileData.add(dataVector);
				
			}
		
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(0);
		}

		return fileData;
		
	}
	
	/**
	 * @param filePath
	 * @return list of data labels
	 */
	public static List<BinaryDataLabel> getLabels(String filePath) {
		
		List<BinaryDataLabel> dataLabels = new ArrayList<BinaryDataLabel>();
		
		try {

			//Get raw data as a list of strings
			List<String> labelsList = getDataFileContents(filePath);
			
			//Loop through each string in the list
			for (String labelString : labelsList) {
		
				//Parse the label and add it to the list
				if (Integer.parseInt(labelString.trim()) == 1) {
					dataLabels.add(BinaryDataLabel.POSITIVE_LABEL);
				} else {
					dataLabels.add(BinaryDataLabel.NEGATIVE_LABEL);
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(0);
		}		
		
		return dataLabels;
		
	}
	
}
