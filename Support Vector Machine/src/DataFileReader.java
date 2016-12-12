import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

public class DataFileReader {
	
	public static final String TRAINING_DATA = "trainingData";
	public static final String TRAINING_DATA_LABELS = "trainingDataLabels";
	public static final String TESTING_DATA = "testingData";
	public static final String TESTING_DATA_LABELS = "testingDataLabels";

	/**
	 * @param folderName
	 * @return gray scale image arrays given a folder
	 */
	public static List<List<Double>> getGrayScaleImageArrays(File folderName) {
		
		List<File> directoryListing = getDirectoryListing(folderName);
		List<List<Double>> grayScaleImageArrays = new ArrayList<List<Double>>(directoryListing.size());
		
		for (File imageFile : directoryListing) {
			grayScaleImageArrays.add(getGrayScaleImageArray(getImageFileContents(imageFile)));
		}

		return grayScaleImageArrays;
		
	}
	
	/**
	 * @param numberOfLabels
	 * @param isHumanPresent
	 * @return list of labels 
	 */
	public static List<BinaryDataLabel> getLabelsList(int numberOfLabels, boolean isHumanPresent) {
		
		List<BinaryDataLabel> labelsList = new ArrayList<BinaryDataLabel>(numberOfLabels);
		
		for (int labelCounter = 0; labelCounter < numberOfLabels; ++labelCounter) {
			labelsList.add(isHumanPresent ? BinaryDataLabel.POSITIVE_LABEL : BinaryDataLabel.NEGATIVE_LABEL);
		}
		
		return labelsList;
		
	}
	
	/**
	 * @param data
	 * @param labels
	 * @param trainingDataFraction
	 * @return partitioned data and labels for training and testing
	 */
	public static Map<String, Object> partitionDataAndLabels(List<List<Double>> data, List<BinaryDataLabel> labels, double trainingDataFraction) {
		
		//Training data fraction should be less than 1 and there should be the same number of data and labels
		assert trainingDataFraction < 1.0 && data.size() == labels.size();
		
		//Compute the number of records in each split
		int numberOfTrainingRecords = (int) trainingDataFraction * data.size(), numberOfTestingRecords = data.size() - numberOfTrainingRecords, recordCounter = 0;
		
		Map<String, Object> partitionedDataAndLabels = new HashMap<String, Object>(4);
		List<List<Double>> trainingData = new ArrayList<List<Double>>(numberOfTrainingRecords);
		List<BinaryDataLabel> trainingDataLabels = new ArrayList<BinaryDataLabel>(numberOfTrainingRecords);
		List<List<Double>> testingData = new ArrayList<List<Double>>(numberOfTestingRecords);
		List<BinaryDataLabel> testingDataLabels = new ArrayList<BinaryDataLabel>(numberOfTestingRecords);
		
		//Loop through the data and partition it
		for (List<Double> record : data) {
		
			if (recordCounter < numberOfTrainingRecords) {
				trainingData.add(record);
				trainingDataLabels.add(labels.get(recordCounter));
			} else {
				testingData.add(record);
				testingDataLabels.add(labels.get(recordCounter));
			}
		
			++recordCounter;
		}
		
		//Add partitioned data and labels to the map
		partitionedDataAndLabels.put(TRAINING_DATA, trainingData);
		partitionedDataAndLabels.put(TRAINING_DATA_LABELS, trainingDataLabels);
		partitionedDataAndLabels.put(TESTING_DATA, testingData);
		partitionedDataAndLabels.put(TESTING_DATA_LABELS, testingDataLabels);
		
		return partitionedDataAndLabels;
		
	}
	
	/**
	 * @param imageFileContents
	 * @return gray scale image contents given an image file
	 */
	private static List<Double> getGrayScaleImageArray(BufferedImage imageFileContents) {
		
		int imageHeight = imageFileContents.getHeight(), imageWidth = imageFileContents.getWidth(), redGreenBlue = 0, red =0, green = 0, blue = 0, gray = 0;
		List<Double> grayScaleImageArray = new ArrayList<Double>(imageHeight * imageWidth);
		
		//Get gray scale value at each pixel location
		for (int widthCounter = 0; widthCounter < imageWidth; ++widthCounter) {
			for (int heightCounter = 0; heightCounter < imageHeight; ++heightCounter) {
			
				redGreenBlue = imageFileContents.getRGB(widthCounter, heightCounter);
				red = (redGreenBlue >> 16) & 0xFF;
				green = (redGreenBlue >> 8) & 0xFF;
				blue = (redGreenBlue & 0xFF);
				gray = (red + green + blue) / 3;
			
				grayScaleImageArray.add(Double.valueOf(gray));
			}
		}
		
		return grayScaleImageArray;
	}
	
	/**
	 * @param folderName
	 * @return list of files in directory
	 */
	private static List<File> getDirectoryListing(File folderName) {
		
		List<File> directoryListing = new ArrayList<File>();
		
		if (folderName.isDirectory()) {
			File[] listing = folderName.listFiles();
			for (File fileEntry : listing) {
				if (fileEntry.isFile()) {
					directoryListing.add(fileEntry);
				}
			}
		}
		
		return directoryListing;
		
	}
	
	/**
	 * @param fileName
	 * @return contents of the image in the form of a buffered image
	 */
	private static BufferedImage getImageFileContents(File fileName) {
		
		//Make sure a file is passed in
		if (!fileName.isFile()) {
			return null;
		}
		
		BufferedImage imageFileContents = null;
		try {
			imageFileContents =  ImageIO.read(fileName);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return imageFileContents;
		
	}
	
}
