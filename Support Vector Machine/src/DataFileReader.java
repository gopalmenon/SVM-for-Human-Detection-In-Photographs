import java.awt.Graphics2D;
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
	public static final String EXPECTED_IMAGE_FILE_EXTENSION = "bmp";
	
	public static final double SCALED_WIDTH_FRACTION = 0.1;
	public static final double SCALED_HEIGHT_FRACTION = 0.1;

	/**
	 * @param folderName
	 * @param resizedFolderName
	 * @return gray scale image arrays given a folder
	 */
	public static List<List<Double>> getGrayScaleImageArrays(File folderName, File resizedFolderName) {
		
		resizeImages(folderName, resizedFolderName);
		
		List<File> directoryListing = getDirectoryListing(resizedFolderName);
		List<List<Double>> grayScaleImageArrays = new ArrayList<List<Double>>(directoryListing.size());
		
		for (File imageFile : directoryListing) {
			if (imageFile.getName().endsWith(EXPECTED_IMAGE_FILE_EXTENSION)) {
				grayScaleImageArrays.add(getGrayScaleImageArray(getImageFileContents(imageFile)));
			}
		}

		return grayScaleImageArrays;
		
	}
	
	/**
	 * Resize the images for faster processing
	 * @param folderName
	 * @param resizedFolderName
	 */
	private static void resizeImages(File folderName, File resizedFolderName) {
		
		List<File> directoryListing = getDirectoryListing(folderName);
		
		BufferedImage originalImage = null, scaledImage = null;
		int scaledWidth = 0, scaledHeight = 0;
		String scaledFileName = null;
		for (File imageFile : directoryListing) {
			
			if (imageFile.getName().endsWith(EXPECTED_IMAGE_FILE_EXTENSION)) {
				
				try {
					
					//Get the original image
					originalImage = ImageIO.read(imageFile);

					//Scale the image
					scaledWidth = (int) (originalImage.getWidth() * SCALED_WIDTH_FRACTION);
					scaledHeight = (int) (originalImage.getHeight() * SCALED_HEIGHT_FRACTION);
					scaledImage = new BufferedImage(scaledWidth, scaledHeight, originalImage.getType());
					Graphics2D g2d = scaledImage.createGraphics();
			        g2d.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null);
			        g2d.dispose();
			        
			        //Save scaled image
			        scaledFileName = resizedFolderName.getPath() + "/" + imageFile.getName();
			        ImageIO.write(scaledImage, EXPECTED_IMAGE_FILE_EXTENSION, new File(scaledFileName));
			        
				} catch (IOException e) {
					e.printStackTrace();
					System.exit(0);
				}
				
			}
		}
		
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
