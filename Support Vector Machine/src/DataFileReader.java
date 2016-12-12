import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class DataFileReader {
	
	public List<List<Integer>> getGrayScaleImageArrays(File folderName) {
		
		List<File> directoryListing = getDirectoryListing(folderName);
		List<List<Integer>> grayScaleImageArrays = new ArrayList<List<Integer>>(directoryListing.size());
		
		for (File imageFile : directoryListing) {
			grayScaleImageArrays.add(getGrayScaleImageArray(getImageFileContents(imageFile)));
		}

		return grayScaleImageArrays;
		
	}
	
	private static List<Integer> getGrayScaleImageArray(BufferedImage imageFileContents) {
		
		int imageHeight = imageFileContents.getHeight(), imageWidth = imageFileContents.getWidth(), redGreenBlue = 0, red =0, green = 0, blue = 0, gray = 0;
		List<Integer> grayScaleImageArray = new ArrayList<Integer>(imageHeight * imageWidth);
		
		//Get gray scale value at each pixel location
		for (int widthCounter = 0; widthCounter < imageWidth; ++widthCounter) {
			for (int heightCounter = 0; heightCounter < imageHeight; ++heightCounter) {
			
				redGreenBlue = imageFileContents.getRGB(widthCounter, heightCounter);
				red = (redGreenBlue >> 16) & 0xFF;
				green = (redGreenBlue >> 8) & 0xFF;
				blue = (redGreenBlue & 0xFF);	
				gray = (red + green + blue) / 3;
			
				grayScaleImageArray.add(Integer.valueOf(gray));
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
