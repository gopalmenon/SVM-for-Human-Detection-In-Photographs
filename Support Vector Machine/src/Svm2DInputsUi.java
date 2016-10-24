import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

import javax.swing.*;

public class Svm2DInputsUi {

	private static final int ADD_POSITIVE_POINTS_MODE = 1;
	private static final int ADD_NEGATIVE_POINTS_MODE = 2;
	private static final int ADD_TEST_POINTS_MODE = 3;
	private static final int POINTS_DIMENSION = 10;
	private static final int PANEL_WIDTH = 640;//1280
	private static final int PANEL_HEIGHT = 480;//780
	
	private static int selectedMode = 0;

	/**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event-dispatching thread.
     */
    private static void createAndShowGUI() {
        //Create and set up the window.
        JFrame frame = new JFrame("Support Vector Machine 2D Inputs");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);

        //Create the menu bar.  Make it have a green background.
        JMenuBar menuBar = new JMenuBar();
        menuBar.setOpaque(true);        
        menuBar.setPreferredSize(new Dimension(PANEL_WIDTH, 20));
        
        //Create user menus
        JMenu fileMenu, trainingMenu, testingMenu, visualizationMenu;
        JMenuItem exitMenuItem, addPositivePointMenuItem, addNegativePointMenuItem, findBestSeparatorMenuItem, addTestPointMenuItem;
        JCheckBoxMenuItem showMarginsMenuItem, showSupportVectorsMenuItem;
        
        //File menu
        fileMenu = new JMenu("File");
        exitMenuItem = new JMenuItem("Exit");
        exitMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {System.exit(0);} });
        fileMenu.add(exitMenuItem);
        menuBar.add(fileMenu);
        
        //Create the UI panel
        final UiPanel uiPanel = new UiPanel();
        uiPanel.setPreferredSize(new Dimension(PANEL_WIDTH, PANEL_HEIGHT));
        
        //Training menu
        trainingMenu = new JMenu("Training");
        addPositivePointMenuItem = new JMenuItem("Add Positive Points");
        addPositivePointMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {selectedMode = ADD_POSITIVE_POINTS_MODE;} });
        addNegativePointMenuItem = new JMenuItem("Add Negative Points");
        addNegativePointMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {selectedMode = ADD_NEGATIVE_POINTS_MODE;} });
        findBestSeparatorMenuItem = new JMenuItem("Find Best Separator");
        findBestSeparatorMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {uiPanel.fitTrainingData();} });
        trainingMenu.add(addPositivePointMenuItem);
        trainingMenu.add(addNegativePointMenuItem);
        trainingMenu.add(findBestSeparatorMenuItem);
        menuBar.add(trainingMenu);

        //Make a list of menu items to be disabled after training
        List<JMenuItem> menuItemList = new ArrayList<JMenuItem>();
        menuItemList.add(addPositivePointMenuItem);
        menuItemList.add(addNegativePointMenuItem);
        menuItemList.add(findBestSeparatorMenuItem);
        findBestSeparatorMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {disableTrainingInputsAfterTraining(menuItemList);} });
        
        //Testing menu
        testingMenu = new JMenu("Testing");
        addTestPointMenuItem = new JMenuItem("Add Test Point");
        addTestPointMenuItem.addActionListener(new ActionListener() {public void actionPerformed(ActionEvent e) {selectedMode = ADD_TEST_POINTS_MODE;} });
        testingMenu.add(addTestPointMenuItem);
        menuBar.add(testingMenu);
        
        //Visualization menu
        visualizationMenu = new JMenu("Visualization");
        showMarginsMenuItem = new JCheckBoxMenuItem("Show Margins", true);
        //showVoronoiCellsMenuItem.addItemListener(new ItemListener() {public void itemStateChanged(ItemEvent e) {UavTerrainUi.this.uavTerrainPanel.setShowVoronoiCells(e.getStateChange() == ItemEvent.SELECTED ? true : false);}});
        visualizationMenu.add(showMarginsMenuItem);
        showSupportVectorsMenuItem = new JCheckBoxMenuItem("Show Support Vectors", true);
        visualizationMenu.add(showSupportVectorsMenuItem);
        menuBar.add(visualizationMenu);
        
        //Set the menu bar and add the label to the content pane.
        frame.setJMenuBar(menuBar);
        frame.getContentPane().add(uiPanel, BorderLayout.CENTER);
        uiPanel.addMouseListener(uiPanel);
        
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
    
    /**
     * Disable menu options used for training
     * @param menuItemList
     */
    private static void disableTrainingInputsAfterTraining(List<JMenuItem> menuItemList) {
    	
    	for (JMenuItem menuItem : menuItemList) {
    		menuItem.setEnabled(false);
    	}
    	
    }
    
    //Define the terrain that will contain the obstacles, start an d end points
    @SuppressWarnings("serial")
	static class UiPanel extends JPanel implements MouseListener {
    	
    	private List<InputPoint> positiveInputs = new ArrayList<InputPoint>();
    	private List<InputPoint> negativeInputs = new ArrayList<InputPoint>();
    	private List<InputPoint> positivePredictions = new ArrayList<InputPoint>();
    	private List<InputPoint> negativePredictions = new ArrayList<InputPoint>();
    	private List<InputPoint> supportVectors = new ArrayList<InputPoint>();
    	private SupportVectorMachine supportVectorMachine = null;
    	
    	private List<Double> weightVector = null;
    	private boolean trainingAttempted = false;
    	private  boolean bestSeparatorFound = false;
    	
    	private int numberOfSupportVectors = 0;
    	
    	public void paintComponent(Graphics g) {
        	
            super.paintComponent(g);

            //Draw positive points
            g.setColor(Color.GREEN);
            for (InputPoint point : positiveInputs) {
            	g.fillOval(point.getxCoordinate() - POINTS_DIMENSION / 2, point.getyCoordinate() - POINTS_DIMENSION / 2, POINTS_DIMENSION, POINTS_DIMENSION);
            }

            //Draw positive predictions
            g.setColor(Color.GREEN);
            for (InputPoint point : positivePredictions) {
            	g.drawOval(point.getxCoordinate() - POINTS_DIMENSION / 2, point.getyCoordinate() - POINTS_DIMENSION / 2, POINTS_DIMENSION, POINTS_DIMENSION);
            }
            
            //Draw negative points
            g.setColor(Color.RED);
            for (InputPoint point : negativeInputs) {
            	g.fillOval(point.getxCoordinate() - POINTS_DIMENSION / 2, point.getyCoordinate() - POINTS_DIMENSION / 2, POINTS_DIMENSION, POINTS_DIMENSION);
            }

            //Draw negative predictions
            g.setColor(Color.RED);
            for (InputPoint point : negativePredictions) {
            	g.drawOval(point.getxCoordinate() - POINTS_DIMENSION / 2, point.getyCoordinate() - POINTS_DIMENSION / 2, POINTS_DIMENSION, POINTS_DIMENSION);
            }
             
            //Draw separator
            
            //Draw margins
            
            
            //Draw support vectors

    	}
    	
    	/**
    	 * Fit the training data
    	 */
    	public void fitTrainingData() {
    		
    		findBestSeparator();
    		markSupportVectors();
    	}

		/**
		 * Find the separator with the widest margin
		 */
		public void findBestSeparator() {
			
			List<List<Double>> featureVectors = getTrainingInputs(this.positiveInputs);
			featureVectors.addAll(getTrainingInputs(this.negativeInputs));
			
			List<BinaryDataLabel> trainingDataLabels = getTrainingLabels(this.positiveInputs.size(), true);
			trainingDataLabels.addAll(getTrainingLabels(this.negativeInputs.size(), false));
			
			this.supportVectorMachine = new SupportVectorMachine();
			supportVectorMachine.fit(featureVectors, trainingDataLabels);
			
			this.trainingAttempted = true;
			this.bestSeparatorFound = true;
			
			this.weightVector = supportVectorMachine.getWeightVector();

		}
		
		public void markSupportVectors() {
			
			for (InputPoint inputPoint : this.positiveInputs) {
				
				int weightIndex = 0;
				double dotProduct = 0.0;
				for (Double weight : this.weightVector) {
					if (weightIndex == 0) {
						dotProduct += weight;
					} else {
						dotProduct += (weight * (weightIndex == 1 ? inputPoint.getxCoordinate() : inputPoint.getyCoordinate()));
					}
					++weightIndex;
				}
				
			}

			
			for (InputPoint inputPoint : this.negativeInputs) {
				
				int weightIndex = 0;
				double dotProduct = 0.0;
				for (Double weight : this.weightVector) {
					if (weightIndex == 0) {
						dotProduct += weight;
					} else {
						dotProduct += (weight * (weightIndex == 1 ? inputPoint.getxCoordinate() : inputPoint.getyCoordinate()));
					}
					++weightIndex;
				}
				
			}
		}
		
		/**
		 * @param inputPoints
		 * @return list of x and y coordinate pairs
		 */
		public List<List<Double>> getTrainingInputs(List<InputPoint> inputPoints) {
			
			List<List<Double>> trainingInputs = new ArrayList<List<Double>>();
			
			ArrayList<Double> trainingInput = null;
			for (InputPoint inputPoint : inputPoints) {
				trainingInput = new ArrayList<Double>(2);
				trainingInput.add(Double.valueOf((double)inputPoint.getxCoordinate()));
				trainingInput.add(Double.valueOf((double)inputPoint.getyCoordinate()));
				trainingInputs.add(trainingInput);
			}
			
			return trainingInputs;
			
		}
		
		/**
		 * @param numberOfInputPoints
		 * @param positiveLabel
		 * @return list of training data labels
		 */
		public List<BinaryDataLabel> getTrainingLabels(int numberOfInputPoints, boolean positiveLabel) {
			
			List<BinaryDataLabel> trainingLabels = new ArrayList<>();
			for (int labelCounter = 0; labelCounter < numberOfInputPoints; ++labelCounter) {
				if (positiveLabel) {
					trainingLabels.add(BinaryDataLabel.POSITIVE_LABEL);
				} else {
					trainingLabels.add(BinaryDataLabel.NEGATIVE_LABEL);
				}
			
			}
			
			return trainingLabels;
			
		}

		@Override
		public void mouseClicked(MouseEvent e) {
			
			switch (selectedMode) {
			
			case ADD_POSITIVE_POINTS_MODE:
				if (!trainingAttempted) {
					positiveInputs.add(new InputPoint(e.getX(), e.getY()));
				}
				break;
				
			case ADD_NEGATIVE_POINTS_MODE:
				if (!trainingAttempted) {
					negativeInputs.add(new InputPoint(e.getX(), e.getY()));
				}
				break;
				
			case ADD_TEST_POINTS_MODE:
				if (trainingAttempted && bestSeparatorFound) {
					List<Double> testVector = new ArrayList<Double>();
					testVector.add(Double.valueOf((double)e.getX()));
					testVector.add(Double.valueOf((double)e.getY()));
					if (this.supportVectorMachine.getPrediction(testVector) == BinaryDataLabel.POSITIVE_LABEL) {
						positivePredictions.add(new InputPoint(e.getX(), e.getY()));
						System.out.println("Positive point " + testVector + ", weight vector " + this.weightVector);
					} else {
						negativePredictions.add(new InputPoint(e.getX(), e.getY()));
						System.out.println("Negative point " + testVector + ", weight vector " + this.weightVector);
					}
				}
				break;
				
			}
			
			repaint();
		}

		@Override
		public void mousePressed(MouseEvent e) {
		}

		@Override
		public void mouseReleased(MouseEvent e) {
		}

		@Override
		public void mouseEntered(MouseEvent e) {
		}

		@Override
		public void mouseExited(MouseEvent e) {
		}    	
    }
    
    /**
     * Class to store x and y coordinates of points
     *
     */
    static class InputPoint {

		private int xCoordinate;
    	private int yCoordinate;
    	
    	public InputPoint(int xCoordinate, int yCoordinate) {
    		this.xCoordinate = xCoordinate;
    		this.yCoordinate = yCoordinate;
    	}
    	
    	public int getxCoordinate() {
			return xCoordinate;
		}

		public int getyCoordinate() {
			return yCoordinate;
		}
    	
    }
    
    public static void main(String[] args) {
        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}