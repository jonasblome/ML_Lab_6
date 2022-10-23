package de.htw.ml;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.jblas.util.Random;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class Ue06_Blome_Jonas {
	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Car Index";
	public static final String yAxisLabel = "mpg";
	
	public static void main(String[] args) throws IOException {
		/*
		 * Problem definition:
		 * Your task is to port the Python program from exercise 5 to Java, using JBlas.
		 * In JBlas a matrix is represented with the class FloatMatrix.
		 * As in Python, a vector has no own class, they use FloatMatrix as well.
		 * Calculate again the RMSE values per training iteration
		 * and plot the curve into a diagram with the plot() method.
		 *
		 * The project also contains another data set (german_credit.csv).
		 * For more information about the data set, see the UCI repository.
		 * Use the set and train a model which can predict the "Credit Amount".
		 * Plot the RMSE for each training iteration
		 * and output the best RMSE value on the console.
		 * Vary the learning rate and the number of training iterations
		 * to achieve decent RMSE values.
		 * The set is quite complicated
		 * and RMSE values below 2000 are already very good
		 */

		// IM NOT PLOTTING THE CURVES BECAUSE THE JAVA FX PACK WOULDN'T WORK ON MY PC!!!

		System.out.println("Predicting mileage");
		FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");
		approximateMPG(cars);

		System.out.println();

		System.out.println("Predicting credit amount");
		FloatMatrix germanCredit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		approximateCreditAmount(germanCredit);
	}

	// ---------------------------------------------------------------------------------
	// ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
	// ---------------------------------------------------------------------------------
	private static void approximateMPG(FloatMatrix cars) {
		// Amount of data points in the car dataset
		int numDatapoints = cars.rows;
		int numIterations = 1000;
		float[] learningRates = {0.01f, 0.1f, 1.0f, 2.0f};

		// RMSE array for each iteration
		float[][] rmseValues = new float[learningRates.length][numIterations];

		// Store max and min values of cars columns for denormalization
		float[] maxValues = new float[cars.columns];

		for(int col = 0; col < cars.columns; col++) {
			maxValues[col] = cars.getColumn(col).max();
		}

		float[] minValues = new float[cars.columns];

		for(int col = 0; col < minValues.length; col++) {
			minValues[col] = cars.getColumn(col).min();
		}

		// Normalize credit data
		float[][] normalizedData = new float[cars.columns][cars.rows];

		for(int col = 0; col < maxValues.length; col++) {
			normalizedData[col] = cars.getColumn(col).sub(minValues[col]).div(maxValues[col] - minValues[col]).toArray();
		}

		// Turn normalized data arrays into vectors
		FloatMatrix[] normalizedVectors = new FloatMatrix[cars.columns];

		for(int col = 0; col < normalizedVectors.length; col++) {
			normalizedVectors[col] = new FloatMatrix(normalizedData[col]);
		}

		// Put credit data into matrix
		FloatMatrix inputMatrix = new FloatMatrix(normalizedData[0]);

		for(int i = 1; i < normalizedVectors.length - 1; i++) {
			inputMatrix = FloatMatrix.concatHorizontally(inputMatrix, normalizedVectors[i]);
		}

		for(int col = 0; col < cars.columns; col++) {
			maxValues[col] = cars.getColumn(col).max();
		}
		
		// Create four different learning curves
		for(int lc = 0; lc < learningRates.length; lc++) {
			// Set seed to start all learning curves at same value
			Random.seed(5);

			// Set initial theta values
			float[] theta = new float[cars.columns - 1];
			for(int j = 0; j < theta.length; j++) {
				theta[j] = Random.nextFloat();
			}
			FloatMatrix thetaVector = new FloatMatrix(theta);
			float bestRMSE = 100000;
			
			// Try for many iterations to find the best theta values
			for(int it = 0; it < numIterations; it++) {
				// Calculate approximated mpg values with weighted function (hypothesis function)
				FloatMatrix hypothesis = inputMatrix.mmul(thetaVector);
				
				// Calculate disparity
				FloatMatrix disparity = hypothesis.sub(normalizedVectors[cars.columns - 1]);
				
				// Calculate theta-delta values and update them
				FloatMatrix thetaDeltaVector = inputMatrix.transpose().mmul(disparity);
				thetaVector = thetaVector.sub(thetaDeltaVector.mul(learningRates[lc] / numDatapoints));
				
				// Calculate and output RMSE value
				FloatMatrix errors = hypothesis.add(minValues[cars.columns - 1]).mul(maxValues[cars.columns - 1] - minValues[cars.columns - 1]).sub(normalizedVectors[cars.columns - 1].add(minValues[cars.columns - 1]).mul(maxValues[cars.columns - 1] - minValues[cars.columns - 1]));
				FloatMatrix squaredErrors = errors.mul(errors);

				float rmse = (float) Math.sqrt(squaredErrors.sum() / numDatapoints);

				// Store current RMSE for later display
				rmseValues[lc][it] = rmse;
				
				// Set new RMSE and Theta if approximation is better
				if(rmse < bestRMSE) {
					bestRMSE = rmse;
				}
			}

			System.out.println("Final RMSE for learning curve " + lc +": " + bestRMSE);
			// FXApplication.plot(rmseValues[0]);
			// Application.launch(FXApplication.class);

		}
	}
	
	private static void approximateCreditAmount(FloatMatrix germanCredit) {
		int numDatapoints = germanCredit.rows;
		int numIterations = 1000;
		float[] learningRates = {0.001f, 0.01f, 0.1f, 1.0f};

		// RMSE array for each iteration
		float[][] rmseValues = new float[learningRates.length][numIterations];

		// Store max and min values of credit columns for normalization
		float[] maxValues = new float[germanCredit.columns];
		
		for(int col = 0; col < maxValues.length; col++) {
			maxValues[col] = germanCredit.getColumn(col).max();
		}

		float[] minValues = new float[germanCredit.columns];

		for(int col = 0; col < minValues.length; col++) {
			minValues[col] = germanCredit.getColumn(col).min();
		}
		
		// Normalize credit data
		float[][] normalizedData = new float[germanCredit.columns][germanCredit.rows];
		
		for(int col = 0; col < maxValues.length; col++) {
			normalizedData[col] = germanCredit.getColumn(col).sub(minValues[col]).div(maxValues[col] - minValues[col]).toArray();
		}

		// Turn normalized data arrays into vectors
		FloatMatrix[] normalizedVectors = new FloatMatrix[germanCredit.columns];

		for(int col = 0; col < normalizedVectors.length; col++) {
			normalizedVectors[col] = new FloatMatrix(normalizedData[col]);
		}
		
		// Put credit data into matrix
		// I PUT THE CREDIT AMOUNT COLUMN IN MY .CSV FILE TO THE FRONT, SO NOW IT'S THE FIRST COLUMN!!!!
		FloatMatrix inputMatrix = new FloatMatrix(normalizedData[1]);

		for(int i = 2; i < normalizedVectors.length; i++) {
			inputMatrix = FloatMatrix.concatHorizontally(inputMatrix, normalizedVectors[i]);
		}
		
		// Create four different learning curves
		for(int lc = 0; lc < learningRates.length; lc++) {
			Random.seed(9);
			
			// Set initial theta values
			float[] theta = new float[germanCredit.columns - 1];
			for(int j = 0; j < theta.length; j++) {
				theta[j] = Random.nextFloat();
			}
			float bestRMSE = 100000;

			// Turn theta values into vector
			FloatMatrix thetaValuesVector = new FloatMatrix(theta);
			
			// Try multiple iterations to find the best theta values
			for(int it = 0; it < numIterations; it++) {
				// Calculate approximated credit amount values with weighted function (hypothesis function)
				FloatMatrix hypothesis = inputMatrix.mmul(thetaValuesVector);
				
				// Calculate disparity
				FloatMatrix disparity = hypothesis.sub(normalizedVectors[0]);
				
				// Calculate theta-delta values and update them
				FloatMatrix thetaDeltaVector = inputMatrix.transpose().mmul(disparity);
				thetaValuesVector = thetaValuesVector.sub(thetaDeltaVector.mul(learningRates[lc] / numDatapoints));
				
				// Calculate and output RMSE value
				FloatMatrix errors = hypothesis.add(minValues[0]).mul(maxValues[0] - minValues[0]).sub(normalizedVectors[0].add(minValues[0]).mul(maxValues[0] - minValues[0]));
				FloatMatrix squaredErrors = errors.mul(errors);

				float rmse = (float) Math.sqrt(squaredErrors.sum() / numDatapoints);

				// Store current RMSE for later display
				rmseValues[lc][it] = rmse;
				
				// Set new RMSE and Theta if approximation is better
				if(rmse < bestRMSE) {
					bestRMSE = rmse;
				}
			}

			System.out.println("Final RMSE for learning curve " + lc +": " + bestRMSE);
			// FXApplication.plot(rmseValues[0]);
			// Application.launch(FXApplication.class);
		}
	}
	
	/**
	 * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
	 * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
	 *
	 * @author Nico Hezel
	 *
	 */
	public static class FXApplication extends Application {
	
		/**
		 * equivalent to linspace in Octave
		 * 
		 * @param lower
		 * @param upper
		 * @param num
		 * @return
		 */
		private static FloatMatrix linspace(float lower, float upper, int num) {
	        float[] data = new float[num];
	        float step = Math.abs(lower-upper) / (num-1);
	        for (int i = 0; i < num; i++)
	            data[i] = lower + (step * i);
	        data[0] = lower;
	        data[data.length-1] = upper;
	        return new FloatMatrix(data);
	    }
		
		// y-axis values of the plot 
		private static float[] dataY;
		
		/**
		 * Draw the values and start the UI
		 */
		public static void plot(float[] yValues) {
			dataY = yValues;
		}
		
		/**
		 * Draw the UI
		 */
		@SuppressWarnings("unchecked")
		@Override 
		public void start(Stage stage) {
	
			stage.setTitle(title);
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel(xAxisLabel);
	        final NumberAxis yAxis = new NumberAxis();
	        yAxis.setLabel(yAxisLabel);
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
	
			XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
			series1.setName("Data");
			for (int i = 0; i < dataY.length; i++) {
				series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
			}
	
			sc.setAnimated(false);
			sc.setCreateSymbols(true);
	
			sc.getData().addAll(series1);
	
			Scene scene = new Scene(sc, 500, 400);
			stage.setScene(scene);
			stage.show();
	    }
	}
}
