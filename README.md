# MSAAI_530_Group_2 Final Project 

## April Bradick and Parker Christenson 
## Masters of Applied Artificial Intelligence, University of San Diego 
## AAI 530-01: Data Analytics and Internet of Things 
## Prof. A. Marburt
### 2024-02-26


![image](https://github.com/ParChristUSD/MSAAI_530_Group/assets/146489811/3ec1be9b-3bc2-498a-8e7a-077e2919bcd4)


### [Link to Data set](https://www.kaggle.com/datasets/kukuroo3/room-occupancy-detection-data-iot-sensor/data)


## `EDA Group 2 JuPyter Notebook`

- This notebook is an analysis and visualization of the occupancy dataset. It begins by loading the data from a CSV file, preprocessing it, including separating datetime information into distinct columns. The analysis includes identifying null values, summarizing the date range, and calculating data distribution across different dates.

- The script then visualizes data distribution and explores relationships between variables through a correlation matrix. It utilizes a UMAP for dimensionality reduction to visualize clustering patterns and extracts features such as the hour of the day to analyze occupancy trends over time. This comprehensive analysis aims to uncover insights into occupancy patterns and data collection consistency.

## `Time Series JuPyter Notebook`

- This notebook focuses on predicting occupancy using time series data. It imports essential libraries for data manipulation, statistical modeling, and visualization. The dataset is loaded into a DataFrame to check for missing values and understand its structure. Occupancy, a binary variable, serves as the dependent variable, while continuous variables like Temperature, Humidity, Light, CO2, and HumidityRatio act as independent variables.

- The core of the analysis involves defining a function `timeseriesreg` to predict occupancy over time using weighted linear regression, where weights decrease for older observations. This method accounts for the time-dependent nature of the data. The script tests various parameters, including prediction horizon (`ph`), weighting factor (`mu`), and sample size (`n_s`), to optimize the model's predictive accuracy, measured by the mean squared error (MSE).

Results from multiple parameter configurations are aggregated and sorted by MSE to identify the most accurate model. This approach allows for a systematic evaluation of how different factors affect the model's performance in predicting occupancy, aiming to provide insights for optimizing building management and energy efficiency.

## `Fully Connected Dense Neural Network JuPyter Notebook` 

- This noteboook explores the relationship between various predictors and the presence of light in an occupancy dataset using a fully connected deep neural network (FCDNN). The dataset is preprocessed to extract features like hour of the day, day of the week, and categorical variables indicating weekdays, working hours, and light status. 

- The analysis involves three separate neural network models, each predicting the binary status of light (on/off) based on different predictors: Working Hours, Occupancy, and Temperature. Each model is trained and tested on data split sequentially to respect the time series nature of the dataset, ensuring that training data precede testing data in time.

- The neural network models are constructed with TensorFlow and Keras, featuring layers with ReLU activation for hidden layers and sigmoid activation for the output layer to accommodate binary classification. The models are compiled with the Adam optimizer and binary crossentropy loss function, and trained with metrics including accuracy, precision, recall, and AUC to evaluate performance.

- Post-training, each model's performance is evaluated on the test set, and the results are visualized to compare actual versus predicted light status over time. Predictions and actual values are also saved to CSV files for further analysis. The script concludes by aggregating and displaying performance metrics (test loss, accuracy, precision, recall, and AUC) across all predictors to facilitate comparison and identify the most effective predictor for light status.

## `Logistic Regressions Final JuPyter Notebook`

### Data Preprocessing
The dataset is preprocessed to create new binary variables and to encode categorical and datetime information, facilitating a richer analysis. Notably, the Light variable is converted to a binary Light On/Off status, hours are extracted to identify Working Hours, and days are encoded to distinguish Weekdays.

### Logistic Regression Analysis
The script performs logistic regression in two distinct ways for each outcome: using a random split and a chronological split for train/test datasets. This approach allows for evaluating model performance under different data segmentation strategies, acknowledging the sequential nature of time-series data.

### Individual and Grouped Predictors
For each binary outcome, the script first evaluates a logistic regression model with a group of predictors, followed by individual regressions for each predictor. This dual approach provides insights into both the collective and isolated impacts of predictors on the binary outcomes.

### Results Compilation
The accuracy scores from these analyses are compiled into tables for both random and chronological splits, offering a clear comparison of model performance across different predictors and outcomes. These tables serve as a concise summary of the predictive capabilities of various environmental and time-related factors in determining building occupancy dynamics.

This analysis not only highlights the significance of individual and grouped predictors in occupancy-related predictions but also showcases the importance of considering the sequential nature of data in predictive modeling.

## `LSTM NN JuPyter Notebook`

### Data Preprocessing and Feature Extraction
- A new feature, `Temp_Hum_Ratio`, is created to capture the relationship between temperature and humidity.
- The day of the week (`DOTW`) is extracted from the date column and encoded numerically to facilitate model training.
- The dataset is then split into training and testing sets to evaluate model performance.

### Model Development
Three different neural network architectures are explored:
- **Model 1**: A simple architecture with one hidden layer containing 50 neurons.
- **Model 2**: An extended architecture with two hidden layers, containing 50 and 100 neurons respectively.
- **Model 3**: A similar architecture to Model 2 but with an additional hidden layer containing 100 neurons.

Each model is compiled and trained with the Adam optimizer, and their performance is evaluated based on training and validation accuracy.

### Model Evaluation and Comparison
- The training process for each model is timed, and the models' accuracies are plotted to compare their performance over epochs.
- Model 3 demonstrates superior performance, achieving the highest training and validation accuracy among the three models.

### Optimization with SGD Optimizer
- Model 3 is further optimized by training it with the SGD optimizer instead of Adam, to explore potential improvements in accuracy.
- A comparison between the performance of Model 3 with Adam and SGD optimizers is visualized, showing the impact of optimizer choice on model accuracy.

### Application and Prediction
- Model 3, trained with the Adam optimizer due to its superior performance, is used to make occupancy predictions on the entire dataset.
- Predictions are rounded to the nearest whole number (0 or 1) to match the binary nature of the target variable, and the distribution of predicted values is compared to the actual occupancy status.

This analysis demonstrates the effectiveness of neural networks in predicting occupancy status from environmental and temporal data. The exploration of different model architectures and optimizers highlights the importance of model tuning in achieving high accuracy.


## `IoT Device PNG`
- This is the group drawing for our IoT device all of the components are clearly labled with a number, and their functions are listed within the Final Paper.
  
- The file, also contains a outline of the device pipeline, which is also clearly labled, and explained in greater detail within the Final Paper. 
