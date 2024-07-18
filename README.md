# Device Price Classification System

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Python Project](#python-project)
   - [Setup and Installation](#python-setup-and-installation)
   - [Project Structure](#python-project-structure)
   - [Data Preparation](#data-preparation)
   - [Model Training](#model-training)
   - [API Endpoints](#python-api-endpoints)
   - [Running the Python Server](#running-the-python-server)
4. [Java Spring Boot Project](#java-spring-boot-project)
   - [Setup and Installation](#java-setup-and-installation)
   - [Project Structure](#java-project-structure)
   - [API Endpoints](#java-api-endpoints)
   - [Running the Spring Boot Application](#running-the-spring-boot-application)
5. [Test API](#testing)
6. [Dataset Description](#dataste-description)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

The Device Price Classification System is an AI-powered application that predicts the price range of mobile devices based on their specifications. This project consists of two main components:

1. A Python-based machine learning model for price prediction
2. A Java Spring Boot application for managing device data and interfacing with the prediction model

## System Architecture

- Python Service: Handles data preprocessing, model training, and prediction.
- Java Spring Boot Service: Manages the REST API for CRUD operations on device data and interfaces with the Python service for predictions.
- Database: Stores device information.

## Python Project

### Python Setup and Installation

1. Ensure you have Python 3.8+ installed.
2. Clone the repository:

git clone https://github.com/eslammofreh/DevicePriceClassification.git
cd device-price-classification/python

3. Create a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate

4. Install required packages:

pip install -r requirements.txt


### Project Structure


python/
├── data/
│ ├── train.csv
│ ├── test.csv
├── models/
│ ├── init.py
│ ├── base_model.py
│ ├── model_trainer.py
│ ├── random_forest_model.py
│ ├── stacking_model.py
│ ├── model_evaluation.py
│ └── model_trainer.py
├── utils/
│ ├── init.py
│ ├── data_loader.py
│ ├── data_preprocessor.py
│ └── model_utils.py
│ └── visualization.py
├── api/
│ ├── init.py
│ └── api.py
├── main.py
├── config.py
└── requirements.txt


### Data Preparation

The dataset includes various device specifications. Data preparation involves:

1. Loading data from CSV files
2. Handling missing values
3. Feature engineering
4. Splitting data into training and testing sets

### Model Training

We use a stacking ensemble method using RandomForest and GradientBoosting as base models, and LogisticRegression as the final estimator for price range prediction.

### Python API Endpoints

- `/predict` (POST): Accepts device specifications and returns a predicted price range.
- `/train` (GET): Triggers model training.

### Running the Python Server

There are three actions can be taken (train, predict, run_server).

To start the Python Flask server:


python main.py run_server


The server will start on `http://127.0.0.1:5000`.

## Java Spring Boot Project

### Java Setup and Installation

1. Ensure you have Java 11+ and Maven installed.
2. Navigate to the Java project directory:

cd ../java/device-price-classification

3. Build the project:

mvn clean install


### Project Structure


java/DevicePriceClassification/
├── src/
│ ├── main/
│ │ ├── java/com/example/devicepriceclassification/
│ │ │ ├── model/
│ │ │ │ ├────── Device
│ │ │ ├── repository/
│ │ │ │ ├────── DeviceRepository
│ │ │ ├── service/
│ │ │ │ ├────── DeviceService
│ │ │ ├── controller/
│ │ │ │ ├────── DeviceController
│ │ │ ├── config/
│ │ │ │ ├────── AppConfig
│ │ │ └── DevicePriceClassificationApplication.java
│ │ └── resources/
│ │ └── application.properties
│ └── test/
└── pom.xml


### Java API Endpoints

- `GET /api/devices`: Retrieve all devices
- `GET /api/devices/{id}`: Retrieve a specific device by ID
- `POST /api/devices`: Add a new device
- `POST /api/devices/predict/{deviceId}`: Predict price for a specific device

### Running the Spring Boot Application

To start the Spring Boot application:


mvn spring-boot:run


The application will start on `http://localhost:8080`.


### Debugging Steps:

1. Check application logs for both Python and Java components.
2. Use debugging tools provided by your IDE.
3. Implement additional logging at key points in the application to trace data flow.

### Test API:

POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "id": 64,
  "battery_power": 1634,
  "blue": 1,
  "clock_speed": 2.3,
  "dual_sim": 1,
  "fc": 2,
  "four_g": 1,
  "int_memory": 39,
  "m_dep": 0.4,
  "mobile_wt": 164,
  "n_cores": 1,
  "pc": 7,
  "px_height": 386,
  "px_width": 636,
  "ram": 2167,
  "sc_h": 12,
  "sc_w": 0,
  "talk_time": 20,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 1
}


### Get all devices
GET http://localhost:8080/api/devices



### Get device by ID
GET http://localhost:8080/api/devices/1


### Add new device
POST http://localhost:8080/api/devices
Content-Type: application/json

{
  "id": 66,
  "battery_power": 1034,
  "blue": 1,
  "clock_speed": 2.3,
  "dual_sim": 1,
  "fc": 2,
  "four_g": 1,
  "int_memory": 39,
  "m_dep": 0.4,
  "mobile_wt": 164,
  "n_cores": 1,
  "pc": 7,
  "px_height": 386,
  "px_width": 636,
  "ram": 2167,
  "sc_h": 12,
  "sc_w": 0,
  "talk_time": 20,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 1
}


### Predict price for device
POST http://localhost:8080/api/devices/predict/1


## Dataset Description

This dataset contains information about mobile devices and their corresponding price ranges. It is designed for a machine learning task to predict the price range of a mobile device based on its specifications.

### Features:

1. battery_power: Total energy a battery can store in one time (measured in mAh)
2. blue: Boolean indicating Bluetooth availability
3. clock_speed: Speed at which microprocessor executes instructions
4. dual_sim: Boolean indicating dual SIM support
5. fc: Front Camera megapixels
6. four_g: Boolean indicating 4G support
7. int_memory: Internal Memory in Gigabytes
8. m_dep: Mobile Depth in cm
9. mobile_wt: Weight of mobile phone
10. n_cores: Number of cores of processor
11. pc: Primary Camera megapixels
12. px_height: Pixel Resolution Height
13. px_width: Pixel Resolution Width
14. ram: Random Access Memory in Megabytes
15. sc_h: Screen Height of mobile in cm
16. sc_w: Screen Width of mobile in cm
17. talk_time: Longest time that a single battery charge will last
18. three_g: Boolean indicating 3G support
19. touch_screen: Boolean indicating touchscreen support
20. wifi: Boolean indicating Wi-Fi support
21. price_range: Target variable (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)


## Exploratory Data Analysis (EDA) Summary

### Dataset Overview
- The dataset contains 21 columns: 20 features and 1 target variable (price_range).
- Features include both numerical (e.g., battery_power, clock_speed) and categorical (e.g., blue, dual_sim) variables.

### Target Variable Analysis
- price_range is a categorical variable with 4 classes (0, 1, 2, 3) representing different price ranges from low to very high cost.

### Numerical Features Analysis
- battery_power: Ranges from 503 to 1954 mAh
- clock_speed: Varies from 0.5 to 3.0 GHz
- int_memory: Ranges from 5 to 64 GB
- m_dep: Mobile depth varies from 0.1 to 1.0 cm
- mobile_wt: Ranges from 80 to 200 grams
- n_cores: Number of cores varies from 1 to 8
- pc: Primary camera megapixels range from 0 to 20
- px_height and px_width: Screen resolution varies widely
- ram: Ranges from 373 to 3995 MB
- talk_time: Ranges from 2 to 20 hours

### Categorical Features Analysis
- blue, dual_sim, four_g, three_g, touch_screen, wifi: Binary features (0 or 1)

### Key Observations
1. Wide range of specifications across devices, indicating diverse product offerings.
2. Large variations in features like battery_power, ram, and pixel dimensions, which could significantly influence price range.
3. Binary features suggest that certain capabilities (e.g., 4G, touch screen) might impact the price range.

### Potential Relationships
- Higher battery_power, ram, and pixel dimensions might correlate with higher price ranges.
- The number of cores (n_cores) could be an indicator of device performance and price.
- Advanced features like 4G capability (four_g) might be more common in higher-priced devices.

### Data Quality
- Missing values and outliers exist and need to be handled.

### Correlation Analysis Results
- Strong positive correlations with price_range:
  * RAM (correlation coefficient ≈ 0.92)
  * Battery power (≈ 0.52)
  * Pixel height and width (both ≈ 0.49)
- Moderate positive correlations:
  * Internal memory (≈ 0.37)
  * Mobile weight (≈ 0.33)
- Weak to no correlation:
  * Clock speed, talk time, and screen dimensions show weak correlations

### Distribution of Numerical Features
- Battery power: Right-skewed distribution, most devices between 500-1500 mAh
- Clock speed: Bimodal distribution, peaks around 0.5 and 2.0 GHz
- Internal memory: Right-skewed, most devices have less than 32 GB
- RAM: Relatively uniform distribution from 500 to 4000 MB
- Pixel dimensions: Both height and width show multimodal distributions

### Balance of Classes in price_range
- The target variable appears to be perfectly balanced:
  * 0 (low cost): 25% of the data
  * 1 (medium cost): 25% of the data
  * 2 (high cost): 25% of the data
  * 3 (very high cost): 25% of the data

### Potential Multicollinearity
- Strong correlation between px_height and px_width (≈ 0.93)
- Moderate correlation between battery_power and px_height/px_width (≈ 0.55)
- Moderate correlation between RAM and battery_power (≈ 0.52)

### Feature Engineering Suggestions
- Pixel density: (px_height * px_width) / (sc_h * sc_w)
- Screen area: sc_h * sc_w
- Battery efficiency: battery_power / mobile_wt
- Performance index: (ram * clock_speed * n_cores) / 1000
- Camera quality index: (fc + pc) / 2
- Connectivity score: sum of blue, wifi, three_g, four_g

To run EDA, run the following command and then check `results/data_exploration_results` folder.

    'cd utils'
    `python EDA.py`


## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and write tests.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.