# Air_quality_index_pred
# EcoGuard: Advanced Air Quality Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EcoGuard is a comprehensive machine learning system designed to predict air quality in urban environments. The project utilizes multiple advanced machine learning models to forecast air quality parameters based on various pollutants, helping to address significant health risks and environmental challenges in urban areas.

## Features

- **Multi-model Approach**: Implements and compares Random Forest, XGBoost, and Neural Network models
- **Data-driven Insights**: Identifies key pollutants affecting air quality through feature importance analysis
- **High Accuracy**: Achieves R-squared scores of up to 0.89 in air quality prediction
- **Comprehensive Analysis**: Processes multiple air pollutants including PM2.5, PM10, NO2, SO2, CO, and others

## Dataset

The project uses air quality data from India, available on Kaggle:
[Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

## Methodology

![Methodology Diagram](media/image2.jpg)

### Data Preprocessing

- Cleaned and normalized data from government monitoring stations and IoT devices
- Handled missing values using appropriate statistical methods
- Applied feature selection to identify the most important predictors

### Machine Learning Models

1. **Random Forest**
   - Ensemble learning technique using multiple decision trees
   - R-squared Score: 0.893

2. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting algorithm optimized for performance
   - R-squared Score: 0.895

3. **Neural Network**
   - Deep learning model with multiple dense layers
   - R-squared Score: 0.888

## Results

Among the evaluated models, the **XGBoost Regressor** demonstrated the highest performance with an R-squared score of 0.895. Both XGBoost and Neural Network outperformed the RandomForestRegressor, suggesting that more complex models are effective in capturing the underlying patterns in air quality data.

![Results Comparison](media/image5.png)

## Model Performance Visualization

![RandomForest Performance](media/image3.png)
![XGBoost and Neural Network Performance](media/image4.png)

## Key Insights

- Identified critical features contributing to air quality variations
- Discovered spatial and temporal patterns in air pollutant levels
- Highlighted the effectiveness of machine learning in air quality prediction

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/akashkatate2015/Air_quality_index_pred.git
cd Air_quality_index_pred

# Install required packages
pip install -r requirements.txt

# Run the model
python air_quality_prediction.py
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- xgboost
- tensorflow
- matplotlib
- seaborn

## Future Work

- Incorporate real-time data streams for continuous prediction
- Expand the model to include more cities and regions
- Develop a web interface for public access to predictions
- Explore advanced deep learning architectures for improved accuracy

## Conclusion

The EcoGuard project demonstrates the effectiveness of machine learning in air quality prediction, offering valuable insights for environmental monitoring and policy interventions. By leveraging advanced models like Random Forest, XGBoost, and Neural Networks, we aim to contribute to sustainable urban development and public health initiatives.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the dataset
- Contributors and researchers in the field of environmental monitoring
