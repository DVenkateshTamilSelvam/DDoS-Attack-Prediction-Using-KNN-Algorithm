# DDoS Attack Prediction Using KNN Algorithm
 

## Overview

This project is a DDoS (Distributed Denial of Service) attack detection system built using Python. The system uses a machine learning model to predict whether a given network request is normal or potentially part of a DDoS attack. The project includes both a machine learning component for training the detection model and a graphical user interface (GUI) built with Kivy for real-time analysis.

## Features

- **Machine Learning Model**: A K-Nearest Neighbors (KNN) classifier is used to detect DDoS attacks based on network traffic data.
- **Real-time Detection**: The GUI allows users to upload network request files and analyze them for potential DDoS attacks.
- **Visual Feedback**: The GUI displays results with color-coded messages—green for normal traffic and red for detected DDoS attacks.

## Project Structure

```
├── ddos_detection.py         # Main Kivy application file for DDoS detection
├── training.py               # Script for training the machine learning model
├── knnpickle_file            # Serialized KNN model file
├── ddos_dataset.csv          # Sample dataset used for training/testing
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Required Python libraries: Kivy, NumPy, Pandas, Scikit-learn

You can install the required libraries using:

```bash
pip install kivy numpy pandas scikit-learn
```

### Running the Project

1. **Training the Model**:
   - The `training.py` script is used to train the KNN model using the provided dataset. The trained model is saved as `knnpickle_file`.
   - To run the training script, use:
   
   ```bash
   python training.py
   ```

2. **Running the GUI Application**:
   - The `ddos_detection.py` file is the main application that provides a GUI for real-time DDoS detection.
   - To start the application, use:

   ```bash
   python ddos_detection.py
   ```

3. **Using the Application**:
   - Upon running the application, you can upload a CSV file containing network request data. The system will analyze the data and provide feedback on whether a DDoS attack is detected.

## Dataset

The project uses a dataset (`ddos_dataset.csv`) containing network traffic data. The dataset includes features like `N_IN_Conn_P_SrcIP`, which are used by the KNN model to classify network requests.

### Data Columns

- `attack`: Indicates whether the request is part of a DDoS attack (`1`) or normal traffic (`0`).
- `N_IN_Conn_P_SrcIP`: Number of incoming connections per source IP (used as a feature for detection).
- *[Add descriptions for other relevant features]*

## Model Details

The model used in this project is a K-Nearest Neighbors (KNN) classifier. Key parameters:

- **n_neighbors**: 2
- **weights**: 'distance'
- **algorithm**: 'brute'

### Evaluation Metrics

- **Accuracy**: [Include the accuracy score obtained during testing]
- **Precision**: [Include precision score]
- **Recall**: [Include recall score]
- **F1 Score**: [Include F1 score]
- **Cohen's Kappa**: [Include Cohen's kappa score]

## Future Enhancements

- **Additional Models**: Experiment with other machine learning models like SVM, Random Forest, etc.
- **Extended GUI Features**: Include more detailed analysis, real-time traffic monitoring, and logging.
- **Deployment**: Deploy the application on a server for real-time network monitoring.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [Kivy](https://kivy.org/) for the GUI framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
