# deep-learning-challenge

## Overview
This project aims to assist the nonprofit foundation Alphabet Soup in selecting the applicants for funding who have the best chance of success in their ventures. Leveraging machine learning and neural networks, a binary classifier is developed to predict the likelihood of success for funded applicants based on provided dataset features.

### Purpose
The main goal is to analyze historical data from over 34,000 organizations that received funding from Alphabet Soup. By creating an efficient model, Alphabet Soup can make more informed decisions, potentially leading to more successful ventures funded by the foundation.

## Results

### Data Preprocessing
- **Target Variable**: IS_SUCCESSFUL column, indicating whether the money was used effectively.
- **Feature Variables**: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.
- **Removed Variables**: EIN and NAME, as they do not contribute to the outcome of the funding success.

### Compiling, Training, and Evaluating the Model
- **Model Design**: The neural network model was designed with multiple layers, including input, hidden, and output layers. The number of neurons and layers was determined based on the complexity of the dataset and the feature count.
  - **Hidden Layers and Neurons**: Experimented with different configurations to optimize performance.
  - **Activation Functions**: Used ReLU for the hidden layers due to its efficiency and simplicity, and Sigmoid for the output layer since this is a binary classification problem.
- **Performance**: Initial models did not achieve the target performance. Various strategies, including adjusting the network architecture, modifying the input data, and experimenting with training epochs, were employed to improve the model.
- **Optimization Attempts**: Included adding more data, increasing hidden layers, changing activation functions, and adjusting epochs.

### Model Optimization
- After several iterations, the optimized model achieved a predictive accuracy higher than 75%, meeting the project goal.
- **Optimization Techniques**: Included pruning less relevant features, adjusting the model architecture, and fine-tuning the training process.

## Summary
The deep learning model developed for Alphabet Soup has shown promising results in predicting the success of funded applicants. While the initial models struggled to meet the target performance, through systematic optimization, we achieved a satisfactory predictive accuracy.

### Recommendations
- **Experiment with Alternative Models**: Considering other models like Random Forest or Gradient Boosting might offer insights and possibly better performance due to their different underlying mechanisms in handling feature relationships and complexities.
- **Feature Engineering**: Further analysis and engineering of the input features could improve model performance by highlighting essential patterns and relationships.

## Files
- `AlphabetSoupCharity.ipynb`: Notebook with the initial model development and evaluation.
- `AlphabetSoupCharity_Optimization.ipynb`: Notebook with the optimized model.
- `charity_data.csv`: Dataset used for the model.

## Instructions for Running the Project
1. Clone the repository to your local machine.
2. Ensure you have Python and necessary libraries installed, including TensorFlow, Pandas, and scikit-learn.
3. Open the notebooks in Google Colab or Jupyter Notebook to view the models and run them.

## Conclusion
This project highlights the potential of neural networks in making predictive analyses for nonprofit funding. With continued refinement and adaptation, the model can become an indispensable tool for Alphabet Soup, helping to channel resources towards the most promising ventures.
