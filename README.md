# Predicting Antimicrobial Resistance (AMR) in *Salmonella enterica* Using Deep Learning  

## Overview  
This project aims to predict **Antimicrobial Resistance (AMR)** in *Salmonella enterica*, a bacterial pathogen commonly found in poultry, using machine learning techniques. By leveraging a **Multi-Layer Perceptron (MLP)** model, the study achieves high accuracy in classifying resistance to various antibiotics, helping to improve food safety and guide treatment strategies.

---

## Key Features  
- Classification of AMR in *Salmonella enterica* strains isolated from poultry.  
- Use of **deep learning models** to predict resistance across 11 antibiotic classes.  
- Implementation of **exploratory data analysis (EDA)**, including t-SNE visualization, to identify resistance patterns.  
- Robust evaluation using metrics like accuracy, precision, recall, F1-score, and Cohen’s Kappa.  

---

## Dataset  
- **Source**: NCBI Genome Database  
- **Description**: 496 genomes of *Salmonella enterica* isolated from chicken samples.  
- **Preprocessing**: Removal of duplicates and labeling of samples using the CARD-Resistance Gene Identifier.  

---

## Methodology  
1. **Data Preprocessing**:  
   - Removal of duplicates and preparation of metadata.  
   - Labeling drug classes for resistance genes.  

2. **Exploratory Data Analysis (EDA)**:  
   - Data wrangling and visualization using t-SNE to reveal AMR gene clusters.  

3. **Model Training**:  
   - Evaluation of multiple models, including SVM, KNN, Decision Trees, Random Forest, Hidden Markov Models, and MLP.  
   - 3-fold cross-validation for robust evaluation.  

4. **Evaluation Metrics**:  
   - Accuracy, Precision, Recall, F1-Score, and Cohen’s Kappa.  

---

## Results  
The **Multi-Layer Perceptron (MLP)** model outperformed other approaches with:  
- **Accuracy**: 86.7%  
- **Precision**: 86.4%  
- **Recall**: 86.1%  
- **F1-Score**: 85.8%  
- **Cohen’s Kappa**: 85.4%  

The results demonstrate the model's robustness and its potential for reliable AMR detection.  

---

## Technologies Used  
- **Python**: For model training and evaluation.  
- **Machine Learning Libraries**: Scikit-learn, NumPy, Pandas.  
- **Deep Learning Frameworks**: TensorFlow/Keras.  
- **Visualization Tools**: Matplotlib, Seaborn, t-SNE.

---

## Conclusion  
This project highlights the potential of machine learning, particularly deep learning models, in combating **Antimicrobial Resistance (AMR)**. The findings can improve surveillance systems, guide effective interventions in poultry farming, and reduce the risks associated with drug-resistant bacteria.  

---

