# Expectation_Maximization
Experimentation of Fitting Face data to different models using Expectation Maximization Algorithm 

Modeling of face images with different model types was done for this experiment. 
Parameters were learnt using Expectation Maximation Algorithm.
This algorithm was written for scratch without using api's from scipy/ sklearn 

Models Implemented 
1. Single Gaussian Model
2. Mixture of Gaussian Models 
3. t-Distribution 
4. Mixture of t- Distributions 
5. Factor Analysis 

The code was written in python 2.7 using cv2, numpy scipy, sklearn, matplotlib libraries 
Place your face image data in positive and negative image folders respectively.  
There were numerical issues when the algorithm was run on higher image dimentions becasue of the problem of 
computing determinant and inverse of matrix, during pdf estimation.

The code was tested with [5,5] greyscale image dimentions. For a few other cases, it was seen to be running at higher dimentions as well 
Sample outputs could be found in models folder 
