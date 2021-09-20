# Prediction of photoelectric effect log and facies classification in the Panoma gas field

by
Felipe F. Melo 

## About

This work was accepted to the 17th International Congress of the Brazilian Geophysical Society and EXPOGEf.

This repository contains the source code to perform the results presented. The code is compatible with Python 3.7 programming language.

## Abstract

I present a machine learning application to perform log prediction and facies classification in the Panoma gas field. The training set is composed of two wells without the photoelectric effect (PE) log and one well with missing values. Before predicting the PE log in two wells, I deal with a few missing data. To predict the logs, I perform feature augmentation in the input logs and generate new logs with a polynomial combination and a low pass filter in wavelet domain. Then, I predict the PE log with the random forest algorithm. In this case, the nested cross validation is used to model selection and hyperparameters tuning. The well Churchman Bible is picked as test well and the score of 72% is achieved on the PE prediction. Both predicted logs are aggregated to the input logs and a new feature augmentation is performed. The new training data is generated aggregating features at neighboring depths and with the vertical gradient. For classification, I used the extreme gradient boosting algorithm and the leave two wells out cross validation to model selection and hyperparameter tuning. The score of 58% is achieved in facies classification on the same test set as prediction.

## Content

The notebooks `1_display_data.ipynb` and `2_predict_PE_Facies_Classification.ipynb` show the results. The other notebooks are used to compute the correct parameters. 

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.
