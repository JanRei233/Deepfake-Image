A Deepfake Image Detection System Utilizing EfficientNet V2 Algorithm
_________________________________________________________________________________

Dataset Used https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
==================================================
MODEL PERFORMANCE SUMMARY
==================================================
Architecture: EfficientNetV2B0 + Custom Head
Input Shape: 256x256x3
Classes: ['Fake', 'Real']
Total Parameters: 6,706,769
Test Accuracy: 0.8419
Test Precision: 0.9437
Test Recall: 0.7247
==================================================
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 efficientnetv2-b0 (Function  (None, 8, 8, 1280)       5919312   
 al)                                                             
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 512)               655872    
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 256)               131328    
                                                                 
 dropout_2 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 6,706,769
Trainable params: 787,457
Non-trainable params: 5,919,312
_________________________________________________________________
