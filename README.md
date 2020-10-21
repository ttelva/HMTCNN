# HMTCNN
What is the code?
This code is used for matrix-assisted laser desorption ionization-time of flight mass spectrometry-based bacterial identification. The information of the helix matrix transformation combined with convolutional neural network algorithm can be found in the article entitled “Helix matrix transformation combined with convolutional neural network algorithm for matrix-assisted laser desorption ionization-time of flight mass spectrometry-based bacterial identification”, which is published in Frontiers in Microbiology.

How to use?
1. Build an environment for model training. The requirements of both hardware and software can be found in the Method section of the article. We provide a data sample file to test the environment. The environment is OK if the code can run in its entirety.
2. Prepare your MALDI-TOF MS spectrum dataset in a y value only data format then write into a text file. Each line in the text file contains one MS spectrum data. The y values in a line are split by comma. The number of label is located at the end of the line, which is split with y values by semicolon. For example, “y1,y2,y3,…yn-1,yn;label” is a line of dataset. 
3. Revise parameters in the code. The length of y value list is 2,500 in this sample code, which can be changed depending on your data length. Notice that, the size of helix matrix, the size of the input of neural network and other relevant parameters have to be revised if you use a dataset with different data length. The shape of helix matrix is not necessarily a square. A rectangle is also accepted. The shape can be altered as you need.
4. Optimize the parameters in convolutional neural network. The parameters of filter and kernel size, number of convolution layers, dense, and functions can be optimized to obtain a better model.

THANKS!
