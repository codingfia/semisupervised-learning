# Seismic Elastic Parameters Inversion with Semi-supervised Machine Learning
## We propose a joint data- and physics-model-driven full-waveform inversion method based on semisupervised learning framework, which use the generated gathers data and real data to train the neural networks together. 

The conventional model-driven method is used to produce pseudo label datasets to train neural network, which can reduce the reliance on massive datasets. At the same time, physical model constraint is applied on the neural network to make the inversion result more physical interpretable. Using our method, the neural network is able to utilize information from both real data and physical models, thus it can take advantage of well-logging data to make the inversion more accurate and stable.

We combined CNN with physical equations to predict elastic parameters from seismic angle gathers. Utilize both labeled and unlabeled data for training, leveraging data-driven and physics-based errors to update network weights.

To verify the validity of our method, we perform synthetic experiments on a linear initial model and seismic data lack of low-frequency component. As the figure shown below, the more accurate inversion results was obtained by using the joint data- and physics-model-driven method.

<img width="258" alt="image" src="https://github.com/user-attachments/assets/e2afecbc-7574-4b43-a19e-ab6891da04a4">



 
