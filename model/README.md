# Model Used for this Multi-class classification:
<br>
## Best model: roberta-large
<br>
* After testing multiple baseline models, roberta-large seemed to outperform all of them.
* Acheived a validation accuracy of 99.55% and a testing accuracy of 85.12%.
* Used a Custom Weighted Cross Entropy Loss function to fix class imbalance.
* Hyper-Parameters used:
	1) Epochs: 5
	2) Learning Rate: 2e-5
	3) Batch Size: 16
* Tracking all metrics using Weights and Biases Tool - found to be very useful to check if peft and quantization techniques are working on small-language models.
* Performance on test data:
	![alt text](https://github.com/saianish03/multiway_machinegen_text_clf/tree/main/model/clf_report.png)
* Confusion Matrix:
	![alt text](https://github.com/saianish03/multiway_machinegen_text_clf/tree/main/model/conf_matrix.png)