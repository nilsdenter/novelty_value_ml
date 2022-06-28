# On the relationship of novelty and value in digitalization patents: A machine learning approach

This project contains the data, code and results used in the paper title "On the relationship of novelty and value in digitalization patents: A machine learning approach".

To reproduce the results, first, download and unzip all files contained in the Data folder.

Then follow the sequence of code.

The code "01_Supervised_machine_learning.py" imports both tables, that each contain a value variables, 12 novelty variables, and 8 control variables. In total three supervised machine learning algorithms are trained: Decision Tree, Random Forest and Multi-Layer Perceptron. In before, the input data is standardized and splitted into 80% training data and 20% hold-out data. Furthermore, the algorithms are prone to parameter settings. These settings are optimized by means of a grid search of common parameter values and a 5-fold stratified cross-validation. Then, each algorithm is trained on the top 10 percent of citations received within 7 years and the top 10 percent of market reaction values (deflated to 1982 dollars using the Consumer Price Index) (stock market reaction data was obtained from Kogan et al. 2017: Kogan, L., Papanikolaou, D., Seru, A., & Stoffman, N. (2017). Technological Innovation, Resource Allocation, and Growth*. The Quarterly Journal of Economics, 132, 665–712. doi:10.1093/qje/qjw040, https://github.com/KPSS2017/.

The code "02_Permutation_importance.py" performs the first supervised machine learning interpretation task. The permutation importance of each novelty variables is calculated for the best performing model for both perspectives (citations and stock market reactions). For both perspectives, the Multi-Layer-Perceptron performed best. 

The code "03_Partial_dependence_plots" performs the second supervised machine learning interpretation task. Given the three most important novelty variables, the code computes their relationship to technological importance (citation perspective) and economic importance (stock market reaction perspective) in more detail. 

Please see the paper for further information.
