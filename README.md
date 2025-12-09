# Student Dropout Prediction
**Darya Yarparvar** | November 2025
______________________________________________________________

## Problem Statement
We aim to help ********** anticipate student dropout by developing a predictive model. The insights will support them in identifying key metrics to monitor and enabling timely interventions to retain more students. This, in turn, helps reduce the risk of financial instability, revenue loss, reputational damage, and student dissatisfaction.
## Dataset
The data was provided in three distinct stages (Table 1).

<img width="986" height="337" alt="Screenshot 2025-12-09 at 18 16 18" src="https://github.com/user-attachments/assets/7e5fe6a1-0605-4ff0-9214-e517821821e2" />

Table 1. Overview of the three data Stages, including information type, actionability, and volume (Data provided by **********). 

## Data Cleaning and Feature Engineering
No duplicated records were detected. Data was cleaned by removing irrelevant features (LearnerCode) and those with missing rates higher than 50% (DiscountType, State). CourseLevel was mapped to numeric values to reflect its inherent ordinal nature. Before discarding the features with cardinality higher than 200, the following features were created to keep as much information as possible:
- Age: derived from the student’s birth date
- Subject Group: CourseName aggregated into fewer subject categories
- Degree Group: Degree aggregated into fewer broad degree categories
- University: University aggregated into fewer educational centres
- Pass Rate: The ratio of passed modules to assessed modules



## Exploratory Data Analysis
The dataset is highly imbalanced (dropout prevalence 15%), which may affect model performance. Numeric features are skewed and contain outliers, with variables such as unauthorised absence count and pass rate expected to be strong predictors of dropout (Table 2; Figure 1-2).

<img width="858" height="398" alt="Screenshot 2025-12-09 at 18 20 00" src="https://github.com/user-attachments/assets/10a56465-3f31-43da-b023-e846ca3b9f0b" />

Table 2. Descriptive statistics of the numeric features

<img width="469" height="215" alt="Screenshot 2025-12-09 at 18 22 16" src="https://github.com/user-attachments/assets/9e24f7d2-62ab-4a8e-afde-9e594c72340d" />

Figure 1. A highly imbalanced dataset, with the dropout group comprising only 15% of the observations (Prevalence=0.15) may adversely affect model performance. Although we could up-sample the minority class, we would perform modelling with the unaltered data and only revisit imbalance handling if needed.

<img width="952" height="635" alt="Screenshot 2025-12-09 at 18 22 46" src="https://github.com/user-attachments/assets/3d526930-db3a-4c41-a193-42216e72e2d7" />

<img width="960" height="604" alt="Screenshot 2025-12-09 at 18 22 59" src="https://github.com/user-attachments/assets/38a498bd-88c7-4e2e-820a-20a8821c89a0" />

Figure 2. Numeric features are non-normally distributed. While they contain outliers, these may hold useful signals, so they were retained. Variables such as unauthorised absence count and pass rate are expected to be strong predictors of dropout as they show clear relationships with dropout.


## Modelling
Neural Network (NN) and XGBoost techniques were used to create predictive models for dropout at each Stage (Rebelo Marcolino et al., 2025).

### Data Pre-processing
The dataset was stratified on the target variable to preserve the integrity of its class distribution, and then split into training, validation, and test set (20%).

The data used for NN modelling was prepared by performing one-hot encoding and scaling (StandardScaler). Missing numeric values were imputed (SimpleImputer(strategy="median")). Missing nominal features were retained as a separate class. The last three steps were not necessary for XGBoost. As a tree-based model, XGBoost can handle missing values and nominal data and is not affected by scale differences among features. 

Outliers were retained since they may represent the minority dropout class. XGBoost’s inherent robustness to feature magnitude, together with the NN’s preprocessing scaling, ensures these values are handled reliably.
For reproducibility, all random processes were fixed using a constant seed.

### Neural Network 
A sequential NN was fitted and compiled with the baseline configuration. The model was trained on each Stage’s training data and its performance was monitored on the validation data. A 0.5 dropout and an early-stopping callback were used to regularise the model, prevent overfitting, and save computational resources (Table 3).

<img width="758" height="1089" alt="Screenshot 2025-12-09 at 18 25 06" src="https://github.com/user-attachments/assets/b7aa0d3e-e611-4e3f-8e06-6be8f00b7a39" />

Table 3. Baseline Neural Network architecture, training parameters, and evaluation metrics.

### XGBoost
XGBoost was trained on each Stage’s training data using the baseline configuration, and its performance was monitored on the validation data (Table 6).

<img width="757" height="611" alt="Screenshot 2025-12-09 at 18 25 26" src="https://github.com/user-attachments/assets/1375d711-2df1-4b77-8235-910ae2783d88" />

Table 4. Baseline XGBoost hyperparameters and evaluation metrics.

### Hyperparameter Tuning
Hyperparameter tuning was performed using an exhaustive GridSearch over the specified hyperparameters (Table 5-6) to improve each model’s performance. At each Stage, the best hyperparameters were selected based on the highest PR-AUC recorded on the validation data (Table 7 & 9). 

PR-AUC was used as the main evaluation metric because it captures the balance between recall and precision across all possible thresholds. It is the optimal choice for binary classification with strong class imbalance, as it does not include True Negatives, which are very common in this dataset and can easily hide weak performance on the minority class (Davis & Goadrich, 2006). The baseline PR-AUC is equal to the Prevalence, which in this case is 0.15.

<img width="723" height="513" alt="Screenshot 2025-12-09 at 18 26 06" src="https://github.com/user-attachments/assets/6754172a-5f1c-4256-a6fe-64ec51e5bcd4" />

Table 5. Neural Network hyperparameter grid exploring different architectures, optimisers, learning rates, and activation functions.

<img width="720" height="510" alt="Screenshot 2025-12-09 at 18 26 27" src="https://github.com/user-attachments/assets/c939dee1-f789-42b0-bf21-09c841553396" />

Table 6. XGBoost hyperparameter grid exploring different tree depths, learning rates, and ensemble sizes. 

## Results

### Neural Network
Across all stages, the NN learned quickly and showed no signs of overfitting. The baseline NN models already performed strongly, and hyperparameter tuning produced only modest gains, if any, indicating that even a simple architecture was capable of capturing most of the underlying structure without extensive optimisation (Table 8). 

The largest improvements in predictive performance came not from tuning but from the addition of informative data: engagement-related features boosted Stage 1 performance noticeably, while academic performance produced the strongest recall gains from Stage 2 to Stage 3. Since the tuned models shared the same optimal configuration across all stages, these improvements can be attributed entirely to the predictive value of the additional data (Table 9). SHAP analysis mirrored these patterns, with engagement dominating Stage 2 and academic performance emerging as the strongest predictor in Stage 3 (Fig. 3-5).

<img width="743" height="160" alt="Screenshot 2025-12-09 at 18 26 49" src="https://github.com/user-attachments/assets/7895ea29-5414-47e2-9a01-17c3299faeff" />

Table 7. Final optimised Neural Network hyperparameters. The same configuration emerges as optimal in all stages.

<img width="743" height="267" alt="Screenshot 2025-12-09 at 18 27 05" src="https://github.com/user-attachments/assets/77f6f9e6-a280-4548-b047-a317ae6269e1" />

Table 8. Baseline vs tuned model performance across stages for Neural Network. Hyperparameter tuning yields only modest gains at each stage, indicating that most learnable structure is already captured by the baseline networks. All models achieve PR-AUC well above the baseline for this imbalanced dataset (prevalence=0.15), indicating that they capture meaningful patterns. ROC-AUC values exceed the 0.5 baseline of a random classifier, confirming acceptable discriminative ability. As expected, accuracy is relatively high across all Stages, reflecting an inflated estimate of the model's true performance. [Colour scales are matched to allow direct comparison between NN and XGBoost.]

<img width="749" height="237" alt="Screenshot 2025-12-09 at 18 27 29" src="https://github.com/user-attachments/assets/b0284112-86db-49d0-8bb3-aba80288c0a1" />

Table 9. Stage-to-stage performance gains for Neural Network. Engagement data (Stage 2) improves early detection of at-risk learners, while academic performance data (Stage 3) yields the largest recall gains and overall separation between dropout and non-dropout classes. [Colour scales are matched to allow direct comparison between NN and XGBoost.] * Since the tuning process arrived at the same model configuration across all Stages, we can conclude that the improvement in tuned model performance from Stage 1 to Stage 3 is driven solely by the addition of valuable data.

<img width="755" height="572" alt="Screenshot 2025-12-09 at 18 27 56" src="https://github.com/user-attachments/assets/110b42ea-b2ec-4f6b-a0f3-d17f3d27f8b7" />

Figure 3. SHAP-based interpretation of dropout predictors for Stage 1 tuned Neural Network model. Learners from China or Hong Kong show lower dropout risk, while Bangladeshi and Indian learners show higher risk. Lower course levels are linked to increased risk. Learners progressing to University of Sheffield International College, Durham University, the University of Leeds, or the University of Huddersfield are less prone to dropout, as are those located at Sheffield, Durham, Sussex, Huddersfield, or Kingston ISC centres. Older and male learners show higher risk, while those studying engineering or science are less prone to dropout. A large high-risk group with missing degree information stands out and raises the need for further investigation, as it may reflect hidden structure or missing-not-at-random effects.

<img width="755" height="571" alt="Screenshot 2025-12-09 at 18 28 10" src="https://github.com/user-attachments/assets/ae2882a8-56aa-4a78-b682-c21246834166" />

Figure 4. SHAP-based interpretation of dropout predictors for Stage 2 tuned Neural Network model. The higher the unauthorised absence count and the lower the authorised absence count, the higher the risk for dropout. Learners from China or Hong Kong show lower dropout risk, while Bangladeshi and Indian learners show higher risk; the pattern for Pakistani learners is not clearly indicated. Lower course levels are linked to increased risk. Learners progressing to the University of Sheffield International College, Kingston University London, or Lancaster University are less prone to dropout, as are those located at Sheffield, Durham, or Kingston ISC centres. Older and male learners show higher risk, while those studying engineering or science, those in their first intake, and those who found out about the course via a sponsor are less prone to dropout. A large high-risk group with missing degree information stands out and raises the need for further investigation, suggesting possible hidden structure or missing-not-at-random patterns. Newly added engagement-related features appear as the most influential predictors, consistent with the 12.6% performance improvement observed from Stage 1 to Stage 2, and their association with dropout outcome, as observed in the EDA.

<img width="755" height="573" alt="Screenshot 2025-12-09 at 18 28 26" src="https://github.com/user-attachments/assets/e96b049f-0d79-41ef-881b-a1fd1e57cda5" />

Figure 5. SHAP-based interpretation of dropout predictors for Stage 3 tuned Neural Network model. Learners with low pass rate have higher risk of dropout. The higher the unauthorised absence count and the lower the authorised absence count, the higher the risk for dropout. Lower course levels are linked to increased risk. Learners from China, Hong Kong, or Pakistan show lower dropout risk; the pattern for Bangladeshi or Indian learners is not clearly indicated. Learners progressing to the University of Sheffield International College, Kingston University London, or Durham University are less prone to dropout, as are those located at Sheffield, Kingston, Huddersfield, or Durham ISC centres. Older learners show higher risk, while those studying engineering or science, those in their first intake are less prone to dropout. A large high-risk group with missing degree information stands out and raises the need for further investigation, suggesting possible hidden structure or missing-not-at-random patterns. The newly added academic performance feature appears as the most influential predictor, followed by the engagement-related features added at Stage 2. This is consistent with the 10.1% performance improvement and the largest recall gains observed from Stage 2 to Stage 3. Pass rate’s dominant importance at Stage 3 aligns with its association with dropout outcome, as observed in the EDA.


### XGBoost
The baseline XGBoost models generalised well with no signs of overfitting, except at Stage 2, where overfitting was mitigated through hyperparameter tuning. The tuning provided only marginal improvements at each stage, less pronounced than for the NNs (Table 9). This aligns with XGBoost’s strength in handling imbalanced, medium-sized tabular data (Grinsztajn, Oyallon and Varoquaux, 2022).

Similar to the NN modelling, performance improved steadily as new features were introduced, confirming that gains were driven by valuable data rather than by changes in model configuration. Academic performance data boosted the predictive power to a greater extent in comparison with engagement-related data (Table 10). Feature importance analysis mirrored these patterns, with engagement-related features dominating Stage 2 and academic performance emerging as the strongest predictor in Stage 3 (Fig. 6; Table 12-14).

<img width="742" height="144" alt="Screenshot 2025-12-09 at 18 28 47" src="https://github.com/user-attachments/assets/58bb183a-197d-4d07-b248-7d8e18fd5d98" />

Table 9. Final optimised XGBoost hyperparameters. From Stage 1 to Stage 3, the depth increases and the learning rate decreases, indicating a move towards a deeper, more complex model that needs slower, careful learning to fine-tune and avoid overfitting.

<img width="746" height="248" alt="Screenshot 2025-12-09 at 18 29 03" src="https://github.com/user-attachments/assets/c73fe16d-3045-419a-9869-f713dab185f6" />

Table 10. Baseline vs tuned model performance across stages for XGBoost. For Stage 1, hyperparameter tuning marginally improved model performance. For Stage 2, tuning produced minimal performance gains but successfully mitigated the baseline model’s overfitting. For Stage 3, tuning failed to improve performance, indicating that most learnable structure is already captured by the baseline models. All models achieve PR-AUC well above the baseline for this imbalanced dataset (prevalence=0.15), indicating that they capture meaningful patterns. ROC-AUC values exceed the 0.5 baseline of a random classifier, confirming acceptable discriminative ability. As expected, accuracy is relatively high across all Stages, reflecting an inflated estimate of the model's true performance. [Colour scales are matched to allow direct comparison between NN and XGBoost.] * overfitting

<img width="746" height="235" alt="Screenshot 2025-12-09 at 18 29 16" src="https://github.com/user-attachments/assets/3fc04be6-788b-4fdc-b0fd-1d4651ad41ea" />

Table 11. Stage-to-stage performance gains for XGBoost. Engagement data (Stage 2) improves early detection of at-risk learners, while academic performance data (Stage 3) yields the largest recall gains and overall separation between dropout and non-dropout classes. [Colour scales are matched to allow direct comparison between NN and XGBoost.] * Since the tuning process did not arrive at the same model configuration across all Stages, performance gains from Stage 1 to Stage 3 cannot be attributed solely to the addition of new data. However, if we assume that tuning has effectively extracted most of the signal available at each Stage, then comparing the Stages remains meaningful. In fact, the improvement pattern closely mirrors what we observed with the baseline models, reinforcing the same underlying interpretation.

<img width="875" height="1096" alt="Screenshot 2025-12-09 at 18 29 54" src="https://github.com/user-attachments/assets/f6950122-d436-490a-b3c1-675324bcda67" />

Figure 6. Baseline (left) vs tuned (right) models’ feature importance ranked by predictive gain. A 90% cumulative importance cutoff line (dashed red line) separates the features contributing to 90% of the predictive power. Nationality, university, degree, gender, booking type, course level, and subject dominated Stage 1 (top), then engagement variables (unauthorised and authorised absence) became highly influential in Stage 2 (middle), and finally pass rate overwhelmingly drove Stage 3 performance. For Stages 1 and 2, hyperparameter tuning achieved a modest redistribution of feature importance, resulting in a more balanced model. For Stage 3, tuning failed to improve feature distribution, leaving the model heavily dependent on a single dominant feature and vulnerable to potential data unavailability. Pass rate’s dominant importance at Stage 3 aligns with its association with dropout outcome, as observed in the EDA.

<img width="562" height="366" alt="Screenshot 2025-12-09 at 18 30 14" src="https://github.com/user-attachments/assets/6f2f26f7-9daa-492a-80bc-ebecf83aede7" />

Table 12. Feature importance ranked by predictive gain; Stage 1. A 90% cumulative importance cutoff line (red line) separates the features contributing to 90% of the predictive power. Nationality, university, degree, gender, booking type, course level, and subject dominated Stage 1. [Colour scales are matched to allow direct comparison between Stages.]

<img width="561" height="419" alt="Screenshot 2025-12-09 at 18 30 36" src="https://github.com/user-attachments/assets/9ccc0096-06b5-465a-a549-25726ba88234" />

Table 13. Feature importance ranked by predictive gain; Stage 2. A 90% cumulative importance cutoff line (red line) separates the features contributing to 90% of the predictive power. With the addition of the engagement-related data at Stage 2 (blue), unauthorised and authorised absences deprioritise some of the features that were previously prominent in Stage 1. [Colour scales are matched to allow direct comparison between Stages.]

<img width="561" height="447" alt="Screenshot 2025-12-09 at 18 30 58" src="https://github.com/user-attachments/assets/26b09809-cc4e-468f-9b8d-8f64e9e72626" />

Table 14. Feature importance ranked by predictive gain; Stage 2. A 90% cumulative importance cutoff line (red line) separates the features contributing to 90% of the predictive power. With the addition of the academic performance data at Stage 3 (pink), pass rate becomes overwhelmingly important, making the model heavily reliant on a single dominant feature and vulnerable to potential data unavailability. Pass rate’s dominant importance at Stage 3 aligns with its association with dropout outcome, as observed in the EDA. [Colour scales are matched to allow direct comparison between Stages.]

## Conclusions
Overall, academic performance is the primary and most influential predictor, though engagement-related data also offers strong predictive value, particularly when academic performance information is unavailable. The lower PR-AUC at Stage 1, compared with the later stages, reflects the limited effectiveness of application data on its own for identifying learners at risk.

At Stage 1, the model depends only on application data, which are weak predictors of dropout and often linked to sensitive characteristics like nationality, gender, or socioeconomic background. Therefore, the model is more exposed to biased patterns and uneven errors across different groups. This means we need error analysis and fairness checks to make sure early interventions don’t consistently miss or misclassify certain learners. This helps maintain fairness and stops the model from repeating existing structural inequalities in the learner population. Also, a large high-risk group with missing degree information is evident, raising the need for further investigation, as it may reflect hidden structures or missing-not-at-random patterns (Fig. 3-5).

For student dropout prediction on a highly imbalanced dataset, we recommend an ensemble approach: NNs for Stage 1 and Stage 2, and XGBoost for Stage 3. NNs seem to handle the complex, non-linear interactions of application and engagement-related data better at the early stages, when the strong academic signal is not yet available. They achieve stable and balanced predictions early on (Stage 1 PR-AUC: 0.67, Stage 2 PR-AUC: 0.75). At Stage 3, XGBoost clearly performs best (PR-AUC: 0.87, ROC-AUC: 0.96, Precision: 0.93, Recall: 0.92, F1: 0.92), showing that it responds most effectively to academic performance data for final high-confidence intervention decisions (Table 15).

<img width="727" height="282" alt="Screenshot 2025-12-09 at 18 31 19" src="https://github.com/user-attachments/assets/1f3c794d-94c1-4588-849a-e2db78084c1b" />

Table 15. Recommended final models to predict student dropout.
This strategy naturally supports a two-phase intervention system based on the trade-off between timing and impact. The staged approach also prevents less prominent yet meaningful features, such as course level, degree, centre, university, subject, nationality, and engagement-related features from being masked by the overwhelming predictive power of academic performance data.

- Early, low-cost, proactive interventions, like check-ins or alerts can be started as soon as engagement signals become available at Stage 2. This point gives the largest early jump in predictive power (+12.6% PR-AUC).
- More resource-intensive interventions like tutoring or counselling can be offered to learners flagged at Stage 3, when academic performance data becomes available. XGBoost’s strong class separation at this stage helps confidently identify learners with high risk of dropout.
  
The current decision threshold of 0.5 treats Precision and Recall as equally important. This should be reviewed with stakeholders and, if necessary, adjusted to reflect available resources and the trade-off between incorrectly flagged cases and overlooked dropout cases. 

One important limitation is that XGBoost becomes extremely dependent on a single predictor at Stage 3 (Fig. 6). Pass rate, with an importance of 81.2%, dominates the model, leaving the model vulnerable to missing or unreliable data (Table 14). 

To reduce the black box problem of NN and XGBoost models, explainability tools such as SHAP are necessary to support the decision-making process and to maintain transparency.


## References
Clevert, D.-A., Unterthiner, T. and Hochreiter, S. (2016) Fast and accurate deep network learning by exponential linear units (elus), arXiv.org. Available at: https://arxiv.org/abs/1511.07289 (Accessed: 16 November 2025). 

Davis, J. and Goadrich, M. (2006) ‘The relationship between precision-recall and ROC curves’, Proceedings of the 23rd international conference on Machine learning  - ICML ’06, pp. 233–240. doi:10.1145/1143844.1143874. 

Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). ‘Why do tree-based models still outperform deep learning on tabular data?’, 36th Conference on Neural Information Processing Systems (NeurIPS 2022), doi:https://doi.org/10.48550/arXiv.2207.08815.

Rebelo Marcolino, M., Reis Porto, T., Thompsen Primo, T., Targino, R., Ramos, V., Marques Queiroga, E., Munoz, R. and Cechinel, C. (2025). ‘Student dropout prediction through machine learning optimization: insights from moodle log data’, Scientific reports, [online] 15(1), p.9840. doi:https://doi.org/10.1038/s41598-025-93918-1.


















_____________________________________________________________________________________________________
_Word count (main text only, excluding cover, tables, figures, captions, and references): ~ 1300 words_
