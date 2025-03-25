# Project Report

## Abstract

Clickbait headlines are designed to captivate readers by strategically omitting key details,often leading to ambiguous or misleading inter pretations. This project addresses spoiler-type classifcation—predicting whether a clickbait post requires clarifcation via a phrase, pas sage, or multi-part spoiler—using the Webis Clickbait Spoiling Corpus 2022. Our approach combines domain-adapted (DA) word embed dings to capture clickbait semantics with three models of increasing complexity: a baseline Lo gistic Regression, a Bi-directional Long Short Term Memory Network (Bi-LSTM) with at tention mechanism, and a transformer-based ModernBERT model. Our work explores how advanced natural language processing (ANLP)can be leveraged to unravel the deceptive tactics of clickbait, thereby promoting clearer commu nication and more informed online discourse.

## 1Introduction

Clickbait headlines have become ubiquitous in on line media, designed to entice users to click through to articles by deliberately withholding key infor mation or creating a "curiosity gap" (Loewenstein,1994). These headlines often use sensationalist language, hyperbolic phrases, or deliberately am biguous language to capture attention. While effec tive at generating traffc, clickbait can lead to user frustration when the content fails to deliver on the headline’s implicit promise, contributing to infor mation pollution and undermining trust in digital media (Molyneux and Coddington, 2020).

Clickbait spoiling addresses this issue by auto matically providing the missing information that a clickbait headline intentionally omits. Rather than forcing users to click through to potentially disap pointing content, spoiling systems reveal the key information, the "spoiler", that was deliberately withheld (Hagen et al., 2022). However, not all clickbait can be spoiled in the same way. Some re quire just a short phrase, while others need a more

detailed passage to adequately explain the withheld information. Some complex cases may even re quire multiple spoilers to fully address the promise of clickbait (Fröbe et al., 2023).

In this project, we focus on the task of clickbait spoiler type classifcation—determining whether a given clickbait post requires a phrase, passage,or multi-part spoiler. This classifcation serves as a crucial frst step in a complete clickbait spoiling system, allowing subsequent processes to generate appropriate spoilers based on the identifed type.By correctly classifying the spoiler type needed,we can more effectively counter clickbait tactics and provide users with the information they seek without unnecessary clicks.

This task has practical implications for improv ing information access and digital literacy. When implemented in browser extensions or social media platforms, automated spoiling systems could help users quickly assess whether content is worth their attention, reducing the effectiveness of misleading clickbait strategies. Additionally, the ability to iden tify and classify clickbait contributes to broader efforts to improve online content quality and trans parency, building upon earlier work in clickbait detection (Potthast et al., 2016; Chakraborty et al.,2016).

In this project, we explored three models of in creasing complexity to tackle this classifcation challenge: a baseline Logistic Regression model using TF-IDF features, a Bidirectional Long Short Term Memory (BiLSTM) network with attention mechanism to better capture contextual informa tion, and a transformer-based ModernBERT model.We also investigated the impact of domain-adapted word embeddings specifcally tuned to clickbait linguistic patterns.

Through this work, we aim to develop an effec tive approach to clickbait spoiler type classifcation while gaining deeper insights into the linguistic characteristics that distinguish different forms of

clickbait, ultimately contributing to more transpar ent and honest online communication.

## 2Related Work

The foundation of our approach builds upon re cent advancements in clickbait spoiling and text classifcation techniques.

(Hagen et al., 2022) introduced clickbait spoil ing as a two-step process: frst classifying the spoiler type (phrase, passage, or multipart), then extracting the appropriate spoiler. Their evaluation demonstrated that state-of-the-art question answer ing models, particularly DeBERTa-large, outper formed passage retrieval methods for generating both phrase and passage spoilers. Their fndings highlight the importance of spoiler type classifca tion as a frst step, as it can signifcantly improve the effectiveness of the spoiling process when done accurately.

(Fröbe et al., 2023) formalized this task through the SemEval 2023 Clickbait Spoiling competition,where RoBERTa-based classifers achieved the highest accuracy (approximately 74%) for spoiler type classifcation. Their work established standard evaluation metrics that we adopt in our methodol ogy, creating a framework for consistent compar ison with existing approaches. The strong partici pation of the competition (30 teams) demonstrated the growing interest in this research direction.

For classifcation specifcally, (Sharma et al.,2023), presented an information condensation ap proach that achieved the highest accuracy in the spoiler type classifcation. Their two-step method frst used contrastive learning to identify the most relevant paragraphs in an article, then applied De BERTa for classifcation based on this condensed information. This approach demonstrated the value of focusing on relevant content rather than process ing entire articles, aligning with our goal to develop effcient classifcation methods.

Our technical approach draws from three ad ditional works representing increasing levels of model complexity. (Sarma et al., 2018) presented domain-adapted word embeddings that combine generic and domain-specifc representations us ing Canonical Correlation Analysis (CCA). This technique is particularly relevant for capturing the unique linguistic patterns of clickbait content that may not be well represented in generic embed dings.

(Liu et al., 2024) proposed an optimized BiL-

STM network with attention for news text classif cation, capturing bidirectional dependencies and fo cusing on the most relevant textual features. Their architecture serves as our intermediate model, bal ancing complexity and performance.

Finally, (Warner et al., 2024) introduced Mod ernBERT, an advanced bidirectional encoder that improves upon previous transformer-based mod els in terms of both performance and effciency.The effectiveness of the model provides a strong foundation for our most advanced classifcation approach.

Our work extends these foundations by: (1) in vestigating whether domain-adapted embeddings can better capture clickbait-specifc linguistic pat terns, (2) implementing a BiLSTM with attention mechanism as an intermediate solution between simple logistic regression and the complex trans former models, and (3) evaluating ModernBERT’s capabilities specifcally for spoiler type classifca tion. Through this combination, we aim to advance classifcation performance while investigating the relationship between model complexity and effec tiveness for this task.

## 3Task Formalization

The clickbait spoiler type classifcation task can be formally defned as a supervised multiclass clas sifcation problem. Given a clickbait post and its linked article, the objective is to classify the post into one of three categories: phrase, passage, or multi-part spoiler. This classifcation is an essential frst step in the broader clickbait spoiling process,determining the appropriate structure of informa tion to extract in a subsequent spoiler generation phase (Hagen et al., 2022).

### 3.1Problem Defnition

Formally, we defne the task as follows:

$\text {Let}\mathcal {D}=\left\{\left(p_{i},a_{i},y_{i}\right)\right\}_{i=1}^{N}$ be a dataset consistingof N samples, where for each sample i:

• pi represents the clickbait post text

$a_{i}$ represents the linked article text

•$y_{i}\in \mathcal {Y}=${phrase, passage, multi} is theground truth label indicating the type of spoiler required

The goal is to learn a classifcation function f :PxA→Ythat maps a clickbait post and itslinked article to the appropriate spoiler type.

## 3.2Model Implementation

Our approach employs three primary modeling strategies of increasing complexity:

1. Logistic Regression with TF-IDF (Base line): A traditional approach using term fre quency features, providing interpretability and serving as a performance benchmark for more sophisticated models.

2. BiLSTM with Attention: A sequential neu ral network that processes text bidirectionally while focusing on relevant portions through an attention mechanism to capture contextual dependencies in clickbait headlines.

3. ModernBERT: A fne-tuned transformer based model leveraging pre-trained contex tual embeddings to capture semantic nuances between clickbait headlines and their corre sponding spoilers.

## 3.3Evaluation Framework

Following the PAN Clickbait Challenge at SemEval 2023 (Fröbe et al., 2023), the primary evaluation metric for this task is balanced accuracy across all three classes, which accounts for potential class imbalance. Secondary metrics include precision,recall, and F1 score for each individual spoiler type,providing a more detailed understanding of model performance.

### 4Data

This section outlines the dataset utilized for our clickbait spoiler classifcation task, along with rele vant statistics and analysis of its characteristics.

## 4.1Dataset Description

For our work, we used the Webis Clickbait Spoil ing Corpus 2022 (Hagen et al., 2022). The cor pus provides a standard split with 3,200 posts for training (80%) and 800 posts for validation (20%).While the complete corpus contains an additional 1,000 test posts, these were not available for our project work and were reserved for future evalua tion purposes. Our model development and experi mentation were therefore conducted using only the training and validation sets.

## 4.2Spoiler Types

A signifcant feature of this dataset is the catego rization of spoilers into three types, illustrated in Figure 1 :


![](https://web-api.textin.com/ocr_image/external/b85a8619a08621f6.jpg)

Figure 1: Distribution of different spoiler types

• Passage spoilers: Longer spans consisting of one or a few sentences, with an average length of 24.1 words.

• Phrase spoilers: Short spans consisting of a single word or phrase from the linked docu ment, with an average length of 2.8 words.

• Multipart Spoilers: Consisting of more than one non-consecutive phrase or passage from the linked document, with an average length of 33.9 words.

## 4.3Data Statement

Following Bender and Friedman (2018) data state ment framework, here are the additional analysis of the corpus:

CURATION RATIONALEThe corpus was cre ated to support research on automatically spoiling clickbait posts by generating short texts that sat isfy the curiosity induced by the post. The click bait posts were primarily sourced from various social media platforms: Twitter (47.5%), Reddit (36%), and Facebook (16.5%) and supplemented with posts from the Webis-Clickbait-17 corpus that were manually spoiled by the creators (Hagen et al.,2022).

LANGUAGE VARIETYThe corpus consists exclusively of English language content.

SPEAKER DEMOGRAPHICSThe demo graphic information of the original authors of the

clickbait posts is not available, as they were col lected from public social media platforms and news websites.

ANNOTATOR DEMOGRAPHICSThe anno tation was primarily performed by one main an notator with verifcation by two additional experts among the co-authors of the original paper (Hagen et al., 2022). No specifc demographic information about the annotators is provided.

SPEECH SITUATIONThe clickbait posts rep resent public social media communication aimed at attracting clicks.

TEXT CHARACTERISTICSThe corpus in cludes clickbait posts that typically employ linguis tic techniques designed to create curiosity gaps,such as sensationalism, teasers, and cataphors. The linked documents are primarily news articles, prod uct reviews, and blog posts from various domains.

RECORDING QUALITYN/A.

OTHERN/A.

PROVENANCE APPENDIXN/A.

## 4.4Sample Data Points

To better illustrate the nature of the dataset, Table 1presents examples of clickbait posts with their asso ciated spoilers across different spoiler types, drawn from the training set.

## 5Experiments

While Section 3 presented the formal approach and high-level modeling strategies, this section elabo rates on the specifc architectural details, implemen tation choices, and experimental setup to ensure reproducibility.

### 5.1Logistic Regression

Our baseline logistic regression model employs multiple feature extraction approaches:

• Feature Variants: We implemented and tested three TF-IDF feature variants:

– Standard: Features extracted from the combined title and paragraph text with a maximum of 10,000 features

– Separate: Title and paragraph texts pro cessed independently, then combined via horizontal stacking

– N-gram: Utilizing unigrams, bigrams,and trigrams (n-gram range 1-3)

• Word Embedding: Beyond TF-IDF, we also implemented weighted document embeddings where title and article text vectors are com bined with weights of 0.7 and 0.3 respectively.

• Hyperparameter Confguration: The model uses balanced class weights, L2 regularization with C=0.1, and the LBFGS solver with 1000maximum iterations.

This array of feature engineering approaches al lows us to identify the most effective text represen tation for the clickbait-spoiling task while maintain ing the interpretability advantages of the logistic regression framework.

### 5.2BiLSTM with Attention

The BiLSTM model architecture includes several technical optimizations:

• Word Embedding: We initially observed se vere overftting.Incorporating pre-trained Google News word embeddings reduced this issue.

• LSTM Structure: A single-layer bidirectional LSTM with confgurable hidden dimensions (default: 124 units) and attention mechanism to focus on the most informative words.

• TrainingProcess:Themodelem ploysAdamWoptimizer \text{(lr=3e-4,}weight_dec\text{ay}=8\mathrm{e}-4), gradientclipping at 0.5, and a learning rate scheduler that reduces the rate by 50% after 3 epochs without improvement.

#### 5.2.1ModernBERT

Given the limited performance of previous ap proaches, we experimented with ModernBERT models across various hyperparameter confgura tions.

• Model Variants: We experiment with both ModernBERT-base and ModernBERT-large variants from the answerdotai repository.

• Tokenization Strategy: Inputs are structured with the title repeated three times, followed by a [SEP] token and article content, then tokenized and padded to 128 tokens.


| Clickbait Post  | Spoiler  | Type  |
| -- | -- | -- |
| "NASA sets date for full recovery of ozone hole"  | "2070"  | Phrase  |
| "Just how safe are NYC’s water foun tains?" | "The Post independently tested eight water fountains in New York City’smost frequented parks, and found that all met or exceeded the state’s guide lines for water quality."  | Passage  |
| "A Harvard nutritionist and brain ex pert says she avoids these 5 foods that weaken memory and focus."  | "1. Added sugar", "2. Fried foods", "3. High-glycemic-load carbohy- drates", "4. Alcohol", "5. Nitrates"  | Multipart  |


Table 1: Examples from the Webis Clickbait Spoiling Corpus 2022

• Training Confguration: Fine-tuning uses a batch size of 16, learning rate of 8e-5 with linear warmup (ratio 0.1), and weight decay of 0.01.

• Optimization Process: The model uses early stopping with a patience of 3 evaluations, eval uating performance every 100 steps, and sav ing checkpoints with a maximum of 2 saved models.

• Resource Management: For environments with limited GPU memory, we implement op tional mixed precision training (fp16) and gra dient accumulation.

### 5.3Evaluation Protocol

Building on the evaluation framework outlined in Section 3.3, our experimental procedure includes:

• MetricsImplementation:Weuse scikit-learn’simplementationsofbal anced_accuracy_scoreandclassifca tion_report for consistent evaluation across models.

• Result Saving: All evaluation results are saved as JSON fles with timestamps for trace ability.

• Model Versioning: Best-performing models are saved with their corresponding tokeniz ers, vocabulary mappings, and confguration parameters to enable direct reuse.

This experimental setup enables systematic com parison of the three modeling approaches while providing suffcient detail for future replication of our results.

## 6Results

This section presents the experimental results of our clickbait spoiling models across three approaches:TF-IDF with logistic regression, BiLSTM with attention, and ModernBERT. We analyze perfor mance through balanced accuracy, precision, recall,and F1 scores, with attention to the three spoiler classes.

### 6.1Logistic Regression

Our baseline logistic regression with TF-IDF fea tures showed moderate performance with signs of overftting. The basic TF-IDF with 10,000 features achieved a training balanced accuracy of 0.6901 but only 0.5209 on validation data. Reducing the fea ture space to 1,500 yielded similar results (0.7078training, 0.5223 validation). The most effective confguration was the separate TF-IDF approach,which treated titles and paragraphs as distinct fea ture sets, achieving a validation balanced accuracy of 0.6018. This suggests that preserving the distinct linguistic patterns between clickbait headlines and spoiler text is important for effective classifcation.

### 6.2BiLSTM with Attention

In our BiLSTM experiments, we initially ob served severe overftting. Incorporating pre-trained Google News word embeddings reduced this is sue but still didn’t yield competitive performance.The fnal confguration (vocabulary size: 20,000,sequence length: 100, embedding dimension: 300,hidden dimension: 128, dropout: 0.65) achieved a balanced accuracy of 0.5438 and a weighted F1score of 0.5499. The class-specifc F1 scores were 0.6037 for phrase spoilers, 0.5240 for passage spoil ers, and 0.4821 for multi-element spoilers.

Despite its theoretical capacity to capture long-

range text dependencies, the BiLSTM model un derperformed compared to the baseline logistic re gression model, suggesting it struggles to model the complex relationships needed for distinguishing be tween spoiler types, particularly for multi-element spoilers.

### 6.3ModernBERT Models

Given the limited performance of previous ap proaches, we experimented with transformer-based ModernBERT models across various hyperparame ter confgurations.

To enable systematic analysis, we curated a struc tured results dataset from our evaluations, selecting representative confgurations that allowed for mean ingful parameter comparisons. All visualizations and analyses presented derive from this organized collection.

#### 6.3.1Overall Performance Distribution

Figure 2 shows the distribution of performance met rics across different spoiler classes for our Mod ernBERT experiments. The distributions show that ModernBERT models signifcantly outperformed our previous approaches, with balanced accuracy values centered around a median of 0.65 and ex tending to a maximum of 0.71. This represents sub stantial improvement over the best TF-IDF model (0.60) and BiLSTM model (0.54). Similarly, the F1 score distribution demonstrates consistent per formance advantages, with a median of 0.67 and best results reaching 0.73, compared to 0.61 and 0.55 for TF-IDF and BiLSTM respectively. The distribution of metrics indicates that ModernBERT models provide robust performance even across varying hyperparameter settings.

#### 6.3.2Learning Rate Analysis

Our learning rate optimization followed a two phase approach: frst testing a logarithmic range (5e-04, 5e-05, 5e-06) to identify promising mag nitudes, then conducting fne-grained experiments from 1e-05 to 9e-05.

Table 2 presents ModernBERT-base perfor mance across learning rates while keeping other parameters constant (sequence length=128, batch \text{size}=16, epochs=3).

Performance improved steadily from 1e-05 to 8e-05, then declined at 9e-05, suggesting optimal gradient updates need to be suffciently large to escape suboptimal regions but not so large as to cause training instability.


| Learning Rate  | Balanced Accuracy  |
| -- | -- |
| 1e-05  | 0.5469  |
| 3e-05  | 0.6542  |
| 5e-05  | 0.6800  |
| 8e-05  | 0.6940  |
| 9e-05  | 0.6646  |


Table 2: Learning Rates Comparison

#### 6.3.3Sequence Length Optimization

Figure 3 demonstrates that sequence length is a critical parameter. A maximum length of 128 to kens provided optimal balance between suffcient context and avoiding noise.Longer sequences (256 tokens) typically led to performance degra dation, while shorter sequences (64 tokens) failed to capture enough contextual information, suggest ing clickbait spoiling requires moderate context but suffers from information dilution with excessively long sequences.

#### 6.3.4Training Duration Effects

As shown in Figure 4, three epochs provided opti mal performance for most confgurations. Beyond this point, validation loss increased while training loss continued to decrease, indicating overftting.This relatively short optimal training duration sug gests the models quickly learn relevant patterns,and additional training primarily reinforces dataset specifc biases rather than improving generaliza tion.

#### 6.3.5Impact of Model Size

As Figure 5 illustrates, ModernBERT-large consis tently outperformed its base counterpart. With the optimal learning rate of 8e-05, the large model achieved a balanced accuracy of 0.7077 versus 0.6940 for the base model. The performance ad vantage was most pronounced for multi-element spoilers, with the large model achieving an F1 score of 0.6642 compared to 0.6520 for the base model,suggesting that additional parameters help capture the nuanced linguistic patterns required for com plex spoiler classifcation.

### 6.4Class-specifc Performance

Figure 6 shows consistent patterns in class-specifc performance. Phrase spoilers (class 0) were eas iest to predict, with the highest F1 scores across all models. Passage spoilers (class 1) showed mod erate performance, while multi-element spoilers


![](https://web-api.textin.com/ocr_image/external/367f665350492258.jpg)


![](https://web-api.textin.com/ocr_image/external/ff9975a7ffca81ee.jpg)


![](https://web-api.textin.com/ocr_image/external/121c7cce40c2a3bd.jpg)


![](https://web-api.textin.com/ocr_image/external/45c130a3a9d3311c.jpg)


![](https://web-api.textin.com/ocr_image/external/2c9bc11ff3a858c3.jpg)

Figure 2: Performance metrics for different ModernBERT confgurations


![](https://web-api.textin.com/ocr_image/external/ee20df7c2d196d42.jpg)

Figure 3: Impact of maximum sequence length on model performance

(class 2) proved most challenging. ModernBERT large substantially narrowed these performance gaps compared to simpler approaches.

### 6.5Error Analysis

DRAFT Our error analysis revealed several linguis tic patterns associated with misclassifcations:

Complex multi-element spoilers: The models struggled most with spoilers requiring understand ing of multiple components and their relationships.For example, spoilers containing both numerical data and named entities were often misclassifed as passage spoilers.

Ambiguous phrase boundaries: When phrase spoilers contained multiple clauses or dependent phrases, models sometimes misclassifed them as passage spoilers, suggesting diffculty in identify ing precise spoiler boundaries.

Implicit information: Spoilers relying on infer ence or implicit knowledge were frequently mis classifed. For instance, headlines asking "Who won?" were challenging when the spoiler contained contextual information beyond just the winner’s name.


![](https://web-api.textin.com/ocr_image/external/848aa9ca88abf80e.jpg)

Figure 4: Performance across different training dura tions


![](https://web-api.textin.com/ocr_image/external/b1c9e876a1ca0c6c.jpg)

Figure5:Performancecomparisonbetween ModernBERT-large and ModernBERT-base

Domain-specifc terminology:Technical or domain-specifc terminology in spoilers (e.g.,sports statistics, medical terms) led to higher error rates, indicating the models’ sensitivity to vocabu lary distribution.

Semantic overlap between classes:Cases where spoilers contained characteristics of mul tiple classes (e.g., a brief passage with numerical elements) showed higher error rates, refecting the inherent ambiguity in spoiler categorization.

These linguistic challenges were progressively better handled by more sophisticated models, with


![](https://web-api.textin.com/ocr_image/external/05edc0f393970d27.jpg)

Figure 6: Class-specifc F1-scores for ModernBERT models

ModernBERT-large showing the most robust per formance across these diffcult cases.

### 6.6Model Comparison Summary

The results in Table 3 demonstrate that transformer based models signifcantly outperform tradi tional approaches for clickbait spoiling, with ModernBERT-large providing the most robust per formance across all metrics and spoiler types. The substantial improvement in class 2 performance is particularly noteworthy, as this represents the most challenging category of spoilers.



| Model  | Balanced Accuracy  | F1-Score Phrase  | F1-Score Passage  | F1-Score Multi  |
| -- | -- | -- | -- | -- |
| Logistic Regression  | 0.6018  | 0.6321  | 0.6175  | 0.5458  |
| BiLSTM  | 0.5438  | 0.6037  | 0.5240  | 0.4821  |
| ModernBERT-base  | 0.6940  | 0.7141  | 0.7294  | 0.6520  |
| ModernBERT-large (best)  | 0.7077  | 0.7413  | 0.7418  | 0.6642  |


Table 3: Results Comparison for different Models


