# AI624 Assignment 4: Results

**Student:**  
**Student ID:**  

## T1.1: Independent Training and Logit KD

**Q1.** Did KD improve accuracy or convergence? Compare the training curves and final results. Did KD change the student's inference cost?

| Model | Top-1 (%) | Top-5 (%) | Parameters | Latency (ms) |
|---|---:|---:|---:|---:|
| EfficientNet-B7 teacher | | | | |
| B0: Cross-entropy | | | | |
| B0: Logit KD | | | | |

**Answer:**


## T1.2: Feature Matching Without an Adapter

**Q1.** Report the teacher and student feature dimensions, hint-loss curve, and validation accuracy. Why can these features be compared directly? Does having the same shape mean they contain the same information?

**Answer:**


## T1.3: Feature Matching With an Adapter

**Q1.** Show the feature shapes before and after the adapter, its parameter count, hint loss, and validation accuracy. Why is the adapter needed in this case?

**Answer:**


## T1.4: Final Comparison

**Q1.** Compare independent training, logit KD, direct feature matching, and adapted feature matching. Which configuration performed best? What additional training cost did feature matching introduce?

| Configuration | Top-1 (%) | Top-5 (%) | Training time | Deployed parameters | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Independent | | | | | |
| Logit KD | | | | | |
| Direct hint | | | | | |
| Adapted hint | | | | | |

**Answer:**


## T2.1: Generate Held-Out Questions

**Q1.** Provide the generated dataset and teacher prompt. Describe how you checked the answers. Include examples of any errors encountered. How do your generated problems compare with GSM8K?

**Answer:**


## T2.2: Fine-Tune on Teacher Rationales

**Q1.** Report the number of training examples, trainable parameter count and percentage, training loss curve, and total training time. Briefly explain how the student learns from teacher-generated reasoning traces.

| Training examples | Trainable parameters | Trainable (%) | Training time |
|---:|---:|---:|---:|
| | | | |

**Answer:**


## T2.3: Evaluate Reasoning

**Q1.** Compare the accuracy of the original and fine-tuned students on the real and synthetic questions. Show examples of correct and incorrect solutions. Did the model ever reach the correct answer through incorrect reasoning? How well did distillation transfer to the synthetic set?

| Model | GSM8K accuracy (%) | Synthetic accuracy (%) |
|---|---:|---:|
| Original student | | |
| Fine-tuned student | | |

**Answer:**


## T3.1: Differentiable Cell

**Q1.** Show the cell graph, architecture parameter dimensions, and verification results. Explain why the architecture parameters can be optimized using gradients.

**Answer:**


## T3.2: First-Order Search

**Q1.** Show parameter-update verification, training curves, and probability histories for at least two edges. Which operations became more likely during search?

**Answer:**


## T3.3: Extract and Evaluate a Discrete Architecture

**Q1.** Compare the continuous supernet, inherited-weight model, and independently trained model. How much did performance change after discretization and independent training?

| Model | Validation accuracy (%) | Parameters | MACs | Model size | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Continuous supernet | | | | | |
| Discrete: inherited weights | | | | | |
| Discrete: trained from scratch | | | | | |

**Answer:**


## T4.1: Supernet and Path Sampling

**Q1.** Show two example paths, their active operations, parameter mappings, and output shapes. How does weight sharing reduce the cost of searching a large number of architectures?

**Answer:**


## T4.2: Train the Supernet

**Q1.** Show training curves, training time, parameter-update checks, and sampling counts. What are the advantages and limitations of evaluating architectures using inherited weights?

**Answer:**


## T4.3: Subnetwork Adaptation

**Q1.** Show the accuracy table, plots, and Kendall rank correlations. Which subnetworks changed rank? Was the best inherited-weight architecture still the best after adaptation? How much additional training did adaptation require?

| Path | 0 steps accuracy (%) | 10 steps accuracy (%) | 30 steps accuracy (%) |
|---|---:|---:|---:|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |
| 12 | | | |

**Answer:**


## T4.4: Compare With Training From Scratch

**Q1.** Compare inherited, adapted, and independently trained validation accuracy for both architectures. Did adaptation provide a better indication of the final architecture ranking than inherited weights alone?

| Path | Inherited accuracy (%) | Adapted accuracy (%) | From-scratch accuracy (%) |
|---|---:|---:|---:|
| Best adapted | | | |
| Random comparison | | | |

**Answer:**


## T5.1: Random Search

**Q1.** Report the search-space size, one example architecture, the best result for each seed, and best-so-far accuracy curves.

**Answer:**


## T5.2: Aging Evolution

**Q1.** Show pseudocode or an execution trace, population-size checks, and curves for current population best and best-ever accuracy. Can the best architecture currently in the population become worse over time? Explain why.

**Answer:**


## T5.3: Compare the Search Methods

**Q1.** Compare random search and aging evolution using best-so-far validation accuracy against the number of evaluations. Report the best, mean, and standard deviation across three runs. Which method performed better under the same evaluation budget?

| Method | Best validation accuracy (%) | Mean (%) | Std. dev. (%) | Evaluations |
|---|---:|---:|---:|---:|
| Random search | | | | 100 |
| Aging evolution | | | | 100 |

**Answer:**


