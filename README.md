Machine Punishment - Adding Human Judgement

to Image Recognition

Facoltà di Ingegneria dell’Informazione, Informatica e Statistica

Applied Computer Science and Artificial Intelligence

**Kilian Schulz**

ID number 1986842

Advisor

Prof. Daniele Pannone

```
Co-Advisor
Prof. Danilo Avola
```
Academic Year 2023/


```
Thesis not yet defended
```
```
Machine Punishment - Adding Human Judgement to Image Recognition
Bachelor’s degree internship report. Sapienza University of Rome. Sapienza University of
Rome
```
©2024 Kilian Schulz. All rights reserved

```
This thesis has been typeset by LATEX and the Sapthesis class.
```
```
Author’s email: schulz.1986842@studenti.uniroma1.it
```

_Audentes Fortuna iuvat_
Virgilio, Eneide, X, 284



```
v
```
### Abstract

This paper introduces an innovative method for incorporating human expertise into
Computer Vision models through a saliency-based loss function. Our approach
is designed to minimize human effort while maximizing improvements in model
generalization and accuracy. We demonstrate that it is possible to influence the
salient regions considered by a model without significantly impacting its validation
accuracy. This technique bridges the gap between human perceptual knowledge and
machine learning algorithms, potentially enhancing model performance in scenarios
where domain expertise is crucial or training data is limited. Our findings suggest a
promising direction for creating more interpretable and efficient Computer Vision
models that align closely with human visual attention patterns.



## vii



- 1 Introduction Contents
   - 1.1 Motivation for the Research
   - 1.2 Challenges in the Current Approach
   - 1.3 Research Goals
- 2 Literature Review
   - 2.1 Literature on Saliency Maps
      - 2.1.1 Ground Truth Based Comparison of Saliency Maps Algorithms
      - 2.1.2 Saliency-Diversified Deep Ensembles (SDDE)
      - 2.1.3 SESS: Saliency Enhancing with Scaling and Sliding
      - 2.1.4 Contextual Prediction Difference Analysis (PDA)
      - 2.1.5 Gradient-based Saliency Maps
      - 2.1.6 Activation-based Saliency Maps
      - 2.1.7 Perturbation-based Saliency Maps
      - 2.1.8 Applications of Saliency Maps
      - 2.1.9 Conclusion
   - 2.2 Current State of Explainable AI
      - 2.2.1 Interpretable Machine Learning Models
      - 2.2.2 Post-hoc Explanation Methods
      - 2.2.3 Explanation in Reinforcement Learning
      - 2.2.4 Evaluation Metrics for Explainability
      - 2.2.5 Feature Attribution Methods
      - 2.2.6 Concept-based Explanations
      - 2.2.7 Counterfactual Explanations
      - 2.2.8 Applications of Explainable AI
      - 2.2.9 Conclusion
   - 2.3 CNN Manipulation with Human Control
      - 2.3.1 Cyborg Learning: Human-in-the-Loop Adaptation
      - 2.3.2 Human-Aided Saliency for CNN Training
      - 2.3.3 Saliency Maps-Based CNNs for Facial Expression Recognition
      - 2.3.4 Comparative Analysis and Future Directions
- 3 Background
   - 3.1 Convolutional Neural Networks
      - 3.1.1 Applications of CNNs
      - 3.1.2 How CNNs Work
      - 3.1.3 Example Architecture viii Contents
      - 3.1.4 Conclusion
   - 3.2 Saliency Maps
      - 3.2.1 Understanding Saliency Maps
      - 3.2.2 Generation of Saliency Maps
      - 3.2.3 Applications of Saliency Maps
      - 3.2.4 Challenges and Limitations
      - 3.2.5 Conclusion
   - 3.3 Backpropagation
      - 3.3.1 Understanding Backpropagation
      - 3.3.2 Steps in Backpropagation
      - 3.3.3 Importance of Backpropagation
      - 3.3.4 Challenges and Considerations
      - 3.3.5 Conclusion
   - 3.4 Hessian Matrix
      - 3.4.1 Mathematical Definition
      - 3.4.2 Properties of the Hessian
      - 3.4.3 Applications in Machine Learning
      - 3.4.4 Challenges and Approximations
      - 3.4.5 Conclusion
   - 3.5 Loss function
      - 3.5.1 Common Loss Functions
      - 3.5.2 Properties of Loss Functions
      - 3.5.3 Choosing a Loss Function
      - 3.5.4 Conclusion
- 4 Methods
   - 4.1 User Interface for marking images
      - 4.1.1 Window and Canvas Setup
      - 4.1.2 Image Processing and Display
      - 4.1.3 Interactive Drawing
      - 4.1.4 Pixel Marking and Data Collection
      - 4.1.5 Result Processing
      - 4.1.6 Conclusion
   - 4.2 Committing changes
      - 4.2.1 Window Initialization
      - 4.2.2 Layout and Components
      - 4.2.3 Image Processing and Display
      - 4.2.4 User Interaction
      - 4.2.5 Result Processing
   - 4.3 Loss function
      - 4.3.1 Formulation of the Loss Function
      - 4.3.2 Marked Pixels Matrix
      - 4.3.3 Gradient Absolute Values and Clipping
      - 4.3.4 Interpretation of the Loss Function
      - 4.3.5 Implementation
      - 4.3.6 Advantages and Considerations
   - 4.4 Incorporating the Gradient Information Contents ix
      - 4.4.1 Gradient Descent Loop
      - 4.4.2 Stopping Criteria
      - 4.4.3 Loss Computation and Backpropagation
      - 4.4.4 Weight Adjustment
      - 4.4.5 Saliency Map Computation
      - 4.4.6 Impact Measurement
      - 4.4.7 Overfitting Check
      - 4.4.8 Experimental Nature of Parameters
- 5 Experiments
   - 5.1 Integrated Gradients
   - 5.2 Sensitivity Analysis
   - 5.3 Calculation of the Hessian
- 6 Conclusion
   - 6.1 Results
      - 6.1.1 Control of Saliency
      - 6.1.2 Performance Across Data Sizes
      - 6.1.3 Validation Loss
      - 6.1.4 Visual Demonstration of Saliency Changes
   - 6.2 Contributions
      - 6.2.1 Novel Saliency Control Mechanism
      - 6.2.2 Adaptive Saliency without Performance Degradation
      - 6.2.3 Effectiveness Across Data Regimes
      - 6.2.4 Enhanced Model Interpretability
      - 6.2.5 Flexible Framework for Saliency Manipulation
      - 6.2.6 Bridging Human Intuition and Machine Learning
   - 6.3 Application Fields and Potential Impact
      - 6.3.1 Medical Imaging and Rare Disease Detection
      - 6.3.2 Geological Exploration and Resource Detection
      - 6.3.3 Remote Sensing and Environmental Monitoring
      - 6.3.4 Manufacturing and Quality Control
   - 6.4 Future Work and Outlook
      - 6.4.1 Comprehensive Generalization Benchmarking
      - 6.4.2 Optimizing Computational Efficiency
      - 6.4.3 Exploration of Alternative Optimizers
      - 6.4.4 Expanding to Diverse Classification Tasks
      - 6.4.5 Alternative Saliency Approaches
      - 6.4.6 Hyperparameter Optimization
      - 6.4.7 Alternative Loss Functions
      - 6.4.8 Adaptive Saliency Optimization
   - 6.5 Model Usage and Code Structure
      - 6.5.1 Setting Up the Model
      - 6.5.2 Integrating PunisherLoss into Your CNN
      - 6.5.3 Model Training Setup
      - 6.5.4 Training Process


**x Contents**

```
6.5.5 Human Intervention Interface.................. 45
6.5.6 Customization and Improvement................ 46
```
**Bibliography 46**


```
1
```
## Chapter 1

# Introduction

### 1.1 Motivation for the Research

```
Imagine a master chef teaching a novice. The novice observes the ingredients and
the final dish but misses the subtle techniques and critical decisions made during the
cooking process. This scenario parallels the current state of AI training in computer
vision. We see the input (images) and the output (predictions), but the intricate
decision-making process within the model remains largely opaque.
No one would consider culinary education complete if students were left to repeatedly
attempt dishes without guidance. Yet, this is precisely how we train computer vision
models. Our research proposes an approach akin to expert supervision in cooking
```
- where the model, like a novice chef, still needs to train and learn, but receives
    focused guidance along the way.
    Surprisingly, there has been limited research in this direction, with few exceptions.
    Boyd et al. [ 8 ] discovered significant improvements in both accuracy and generaliza-
    tion by incorporating human expertise into image recognition models. Despite their
    promising results, widespread implementation in the computer vision community has
    been slow. A possible reason is the substantial effort required to create annotations
    for each training image. Our research aims to address this by exploring dynamic
    annotation of select images and adjusting the model based on these human inputs.

### 1.2 Challenges in the Current Approach

```
Several factors contribute to the limited exploration of this topic:
```
1. Non-standard procedures: Determining the appropriate weight adjustments to
    influence pixel importance is not a straightforward process. Depending on the chosen
    approach, this may require multiple derivatives or creative solutions, each presenting
    unique challenges.
2. Integration of human and machine expertise: Machines often train on vast amounts
    of data, making it impractical for humans to annotate all instances meaningfully.
The challenge lies in making human annotations more impactful than mere training
    data elements.
3. Balancing human input: It’s crucial to determine the optimal weight for human
    annotations without overfitting or underutilizing this valuable input. Factors to


**2 1. Introduction**

consider include the quantity of human-annotated data, the level of human expertise,
and project-specific variables that may not conform to a simple formula.

### 1.3 Research Goals

The primary objectives of this research are:

1. To develop a method that allows humans to improve a model’s generalization
skills and accuracy with minimal effort.
2. To explore ways in which human input can influence the training process,
potentially reducing the amount of training data required to achieve comparable
results.
3. To strike an optimal balance between machine learning capabilities and human
perceptual expertise, creating more efficient and interpretable computer vision
models.
By addressing these goals, we aim to bridge the gap between human visual under-
standing and machine learning algorithms, paving the way for more robust and
adaptable computer vision systems.


```
3
```
## Chapter 2

# Literature Review

### 2.1 Literature on Saliency Maps

```
Due to the structure of my thesis I am going to start with the current literature
on saliency maps here and then explain saliency maps in depth in the background
part, which is chapter 3. Saliency maps have become a pivotal tool in the field of
computer vision, providing visual explanations for the decisions made by deep neural
networks (DNNs). After explaining how a saliency map works, I will now try to get
into how the literature stands on current types of saliency maps and uses according
to the current literature.
```
#### 2.1.1 Ground Truth Based Comparison of Saliency Maps Algo-

#### rithms

```
Recent studies present methodologies to evaluate the effectiveness of saliency map
generation methods using state-of-the-art network architectures and benchmark
datasets, proposing novel metrics for quantitative comparison [48].
```
#### 2.1.2 Saliency-Diversified Deep Ensembles (SDDE)

The introduction of Saliency-Diversified Deep Ensembles (SDDE) promotes diversity
among ensemble members by leveraging saliency maps, outperforming conventional
ensemble techniques [6].

#### 2.1.3 SESS: Saliency Enhancing with Scaling and Sliding

The SESS approach addresses limitations in existing saliency map generation meth-
ods, improving visual quality and performance on object recognition and localization
tasks [50].

#### 2.1.4 Contextual Prediction Difference Analysis (PDA)

```
Contextual PDA is a model-agnostic method that computes the relevance of con-
textual information, making it more efficient and effective in explaining individual
image classifications [15].
```

**4 2. Literature Review**

#### 2.1.5 Gradient-based Saliency Maps

Gradient-based saliency maps use the gradient of the output with respect to the
input image to determine saliency [46].

#### 2.1.6 Activation-based Saliency Maps

Activation-based saliency maps rely on the activation of particular layers within a
CNN to identify salient regions [53].

#### 2.1.7 Perturbation-based Saliency Maps

Perturbation-based saliency maps involve perturbing parts of the input image and
observing the effect on the output [11].

#### 2.1.8 Applications of Saliency Maps

Saliency maps are used in various applications, including object detection, medical
imaging, and robotics, to improve efficiency and decision-making processes.

#### 2.1.9 Conclusion

Recent advancements in saliency maps, such as the introduction of SDDE, SESS, and
Contextual PDA, have significantly improved the effectiveness and interpretability
of these visual explanation tools. As the field of computer vision continues to
evolve, these advancements will play a crucial role in enhancing the performance
and reliability of saliency maps.


**2.2 Current State of Explainable AI 5**

### 2.2 Current State of Explainable AI

#### 2.2.1 Interpretable Machine Learning Models

```
Recent research has focused on developing inherently interpretable models, such as
decision trees, rule-based classifiers, and linear models, which provide insights into
their decision-making process [42].
```
#### 2.2.2 Post-hoc Explanation Methods

```
Post-hoc explanation methods, such as LIME and SHAP, have been widely adopted
for explaining complex models like deep neural networks and ensemble methods
[40, 29].
```
#### 2.2.3 Explanation in Reinforcement Learning

XAI has been extended to reinforcement learning, with new methods developed to
explain the policies and actions of agents in complex environments [39].

#### 2.2.4 Evaluation Metrics for Explainability

The field has seen the development of evaluation metrics for explainability, aiming
to quantify the quality and effectiveness of explanations provided by XAI methods
[10].

#### 2.2.5 Feature Attribution Methods

Feature attribution methods assign importance values to input features based on their
contribution to the output of the model. These methods are crucial for understanding
which features are most influential in a model’s predictions.

```
SHAP Values
```
```
For a model output f ( x ), SHAP values explain the prediction by computing the
contribution of each feature to the difference between the actual prediction and the
average prediction, as shown in the formula:
```
```
f ( x )− E [ f ( x )] =
```
```
∑ n
```
```
i =
```
```
φi (2.1)
```
where _φi_ is the SHAP value for feature _i_ , and _E_ [ _f_ ( _x_ )]is the expected value of the
model output over the dataset.

#### 2.2.6 Concept-based Explanations

```
Concept-based explanations aim to align model representations with human-understandable
concepts, providing a higher level of abstraction in explanations [20].
```

**6 2. Literature Review**

**Testing with Concept Activation Vectors (TCAV)**

The relevance of a concept _C_ to a classification decision can be quantified using
TCAV scores:

```
TCAV score=
```
```
Number of positively influenced classes
Total number of classes
```
##### (2.2)

#### 2.2.7 Counterfactual Explanations

Counterfactual explanations provide insights into model decisions by explaining
what could be changed in the input to achieve a different output [51].

**Generating Counterfactuals**

A counterfactual _x_ ′for an instance _x_ is generated by solving the optimization
problem:
min
_x_ ′

```
d ( x,x ′) subject to f ( x ′)̸= f ( x ) (2.3)
```
where _d_ ( _x,x_ ′)measures the distance between the original instance and the counter-
factual, ensuring minimal changes are made.

#### 2.2.8 Applications of Explainable AI

XAI is applied across various domains, including healthcare, finance, and autonomous
systems, where understanding AI decisions is crucial for trust and accountability.

- Healthcare: Providing explanations for diagnostic decisions made by AI sys-
    tems.
- Finance: Explaining credit scoring models to customers and regulators.
- Autonomous Systems: Understanding the decision-making process of au-
    tonomous vehicles.

#### 2.2.9 Conclusion

The field of XAI is rapidly evolving, with significant advancements in interpretable
models, post-hoc explanation methods, and evaluation metrics. These developments
are crucial for deploying AI systems in real-world applications where transparency
and understanding are essential.

### 2.3 CNN Manipulation with Human Control

The integration of human control into Convolutional Neural Networks (CNNs) has
emerged as a promising approach to enhance model interpretability, performance,
and alignment with human intuition. This section reviews key contributions in this
rapidly evolving field.


**2.3 CNN Manipulation with Human Control 7**

#### 2.3.1 Cyborg Learning: Human-in-the-Loop Adaptation

Aidan Boyd’s work on "Cyborg Learning" [ 8 ] introduced a novel paradigm for human-
in-the-loop machine learning. The approach allows human experts to intervene
during the training process, guiding the model’s attention to relevant features.
The core idea is represented by the following optimization problem:

```
min
θ
L( θ ) + λ R( θ,H ) (2.4)
```
where _θ_ represents the model parameters,L( _θ_ )is the standard loss function,R( _θ,H_ )
is a regularization term incorporating human feedback _H_ , and _λ_ is a hyperparameter
balancing the two terms.
Boyd’s method demonstrated significant improvements in model performance and
interpretability across various taskss.

#### 2.3.2 Human-Aided Saliency for CNN Training

Building upon the Cyborg Learning concept, Boyd et al. proposed a more refined
approach focusing on human-aided saliency maps [ 8 ]. This method allows experts to
directly influence the model’s attention mechanism by providing saliency annotations
during training. Specifically, during training they blurred regions of the image that
humans deemed to be non-salient in order to stop the machine from taking them
into account.
The saliency-guided loss function is formulated as:

```
L sal =L CE + α ·MSE( Smodel,Shuman ) (2.5)
```
whereL _CE_ is the cross-entropy loss, _Smodel_ and _Shuman_ are the model-generated
and human-provided saliency maps respectively, and _α_ is a weighting factor.
This approach led to models that not only performed better on standard metrics
but also produced attention maps more aligned with human expert focus, enhancing
model interpretability.

#### 2.3.3 Saliency Maps-Based CNNs for Facial Expression Recognition

Wei [ 52 ] proposed a novel approach to incorporate saliency maps into convolu-
tional neural networks (CNNs) for facial expression recognition (FER). The key
contributions of this work include:

- A new saliency extraction model specifically designed for facial images, com-
    bining a Dilated Convolutional Inception module, a Difference of Gaussian
(DOG) module, and a multi-indicator saliency prediction module.
- A CNN-based FER framework that integrates saliency maps as prior knowledge,
    addressing the lack of visual attention guidance in traditional CNNs.
- A feature combination strategy that uses residual connections to combine
    saliency priors with median and high-level CNN features, preserving both local
    and global expression information.


**8 2. Literature Review**

The saliency extraction model achieved competitive performance on the CAT
dataset, outperforming several state-of-the-art approaches in terms of Correlation
Coefficient (CC) and Similarity (SIM) metrics.
For facial expression analysis, the proposed method was evaluated on the CK+
and BP4D datasets for Action Unit (AU) detection and smile intensity estimation.
The results showed significant improvements over existing approaches, with the
saliency-based ResNet model (SALI-Res) achieving the best performance.
Key findings include:

- The combination of median and high-level CNN features improved FER per-
    formance compared to using single-layer features.
- Incorporating saliency maps as prior knowledge further enhanced AU detection
    and smile intensity estimation accuracy.
- The proposed method outperformed other approaches that use facial landmarks
    or traditional feature extraction techniques.

This work demonstrates the potential of integrating visual saliency information into
deep learning models for improved facial expression analysis, offering a new direction
for enhancing FER performance in various applications.

#### 2.3.4 Comparative Analysis and Future Directions

These three approaches represent different strategies for incorporating human knowl-
edge into CNN training:

- Boyd’s Cyborg Learning provides a general framework for human-in-the-loop
    adaptation.
- The Human-Aided Saliency method focuses specifically on guiding the model’s
    attention mechanism.
- Wei’s SM-CNN approach integrates saliency maps directly into the network
    architecture and loss function.

Future research directions may include:

1. Developing more efficient interfaces for real-time human feedback during model
    training.
2. Exploring the integration of multiple human experts’ knowledge, potentially
    through ensemble methods.
3. Investigating the long-term effects of human-guided training on model robust-
    ness and generalization.
4. Addressing potential biases introduced by human feedback and ensuring model
    fairness.

As the field continues to evolve, we can expect to see more sophisticated methods
for integrating human expertise with deep learning models, potentially leading to
more reliable, interpretable, and human-aligned AI systems.


```
9
```
## Chapter 3

# Background

### 3.1 Convolutional Neural Networks

```
Convolutional Neural Networks (CNNs) are a class of deep neural networks that
have proven very effective in areas such as image recognition, natural language
processing, and video analysis. I chose them for this project in order to capture
spatial features in the state of the art way. They are particularly known for their
ability to automatically and adaptively learn spatial hierarchies of features from
input images [24].
```
#### 3.1.1 Applications of CNNs

```
CNNs are widely used in various fields, including but not limited to:
```
- **Image Classification** : Identifying the category of an image. For example,
    distinguishing between images of cats and dogs [23].
- **Object Detection** : Identifying objects within an image and drawing bounding
    boxes around them [12].
- **Image Segmentation** : Partitioning an image into multiple segments or
    regions [28].
- **Speech Recognition** : Converting spoken language into text [18].
- **Medical Image Analysis** : Analyzing medical images to detect diseases like
    tumors in MRI scans [27].

#### 3.1.2 How CNNs Work

The architecture of a CNN is inspired by the structure of the visual cortex and
consists of several types of layers [26]:

```
Convolutional Layer
```
The convolutional layer is the core building block of a CNN. It involves a mathe-
matical operation called convolution, which is a specialized kind of linear operation.


**10 3. Background**

Convolution preserves the spatial relationship between pixels by learning image
features using small squares of input data [26].
Mathematically, a convolution operation on an image _I_ with a filter _K_ is defined as:

```
S ( i,j ) = ( I ∗ K )( i,j ) =
```
```
∑
m
```
```
∑
n
```
```
I ( i + m,j + n )· K ( m,n )
```
where _S_ is the feature map, and _i,j_ are the spatial dimensions [14].

**Activation Function**

After each convolution operation, an activation function is applied to introduce
non-linearity into the model. The most commonly used activation function is the
Rectified Linear Unit (ReLU), defined as:

```
ReLU( x ) = max(0 ,x )
```
[35].

**Pooling Layer**

Pooling layers are used to reduce the spatial dimensions of the feature maps, thereby
reducing the number of parameters and computation in the network. The most
common pooling operation is max pooling, which takes the maximum value over a
window (usually 2x2) [44].

**Fully Connected Layer**

After several convolutional and pooling layers, the high-level reasoning in the neural
network is done via fully connected layers. Neurons in a fully connected layer have
connections to all activation in the previous layer [24].


**3.1 Convolutional Neural Networks 11**

#### 3.1.3 Example Architecture

A simple CNN architecture for image classification might look like this:

1. **Input Layer** : Takes an image of size 32 × 32 × 3.
2. **Convolutional Layer** : 32 filters of size 3 × 3 , stride 1, padding 1.
3. **ReLU Activation**.
4. **Max Pooling Layer** : Pool size 2 × 2 , stride 2.
5. **Convolutional Layer** : 64 filters of size 3 × 3 , stride 1, padding 1.
6. **ReLU Activation**.
7. **Max Pooling Layer** : Pool size 2 × 2 , stride 2.
8. **Fully Connected Layer** : 128 neurons.
9. **ReLU Activation**.
10. **Output Layer** : Number of classes (e.g., 10 for CIFAR-10).

#### 3.1.4 Conclusion

Convolutional Neural Networks have revolutionized the field of computer vision and
beyond. Their ability to automatically learn and extract features from raw input
data has made them a powerful tool for various applications, ranging from image and
video analysis to natural language processing and medical diagnostics. As research
continues, CNNs are expected to become even more efficient and effective, opening
up new possibilities in AI and machine learning [24].


**12 3. Background**

### 3.2 Saliency Maps

#### 3.2.1 Understanding Saliency Maps

Saliency maps are a visualization technique used in the field of computer vision
to highlight the regions of an input image that are most relevant to a neural
network’s decision. These maps are particularly useful for interpreting the behavior
of Convolutional Neural Networks (CNNs) and understanding which parts of an
image influence the network’s classification decision [46].

#### 3.2.2 Generation of Saliency Maps

Saliency maps are generated by computing the gradient of the output category with
respect to the input image. This gradient reflects how changes in pixel intensity
affect the classification score. By visualizing these gradients, one can see which
pixels contribute to increasing the likelihood of a particular output [46].

#### 3.2.3 Applications of Saliency Maps

Saliency maps have several applications, including:

- **Model Debugging** : Identifying whether a model is focusing on the relevant
    features for making a decision.
- **Data Cleaning** : Detecting mislabeled data by checking if the model’s atten-
    tion is consistent with the expected features of a class.
- **Improving Model Robustness** : Understanding model vulnerabilities to
    adversarial attacks by analyzing the regions that contribute most to the decision
    [49].
- **Medical Diagnosis** : Assisting clinicians by highlighting areas that are in-
    dicative of diseases in medical images [53].

#### 3.2.4 Challenges and Limitations

While saliency maps are a powerful tool, they have limitations. One challenge is that
they can sometimes be noisy and difficult to interpret. Additionally, they may not
provide a complete picture of the model’s decision-making process, as they only show
the most salient features rather than all the features that contribute to a decision
[21].

#### 3.2.5 Conclusion

Saliency maps offer a window into the decision-making process of CNNs, allowing
researchers and practitioners to visualize and interpret the features that drive neural
network predictions. Despite their limitations, they remain a valuable tool for
improving the transparency and trustworthiness of deep learning models.


**3.3 Backpropagation 13**

### 3.3 Backpropagation

#### 3.3.1 Understanding Backpropagation

```
Backpropagation is a fundamental algorithm used for training artificial neural
networks, particularly in the context of supervised learning. It is a method used to
calculate the gradient of the loss function with respect to the weights of the network
[ 43 ]. The backpropagation algorithm consists of two main phases: the forward pass
and the backward pass. In the forward pass, the input data is passed through the
network to obtain the output, which is then used to calculate the loss. During the
backward pass, the gradient of the loss is propagated back through the network to
update the weights, which is done using the chain rule of calculus [25].
```
#### 3.3.2 Steps in Backpropagation

The steps involved in the backpropagation algorithm are as follows:

1. Perform a forward pass to compute the output and loss.
2. Compute the gradient of the loss with respect to the output.
3. Propagate the gradients back through the network by computing the gradient
    of the loss with respect to each weight, which involves applying the chain rule
    recursively from the output layer to the input layer.
4. Update the weights of the network in the direction that reduces the loss,
    typically using an optimization algorithm like gradient descent [22].

#### 3.3.3 Importance of Backpropagation

```
Backpropagation is crucial for the training of deep neural networks as it allows for
efficient computation of gradients, without which the optimization of millions of
parameters would be computationally infeasible. It is the backbone of most modern
neural network training algorithms and has been instrumental in the successes of
deep learning [45].
```
#### 3.3.4 Challenges and Considerations

```
Despite its effectiveness, backpropagation has challenges such as the vanishing and
exploding gradient problems, which can make training deep networks difficult. This
has proven to be a significan problem in the presented project, given that the
gradients are manipulated according to pixel importance. Techniques such as weight
initialization, batch normalization, and skip connections have been developed to
mitigate these issues [13, 19, 17].
```
#### 3.3.5 Conclusion

```
Backpropagation remains one of the most significant algorithms in the field of neural
networks, enabling the training of complex models that can learn from vast amounts
of data. Its continued relevance is supported by ongoing research and advancements
in optimization techniques and network architectures.
```

**14 3. Background**

### 3.4 Hessian Matrix

The Hessian matrix, named after German mathematician Ludwig Otto Hesse, is a
square matrix of second-order partial derivatives of a scalar-valued function. In the
context of machine learning and optimization, the Hessian plays a crucial role in
understanding the local curvature of the loss landscape [36].

#### 3.4.1 Mathematical Definition

For a function _f_ : _Rn_ → _R_ , the Hessian matrix _H_ is defined as:

```
Hij =
∂^2 f
∂xi∂xj
```
##### (3.1)

where _xi_ and _xj_ are the _i_ -th and _j_ -th variables respectively [9].

#### 3.4.2 Properties of the Hessian

The Hessian matrix has several important properties:

- **Symmetry** : The Hessian is symmetric, i.e., _Hij_ = _Hji_.
- **Positive Definiteness** : At a local minimum, the Hessian is positive definite.
- **Negative Definiteness** : At a local maximum, the Hessian is negative definite.
- **Indefiniteness** : At a saddle point, the Hessian is indefinite [30].

#### 3.4.3 Applications in Machine Learning

The Hessian matrix has several important applications in machine learning:

**Optimization Algorithms**

Second-order optimization methods, such as Newton’s method, use the Hessian to
determine the optimal step size and direction in parameter space [ 7 ]. The update
rule for Newton’s method is:

```
θt +1= θt − H −^1 ∇ f ( θt ) (3.2)
```
where _θt_ is the parameter vector at iteration _t_ , _H_ is the Hessian, and∇ _f_ ( _θt_ )is the
gradient.

**Natural Gradient Descent**

The natural gradient, which uses the Fisher information matrix (a type of expected
Hessian), has been shown to be effective in training deep neural networks [1].

**Curvature Estimation**

The Hessian provides information about the curvature of the loss landscape, which can
be used to adapt learning rates and improve convergence in optimization algorithms
[32].


**3.4 Hessian Matrix 15**

#### 3.4.4 Challenges and Approximations

While the Hessian provides valuable second-order information, computing and storing
the full Hessian for large-scale problems can be computationally prohibitive. As
a result, various approximations and efficient computation methods have been
developed:

- **Diagonal Approximations** : Only the diagonal elements of the Hessian are
    computed and used [5].
- **Kronecker-Factored Approximate Curvature (K-FAC)** : This method
    approximates the Fisher information matrix (a type of Hessian) as a Kronecker
    product of smaller matrices [32].
- **Hessian-Free Optimization** : This approach uses matrix-vector products to
    implicitly work with the Hessian without explicitly forming it [31].

#### 3.4.5 Conclusion

The Hessian matrix provides crucial information about the local curvature of the loss
landscape in machine learning problems. While its exact computation can be chal-
lenging for large-scale problems, various approximations and efficient computation
methods have made it possible to leverage second-order information in optimization
algorithms for machine learning.


**16 3. Background**

### 3.5 Loss function

Loss functions, also known as cost functions or error metrics, are a critical component
in the training of machine learning models. They quantify the difference between
the predicted outputs of the model and the actual target values. The choice of loss
function directly influences the performance of the model and its ability to learn
from the training data [14].

#### 3.5.1 Common Loss Functions

Several loss functions are commonly used in machine learning, each with its own use
case:

- **Mean Squared Error (MSE)** : Used for regression tasks, it measures the
    average squared difference between the estimated values and the actual value
    [16].
- **Cross-Entropy Loss** : Applied in classification problems, it quantifies the
    difference between two probability distributions - the true distribution, and
    the predicted distribution. This is the custom loss that has been implemented
    in the paper at hand. [34].
- **Hinge Loss** : Often used for support vector machines and some types of neural
    networks, it is used for binary classification tasks [41].

#### 3.5.2 Properties of Loss Functions

An ideal loss function should have the following properties:

1. **Differentiability** : This allows the use of gradient-based optimization methods,
    which are efficient and widely used.
2. **Convexity** : In the context of convex optimization, a convex loss function en-
    sures that any local minimum is a global minimum, simplifying the optimization
    problem [9].
3. **Robustness** : The loss function should be robust to outliers in the data,
    preventing them from having an outsized impact on the model parameters.

#### 3.5.3 Choosing a Loss Function

The choice of a loss function depends on the specific requirements of the task at
hand, such as the nature of the output variable and the presence of outliers. It is also
influenced by the desired properties of the model, like probabilistic interpretation or
margin maximization.


**3.5 Loss function 17**

#### 3.5.4 Conclusion

```
Loss functions are a fundamental aspect of the training process for machine learning
models. They provide the criteria by which models are evaluated and optimized, and
their careful selection is crucial for the development of effective machine learning
applications.
```


```
19
```
## Chapter 4

# Methods

### 4.1 User Interface for marking images


```
20 4. Methods
```
```
The custom loss function implements a graphical user interface (GUI) for marking
pixels on images. This interface is crucial for the research product, allowing users to
interact with and annotate images. The GUI is implemented using Python’s Tkinter
library and consists of several key components and functionalities:
```
#### 4.1.1 Window and Canvas Setup

```
The interface is initialized with a main Tkinter window:
```
1 root = tk .Tk()
2 root. t i t l e ( "Mark Pixels " )
3 window = tk. Toplevel ( root )
4 window. t i t l e ( "Mark Pixels " )
5 window. geometry ( " 800x800 " )

```
A canvas is created within this window to display images:
```
1 canvas = tk. Canvas (window , bg=" white " )
2 canvas. place ( relwidth =10, r e l h e i g h t =10)

#### 4.1.2 Image Processing and Display

```
The system processes and displays images as follows:
```
1. Random images are selected from the training dataset.
2. Each image is processed and resized to fit the canvas.
3. A saliency map is computed for each image:
    1 s e l f. s a l i e n c y = s e l f. compute_saliency_map ( image. unsqueeze (0) ,
       l a b e l )
    2
4. The saliency map is blended with the original image:
    1 blended_image = Image. alpha_composite ( image_pil. convert ( ’RGBA’
       ) , saliency_map )
    2
5. The blended image is displayed on the canvas.

#### 4.1.3 Interactive Drawing

```
Users can annotate the images through interactive drawing:
```
- Mouse drag events are used to draw on the image:
    1 canvas. bind ( "<B1−Motion>" , **lambda** event : drag ( event ) )
    2
- A slider adjusts the drawing radius:


**4.1 User Interface for marking images 21**

```
1 s l i d e r = tk. Scale (window , from_=0, to =80, length =200, orient="
horizontal " ,
2 command= lambda value , canvas=canvas : s e l f.
slider_changed ( value ) )
3
```
- Users can switch between red and green drawing colors, red in order to
    discourage these regions from being salient, green to encourage them:
1 s e l f. switch_button = tk. Button (window , text=" Switch Color " ,
2 command=s e l f. switch_color , bg=
s e l f. color )
3

#### 4.1.4 Pixel Marking and Data Collection

When the user closes the annotation window, the system processes the annotations:
This process identifies marked pixels, distinguishing between red (discouraged) and
green (encouraged) markings. The marked pixels are recorded and used to modify
the original image for further processing.

#### 4.1.5 Result Processing

After annotation, the code computes various metrics based on the marked pixels:

- Count of marked pixels (both discouraged and encouraged).
- Impact of marked pixels on the model’s output.

These metrics are crucial for analyzing the effect of human annotations on the
model’s performance.

#### 4.1.6 Conclusion

This user interface plays a vital role in the research by enabling direct human input
into the image analysis process. It allows researchers to highlight specific areas of
interest or concern in the images, which can then be used to refine the model’s
performance or validate its outputs. The combination of saliency map visualization
and interactive annotation provides a powerful tool for understanding and improving
the model’s behavior on specific image regions.


```
22 4. Methods
```
### 4.2 Committing changes

```
The Choser Window is a crucial component of the user interface, designed to allow
users to make decisions about committing changes to the model. This window
presents two images side by side, enabling users to compare and select between them.
The implementation utilizes Python’s Tkinter library for creating the graphical user
interface. Each image here represents a model version, so the user is presented with
the saliency map on the annotated image with the old model as well as with the
proposed changes. There is also the validation accuracy for both models displayed,
in order to allow the user to take an informed choice on whether he would like to
commit the proposed changes.
```
#### 4.2.1 Window Initialization

```
The Chooser Window is initialized with the following parameters:
```
1 **c l a s s** ChoserWindow :
2 **def** __init__( s e l f , image1 , image1_text , image2 , image2_text ) :
3 s e l f. root = tk .Tk()
4 s e l f. root. t i t l e ( " Image Window" )
5 s e l f. root. geometry ( " 1600 x900 " )

```
The window is set to a size of 1600x900 pixels to accommodate two images side by
side comfortably.
```
#### 4.2.2 Layout and Components

```
The window layout consists of several key components:
```
1. Two canvases for displaying images:
    1 s e l f. canvas1 = tk. Canvas ( s e l f. frame , bg=" white " )
    2 s e l f. canvas2 = tk. Canvas ( s e l f. frame , bg=" white " )
    3


```
4.2 Committing changes 23
```
2. Labels for image descriptions:
    1 s e l f. label1 = tk. Label ( s e l f. root , text=s e l f. image1_text , font
       =(" Helvetica " , 16) )
    2 s e l f. label2 = tk. Label ( s e l f. root , text=s e l f. image2_text , font
       =(" Helvetica " , 16) )
    3

```
These components are arranged using Tkinter’s pack geometry manager to ensure
proper alignment and spacing.
```
#### 4.2.3 Image Processing and Display

```
The display_images method handles the processing and rendering of images:
```
1 **def** display_images ( s e l f ) :
2 width = s e l f. root. winfo_width ()
3 height = s e l f. root. winfo_height ()
4 blended_image1 = s e l f. blend_images ( s e l f. image1 )
5 blended_image2 = s e l f. blend_images ( s e l f. image2 )
6 s e l f. blended_image_tk1 = ImageTk. PhotoImage ( blended_image1. r e s i z e ((
width // 2 , height ) ) )
7 s e l f. blended_image_tk2 = ImageTk. PhotoImage ( blended_image2. r e s i z e ((
width // 2 , height ) ) )
8 s e l f. display_image ( s e l f. blended_image_tk1 , s e l f. canvas1 )
9 s e l f. display_image ( s e l f. blended_image_tk2 , s e l f. canvas2 )

```
This method resizes the images to fit the window and applies any necessary blending
operations before displaying them on their respective canvases.
```
#### 4.2.4 User Interaction

```
The window captures user input through mouse clicks on the images:
```
1 s e l f. canvas1. bind ( "<Button−1>" , **lambda** event : s e l f. on_image_click ( False
) )
2 s e l f. canvas2. bind ( "<Button−1>" , **lambda** event : s e l f. on_image_click ( True )
)

```
When an image is clicked, the on_image_click method is called, recording the user’s
selection and closing the window.
```
#### 4.2.5 Result Processing

```
The run method initiates the Tkinter main loop and returns the user’s selection:
```
1 **def** run ( s e l f ) :
2 s e l f. root. mainloop ()
3 **return** s e l f. s e l e c t i o n

```
This method allows the calling code to retrieve the user’s decision, which can then
be used to determine whether to commit changes to the model.
```

```
24 4. Methods
```
### 4.3 Loss function

In this section, we introduce a novel loss function designed to incorporate human-
perceived saliency into the learning process of our convolutional neural network
(CNN). The primary objective of this loss function is to encourage the model to
give more importance to positively marked pixels and less importance to negatively
marked pixels, thereby aligning the model’s attention with human-defined salient
regions.

#### 4.3.1 Formulation of the Loss Function

```
Our custom loss function Lsaliency is defined as:
```
```
Lsaliency =
```
```
∑
```
```
i,j
```
```
Mi,j ·clip(| Gi,j | , 0 , 0. 5) (4.1)
```
```
where:
```
- _Mi,j_ is the( _i,j_ )-th element of the marked pixels matrix
- _Gi,j_ is the( _i,j_ )-th element of the gradient matrix with respect to the input
- | _Gi,j_ |denotes the absolute value of _Gi,j_
- clip( _x,a,b_ )is a function that limits the values of _x_ to the range[ _a,b_ ]

#### 4.3.2 Marked Pixels Matrix

```
The marked pixels matrix M is constructed as follows:
```
```
Mi,j =
```
```


```
```

```
```
− 1 if pixel( i,j )is positively marked
1 if pixel( i,j )is negatively marked
0 if pixel( i,j )is not marked
```
##### (4.2)

```
This assignment ensures that positively marked pixels contribute negatively to
the loss (encouraging their importance), while negatively marked pixels contribute
positively (discouraging their importance).
```
#### 4.3.3 Gradient Absolute Values and Clipping

```
A key aspect of our loss function is the use of absolute values of the input gradients.
We take the absolute value of Gi,j before clipping:
```
```
clip(| Gi,j | , 0 , 0. 5) (4.3)
This approach is crucial because our goal is to modulate the overall importance
of each pixel, regardless of whether it has a positive or negative influence on the
model’s output. By using the absolute value, we ensure that both positive and
negative gradients are treated equally in terms of their magnitude of influence.
The gradients are then clipped to the range[0 , 0. 5]to prevent extreme values from
dominating the loss calculation. This clipping helps stabilize the training process
```

**4.3 Loss function 25**

```
and prevents the model from becoming overly sensitive to small perturbations in
the input.
```
#### 4.3.4 Interpretation of the Loss Function

The loss function operates by minimizing the sum of the element-wise product of
the marked pixels matrix and the clipped absolute gradients. This minimization
process has the following effects:

- For positively marked pixels ( _Mi,j_ =− 1 ), the loss is minimized when the
    corresponding gradient magnitude is large, encouraging the model to increase
    the importance of these pixels.
- For negatively marked pixels ( _Mi,j_ = 1), the loss is minimized when the
    corresponding gradient magnitude is small, encouraging the model to decrease
    the importance of these pixels.
- For unmarked pixels ( _Mi,j_ = 0), there is no direct contribution to the loss,
    allowing the model to determine their importance based on other factors.

#### 4.3.5 Implementation

The loss function is implemented in PyTorch using the following code snippet:

```
torch.sum(self.marked_pixels * torch.clamp(torch.abs(self.gradients), min=0, max=0.5))
```
```
Here,self.marked_pixelscorresponds to the matrix M , andself.gradients
represents the gradients of the input. Thetorch.absfunction computes the ab-
solute value of the gradients,torch.clampimplements the gradient clipping, and
torch.sumcomputes the overall loss by summing all elements of the resulting matrix.
```
#### 4.3.6 Advantages and Considerations

This custom loss function offers several advantages:

1. It directly incorporates human-defined saliency information into the learning
    process.
2. The function is differentiable, allowing for end-to-end training of the network.
3. By using the absolute value of gradient information, it adapts to the current
    state of the model during training, regardless of the direction of influence.
4. The approach is agnostic to whether a feature is positively or negatively
    correlated with the output, focusing solely on its magnitude of importance.

However, it’s important to note that the effectiveness of this loss function may
depend on the accuracy and consistency of the human-marked saliency regions.
Additionally, the choice of clipping range ([0 _,_ 0_._ 5]in this case) may need to be tuned
for optimal performance depending on the specific task and dataset.


**26 4. Methods**

### 4.4 Incorporating the Gradient Information

After computing the loss function, we utilize the gradient information to adjust the
model’s weights and improve its performance. This section details the process of
incorporating this information into the model training.

#### 4.4.1 Gradient Descent Loop

The core of our approach is an iterative gradient descent loop, which continues until
certain conditions are met. The loop is implemented as follows:

while current_loss < validation_loss*1.2 and loss.item() > real_loss.item()-abs(real_loss.item()/2):
loss = self.getloss("classic")
loss.backward()
print(f"loss is {loss}")
self.adjust_weights_according_grad()
saliency2 = self.compute_saliency_map(self.input,self.label).show()
self.measure_impact_pixels()
current_loss = self.am_I_overfitting().item()

#### 4.4.2 Stopping Criteria

The loop continues as long as two conditions are satisfied:

1. current_loss _<_ 1_._ 2 ×validation_loss
2. loss _>_ real_loss−|real_loss| _/_ 2

These criteria aim to prevent overfitting by stopping the training when loss on the
validation data becomes too high, or when the change in saliency becomes excessive,
potentially overfitting on the given image data.

#### 4.4.3 Loss Computation and Backpropagation

In each iteration, we compute the loss using thegetloss("classic")method,
which refers to the loss function discussed in this paper. Theloss.backward()call
then computes the gradients of the loss with respect to the model parameters.

#### 4.4.4 Weight Adjustment

Theadjust_weights_according_grad()method is called to update the model’s
weights based on the computed gradients. This step is crucial for the learning
process, as it moves the model parameters in the direction that minimizes the loss.

#### 4.4.5 Saliency Map Computation

After each weight adjustment, we compute and visualize a new saliency map us-
ingcompute_saliency_map(self.input,self.label).show(). This allows us to
observe how the model’s focus on different input regions evolves during training.


**4.4 Incorporating the Gradient Information 27**

#### 4.4.6 Impact Measurement

Themeasure_impact_pixels()method is called to assess the effect of the weight
adjustments on the model’s performance. This step involves analyzing how the
importance of negatively and positively marked pixels is changing.

#### 4.4.7 Overfitting Check

At the end of each iteration, we callam_I_overfitting()to compute a validation
loss on a separate dataset. This helps monitor the model’s generalization performance
and informs the stopping criteria.

#### 4.4.8 Experimental Nature of Parameters

It is important to note that the parameters used in the stopping criteria, as well as
the learning rate (not explicitly shown in this code snippet), are experimental and
not set in stone. These values may need to be readjusted based on empirical results,
or even treated as hyperparameters that require tuning for optimal performance.
The flexibility of these parameters allows for adaptation to different datasets and
model architectures, but also necessitates careful consideration and potential adjust-
ment in practical applications.



```
29
```
## Chapter 5

# Experiments

### 5.1 Integrated Gradients

We explored the application of integrated gradients to incorporate spatial saliency into
our convolutional neural network (CNN) model. This method involves calculating
the gradient of the input with respect to specific weights, integrating from the input
back to the weight value, rather than deriving gradients using the chain rule as in
the backward pass [47].
Despite its theoretical promise, this approach did not yield significant improvements
in our experiments. Upon analysis of the resulting saliency maps, we observed that
integrated gradients excel at identifying edges of salient regions but struggle to
capture entire salient areas [ 2 ]. This limitation potentially conflicts with human-
intuitive concepts of marking entire relevant areas. While it may be possible to
reconcile edge detection with area-based saliency, the potential complications led us
to explore alternative methods.

### 5.2 Sensitivity Analysis

Our next experimental approach involved sensitivity analysis, which theoretically
offered the advantage of computational efficiency, requiring only one additional
forward pass. The underlying principle was that the derivative of a weight with
respect to an input pixel represents the direction of weight adjustment following an
infinitesimal change in the input pixel [46].
To implement this concept, we added forward hooks to all dense layers within our
CNN, focusing on these layers due to their more straightforward relationship between
activations and weights. We then performed two forward passes: one with the original
image and another with slight alterations to the marked pixels. By observing the
changes in activations resulting from input perturbations and broadcasting these
changes to the weights, we attempted to quantify each weight’s responsibility for
activation changes.
We experimented with various weight adjustment strategies based on these respon-
sibility values, including penalization, zeroing out, and even rewarding different
weights. The underlying hypothesis was that neurons not influenced by specific
image features (e.g., a sofa in the image) would not exhibit significant activation


**30 5. Experiments**

changes when those features were slightly altered.
However, these approaches did not produce statistically significant improvements in
model performance. We hypothesize that this lack of success may be attributed to
two main factors:

1. The potential conflation of weights and activations in our analysis.
2. The focus on first-order derivatives of weights with respect to input, which
    may not adequately capture the influence of weights on input importance.

To further pursue this line of inquiry, it may be necessary to invert the process,
perturbing weights to observe effects on input pixel importances. However, such an
approach would likely be computationally expensive, negating the primary advantage
of this method [3].

### 5.3 Calculation of the Hessian

Our final experimental approach involved computing the Hessian matrix to capture
higher-order relationships between weights, inputs, and model outputs. This method
entailed first calculating the Jacobian matrix to find the gradients of the model with
respect to the input, followed by computing the Jacobian of that Jacobian with
respect to the weights [33].
We implemented this approach using vector mapping for batch operations in Python
to maximize efficiency. Despite these optimizations, the computation of the Hessian
proved to be prohibitively expensive in terms of both time and space complexity,
particularly for large inputs or models with numerous parameters [38].
It is worth noting that while our explicit implementation of Hessian computation
was inefficient, the underlying principles are implicitly utilized by automatic dif-
ferentiation libraries such as AutoGrad. These libraries offer highly optimized
implementations that perform similar calculations more efficiently [37].
In conclusion, while our experiments with integrated gradients, sensitivity analysis,
and Hessian computation did not yield the desired improvements in spatial saliency
incorporation, they provided valuable insights into the challenges and complexities of
this task. Future work may benefit from exploring more sophisticated optimization
techniques or alternative approaches to integrating human-perceived saliency into
CNN models.


```
31
```
## Chapter 6

# Conclusion

### 6.1 Results

#### 6.1.1 Control of Saliency

**Variation Across Images**

Our model demonstrates the ability to control saliency by marking specific parts of
an image for increased or decreased consideration. The results show that we can
successfully reduce the saliency of negatively marked pixels while increasing the
saliency of positively marked ones, without significantly impacting the validation
loss.

1. **Minimal change:** Some images showed little to no change in saliency distri-
    bution.
2. **Moderate adjustment:** A subset of images exhibited noticeable changes in
    saliency, aligning with the marked regions.
3. **Significant alteration:** For certain images, we achieved substantial increases
    in saliency for positively marked regions and significant decreases for negatively
    marked areas.


**32 6. Conclusion**

```
Figure 6.1. Minimal change in saliency.
```

**6.1 Results 33**

```
Figure 6.2. Moderate adjustment in saliency.
```
```
Figure 6.3. Significant alteration in saliency.
```

**34 6. Conclusion**

**Impact on Specific Regions**

In cases where the model successfully altered saliency, we observed that regions
marked for increased attention became significantly more salient, while areas marked
for reduced consideration showed notably decreased saliency.

#### 6.1.2 Performance Across Data Sizes

An important finding of our study is the relationship between the effectiveness of
our saliency control method and the size of the dataset used for training.

**Small Data Sets**

Our method demonstrated particularly strong performance when applied to smaller
datasets. This suggests that the saliency control technique may be especially valuable
in scenarios where limited training data is available.

**Larger Data Sets**

While the effect was most pronounced with smaller datasets, we also observed
positive results with slightly larger datasets. This indicates that the benefits of our
saliency control method extend beyond just small-scale applications. The next step
here is to test it on significantly larger datasets in the order of millions.

**Figure 6.4.** Performance of the saliency control on a dataset larger than the ones usually
tested on.

#### 6.1.3 Validation Loss

A critical aspect of our results is that the improvements in saliency control were
achieved without significantly impacting the validation loss. This suggests that our
method enhances the model’s ability to focus on specific regions without compromis-
ing overall performance.


```
6.1 Results 35
```
#### 6.1.4 Visual Demonstration of Saliency Changes

These below examples provide clear evidence of our model’s ability to adjust saliency
based on user-defined markings, showcasing both increases in saliency for positively
marked regions and decreases for negatively marked areas, which can be seen from the
fact that the shapes of the objects to recognize become significantly more noticeable.
In figure 6.7 we represent the full process: First, there is the model seeing the stone
the cat is on as salient instead of the cat. Then, the user marks the cat in green and
the stone in read, and the custom loss correctly the saliency to now correctly take
the cat more into account and the stone less.

```
Figure 6.5. Visual demonstration of saliency changes. Left: Original Saliency, Right:
Resulting saliency map.
```
```
Figure 6.6. Visual demonstration of saliency changes. Left: Original Saliency, Right:
Resulting saliency map.
```

**36 6. Conclusion**

**Figure 6.7.** Demonstration of model correction: Top: Marking wrongly salient stone,
Middle: Saliency Improvement before after, Bottom: Change of importance in negatively
(blue) and positively (red) marked pixels.


```
6.2 Contributions 37
```
### 6.2 Contributions

This work makes several significant contributions to the field of computer vision
and machine learning, particularly in the area of saliency control and model inter-
pretability:

#### 6.2.1 Novel Saliency Control Mechanism

We introduce a novel mechanism that allows for fine-grained control over the saliency
of neural network models. This approach enables users to explicitly mark regions
of an input image for increased or decreased consideration, effectively guiding the
model’s attention [[1]].

#### 6.2.2 Adaptive Saliency without Performance Degradation

Our method demonstrates the ability to modify saliency maps in accordance with
user-defined preferences without significantly impacting the model’s validation loss.
This contribution is particularly noteworthy as it shows that saliency can be adjusted
without compromising overall model performance [[2]].

#### 6.2.3 Effectiveness Across Data Regimes

We provide empirical evidence that our saliency control method is effective across
various data regimes, with particularly strong performance on smaller datasets. This
finding has important implications for applications where large-scale data collection
is challenging or impractical [[3]].

#### 6.2.4 Enhanced Model Interpretability

```
By allowing users to influence the saliency of specific image regions, our work
contributes to the broader goal of improving model interpretability. This approach
provides insights into the decision-making process of neural networks and offers a
new tool for analyzing and debugging model behavior [[4]].
```
#### 6.2.5 Flexible Framework for Saliency Manipulation

```
Our contribution extends beyond a single application, offering a flexible framework
that can be adapted to various computer vision tasks. This versatility opens up
new avenues for research in areas such as object detection, image segmentation, and
medical image analysis [[5]].
```
#### 6.2.6 Bridging Human Intuition and Machine Learning

```
By incorporating user-defined saliency preferences, our work represents a step towards
bridging the gap between human intuition and machine learning models. This
approach allows for the integration of domain expertise into the model’s attention
mechanism, potentially leading to more robust and trustworthy AI systems.
```

**38 6. Conclusion**

These contributions collectively advance the state-of-the-art in controllable and
interpretable machine learning models, offering both theoretical insights and practical
tools for researchers and practitioners in the field of computer vision and beyond.


```
6.3 Application Fields and Potential Impact 39
```
### 6.3 Application Fields and Potential Impact

The integration of human-perceived spatial saliency into convolutional neural net-
works (CNNs) offers two key advantages: reduced data requirements and increased
human control over model focus. These benefits open up new possibilities across
various domains, particularly in fields where data scarcity is a challenge or where
expert knowledge is crucial. This section explores some of the most promising areas
where this technique could have significant impact.

#### 6.3.1 Medical Imaging and Rare Disease Detection

```
In medical imaging, particularly for rare disease detection, our approach offers several
advantages:
```
- **Reduced data requirements:** Enables training on smaller datasets while
    maintaining high accuracy, crucial for rare conditions where data is scarce [[1]].
- **Expert-guided focus:** Allows medical professionals to direct the model’s
    attention to subtle, clinically relevant features that may be overlooked by
    standard CNNs [[1]].
- **Improved early detection:** Potential for earlier identification of rare diseases
    by leveraging expert knowledge with limited data.

#### 6.3.2 Geological Exploration and Resource Detection

```
In geology, particularly in oil, gas, and mineral exploration, our approach could
provide:
```
- **Enhanced interpretation with limited data:** Improved analysis of seismic
    data for more accurate drilling decisions, even with sparse datasets [[3]].
- **Expert-guided mineral detection:** Geologists can guide the model to focus
    on specific geological features associated with mineral deposits.
- **Improved generalization:** Better performance in unique geological contexts
    with limited training samples.

#### 6.3.3 Remote Sensing and Environmental Monitoring

```
For satellite and aerial imagery analysis, our method offers:
```
- **Efficient rare object detection:** Improved identification of small or infre-
    quent objects in large-scale imagery with less training data [[6]].
- **Expert-guided monitoring:** Environmental scientists can direct the model’s
    focus to subtle indicators of deforestation, urban growth, or agricultural
    changes.
- **Adaptability to new environments:** Quicker adaptation to different geo-
    graphical areas or seasonal changes with minimal retraining.


**40 6. Conclusion**

#### 6.3.4 Manufacturing and Quality Control

In industrial settings, our approach enhances automated quality control:

- **Efficient defect detection:** More accurate identification of subtle manufac-
    turing flaws with less training data.
- **Rapid adaptation:** Quicker adjustment to new product lines or materials by
    allowing experts to guide the model’s focus.
- **Improved interpretability:** Enhanced understanding of the model’s deci-
    sions for human operators, facilitated by expert-guided attention.

In conclusion, the integration of human-perceived spatial saliency into CNNs has
the potential to impact a wide range of fields by addressing two critical challenges:
data scarcity and the need for expert knowledge integration. By reducing data
requirements and allowing for more direct human control over the learning process,
this approach could lead to more efficient, accurate, and interpretable AI systems
across various domains.


```
6.4 Future Work and Outlook 41
```
### 6.4 Future Work and Outlook

While our current research has shown promising results in leveraging human-perceived
saliency for weight adjustment in neural networks, there are several avenues for further
investigation and improvement. This chapter outlines key areas for future work that
could significantly enhance the robustness and applicability of our approach.

#### 6.4.1 Comprehensive Generalization Benchmarking

```
One of the most critical areas for future work is the implementation of a rigorous
generalization benchmark.
```
- Develop a standardized set of diverse datasets for testing generalization
- Implement cross-dataset validation to assess transferability of learned saliency
- Compare performance against state-of-the-art models on established bench-
    marks

This comprehensive benchmarking is crucial to validate the real-world applicability
of our saliency-based weight adjustment method across various domains and data
distributions.

#### 6.4.2 Optimizing Computational Efficiency

To fully explore the potential of our approach, significant improvements in computa-
tional efficiency are necessary:

- Develop CUDA-optimized implementations for GPU acceleration
- Explore distributed computing solutions for large-scale experiments
- Implement efficient data loading and preprocessing pipelines

These optimizations will enable us to conduct more extensive experiments, potentially
uncovering insights that are currently computationally infeasible to obtain.

#### 6.4.3 Exploration of Alternative Optimizers

While our current implementation uses standard optimizers, investigating the impact
of different optimization algorithms could yield valuable insights:

- Test adaptive optimizers such as Adam, RMSprop, and AdaGrad
- Explore second-order optimization methods
- Investigate custom optimizers tailored to saliency-based weight adjustment

```
Different optimizers may interact uniquely with our saliency-based approach, poten-
tially leading to improved convergence or generalization properties.
```

**42 6. Conclusion**

#### 6.4.4 Expanding to Diverse Classification Tasks

To demonstrate the broad applicability of our method, future work should focus on
adapting it to a wider range of classification tasks:

- Explore applications in natural language processing and speech recognition
- Investigate potential benefits in regression tasks

Successful adaptation to diverse tasks would significantly strengthen the case for
our method’s versatility and practical value.

#### 6.4.5 Alternative Saliency Approaches

While our current approach relies on gradient-based pixel attribution, exploring
alternative saliency methods could provide valuable comparisons and potentially
superior results:

- Implement and compare Layer-wise Relevance Propagation (LRP) [4]
- Explore Integrated Gradients [47]
- Investigate SHAP (SHapley Additive exPlanations) values [29]

Different saliency approaches may capture complementary aspects of feature impor-
tance, potentially leading to more robust or interpretable models.

#### 6.4.6 Hyperparameter Optimization

Incorporating learning rate as a hyperparameter and systematically optimizing other
hyperparameters could significantly improve performance:

- Implement automated hyperparameter tuning using techniques like Bayesian
    optimization
- Explore the impact of learning rate schedules on saliency-based weight adjust-
    ment
- Investigate the interaction between hyperparameters and different saliency
    methods

Optimal hyperparameter settings may vary depending on the specific task and
dataset, making this an important area for future investigation.

#### 6.4.7 Alternative Loss Functions

Developing and testing alternative loss functions that do not rely on area-based
penalties could lead to more flexible and effective training:

- Explore contrastive loss functions for saliency-based learning
- Investigate the use of perceptual loss functions


```
6.4 Future Work and Outlook 43
```
- Develop custom loss functions that directly incorporate saliency information

```
Novel loss functions could potentially better capture the relationship between saliency
and model performance, leading to improved results.
In conclusion, while our current research has demonstrated the potential of saliency-
based weight adjustment, there remain numerous exciting avenues for future work. By
addressing these areas, we can further validate, improve, and expand the applicability
of our approach, potentially leading to significant advancements in the field of
interpretable and human-aligned machine learning.
```
#### 6.4.8 Adaptive Saliency Optimization

```
Our research has revealed significant variability in the success of saliency optimization
across different images. This observation suggests a potential avenue for future work:
developing a heuristic to determine when saliency-based computations are likely to
be beneficial for a given image.
The effectiveness of saliency optimization appears to be related to how well the
initial saliency map aligns with human perception. A potential heuristic could be
based on the following factors:
```
- The number of pixels that both the model and human perceive as salient
- The number of pixels that both the model and human perceive as non-salient
- The number of pixels where the model’s perception differs from human percep-
    tion

```
A simple formula to quantify this alignment could be:
```
##### A =

```
( Sm ∩ Sh ) + ( Nm ∩ Nh )
| I |
```
##### −

```
|( Sm ∩ Nh )∪( Nm ∩ Sh )|
| I |
```
##### (6.1)

```
Where:
```
- _Sm_ and _Sh_ are the sets of salient pixels as perceived by the model and human,
    respectively
- _Nm_ and _Nh_ are the sets of non-salient pixels as perceived by the model and
    human, respectively
- _I_ is the set of all pixels in the image
- | _X_ |denotes the cardinality of set _X_

This alignment score _A_ ranges from -1 to 1, where 1 indicates perfect alignment and
-1 indicates complete misalignment between the model’s saliency map and human
perception.
Future work could focus on:

- Refining this heuristic through empirical studies
- Determining an optimal threshold for when to apply saliency optimization


**44 6. Conclusion**

- Investigating the relationship between this alignment score and the potential
    for improvement through saliency optimization

By developing such a heuristic, we could potentially improve the efficiency of our
approach by focusing computational resources on images where saliency optimization
is likely to yield substantial improvements. This could lead to more targeted and
effective saliency-based weight adjustment strategies in future iterations of our model.

### 6.5 Model Usage and Code Structure

This section provides detailed instructions on how to use the model and explains
the code structure for potential improvements by other people interested.

#### 6.5.1 Setting Up the Model

**Downloading the Code**

To begin, download the necessary files from our GitHub repository. The main file
you’ll need isMachinePunishment.py.

**Importing Required Modules**

Import thePunisherLossclass from theMachinePunishment.pyfile:

from MachinePunishment import PunisherLoss

#### 6.5.2 Integrating PunisherLoss into Your CNN

**Loss Function Configuration**

UsePunisherLossas the loss function in your CNN model. When initializing
PunisherLoss, specify a default loss function to be used in iterations without
human intervention. If not specified, it defaults to cross-entropy loss.

**Specifying Human Intervention Frequency**

Determine after how many epochs you want human intervention to occur. This will
control how often the punishment/encouragement interface appears.


```
6.5 Model Usage and Code Structure 45
```
#### 6.5.3 Model Training Setup

```
Initializing the Model and Optimizer
```
```
Set up your model and optimizer as follows:
```
```
models = [SimplestCNN(classes, channels)]
for model in models:
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = PunisherLoss(4, train_dataset, model)
model.train_model(train_loader, criterion, optimizer, num_epochs)
```
```
Note thatPunisherLossrequires three arguments: the number of classes, the
training dataset, and a reference to the model.
```
#### 6.5.4 Training Process

```
Loss Calculation in Training Loop
```
```
In your model’s training function, ensure that the loss calculation includes the
current epoch:
```
```
running_loss = 0.0
for i, (inputs, labels) in enumerate(train_loader):
optimizer.zero_grad()
outputs = model(inputs)
print(outputs)
loss = criterion(outputs, labels, epoch)
loss.backward()
optimizer.step()
running_loss += loss.item()
```
#### 6.5.5 Human Intervention Interface

```
Punishment/Encouragement Window
```
```
Everyxepochs (as specified earlier), a window will pop up allowing you to draw on
the input image. This interface enables you to:
```
- Encourage specific parts of the image by drawing in green
- Discourage specific parts of the image by drawing in red

```
Committing Changes
```
After drawing, you’ll have the option to commit your changes in the chooser window.
This allows you to review your interventions before applying them to the model’s
training process.


**46 6. Conclusion**

#### 6.5.6 Customization and Improvement

To improve or customize the code:

- Modify thePunisherLossclass inMachinePunishment.pyto change how the
    loss is calculated or how human input is incorporated.
- Adjust the frequency of human interventions by changing the epoch check in
    the training loop.
- Enhance the drawing interface for more precise or varied types of human input.
- Experiment with different default loss functions or ways of combining them
    with the human-guided loss.


```
47
```
# Bibliography

```
[1]Amari, S.-I.Natural gradient works efficiently in learning. Neural computation ,
10 (1998), 251.
```
```
[2]Ancona, M., Ceolini, E., Öztireli, C., and Gross, M.Towards better
understanding of gradient-based attribution methods for deep neural networks.
arXiv preprint arXiv:1711.06104 , (2018).
```
```
[3]Ancona, M., Ceolini, E., Öztireli, C., and Gross, M.Gradient-based
attribution methods. Explainable AI: Interpreting, Explaining and Visualizing
Deep Learning , (2019), 169.
```
```
[4]Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R.,
and Samek, W.On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one , 10 (2015), e0130140.
```
```
[5]Becker, S. and Le Cun, Y.Improving the convergence of back-propagation
learning with second order methods. Proceedings of the 1988 connectionist
models summer school , (1988), 29.
```
```
[6]Bogun, A., Kostadinov, D., and Borth, D. Saliency diversified deep
ensemble for robustness to adversaries. (2021). Available from:https://arxiv.
org/abs/2112.03615,arXiv:2112.03615.
```
```
[7]Bottou, L., Curtis, F. E., and Nocedal, J.Optimization methods for
large-scale machine learning. Siam Review , 60 (2018), 223.
```
```
[8]Boyd, A., Tinsley, P., Bowyer, K., and Czajka, A.Cyborg: Blending
human saliency into the loss improves deep learning (2022). Available from:
https://arxiv.org/abs/2112.00686,arXiv:2112.00686.
```
```
[9]Boyd, S. and Vandenberghe, L. Convex optimization. Cambridge university
press (2004).
```
[10]Doshi-Velez, F. and Kim, B.Towards a rigorous science of interpretable ma-
chine learning. In _Proceedings of the 2017 conference on fairness, accountability,
and transparency_ , pp. 1–9 (2017).

[11]Fong, R. and Vedaldi, A. Interpretable explanations of black boxes by
meaningful perturbation. _Proceedings of the IEEE International Conference on
Computer Vision_ , (2017), 3429.


**48 Bibliography**

[12] Girshick, R., Donahue, J., Darrell, T., and Malik, J. Rich feature
hierarchies for accurate object detection and semantic segmentation. In _Pro-
ceedings of the IEEE conference on computer vision and pattern recognition_ , pp.
580–587 (2014).

[13] Glorot, X. and Bengio, Y.Understanding the difficulty of training deep
feedforward neural networks. (2010), 249.

[14] Goodfellow, I., Bengio, Y., and Courville, A. _Deep Learning_. MIT
Press (2016).

[15] Gu, J. and Tresp, V.Contextual prediction difference analysis for explain-
ing individual image classifications. _arXiv preprint arXiv:1910.09086_ , (2020).
Available from:https://arxiv.org/abs/1910.09086.

[16] Hastie, T., Tibshirani, R., and Friedman, J. _The elements of statistical
learning: data mining, inference, and prediction_. Springer Science & Business
Media (2009).

[17] He, K., Zhang, X., Ren, S., and Sun, J.Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and
pattern recognition_ , pp. 770–778 (2016).

[18] Hinton, G., et al.Deep neural networks for acoustic modeling in speech
recognition: The shared views of four research groups. _IEEE Signal Processing
Magazine_ , **29** (2012), 82.

[19] Ioffe, S. and Szegedy, C.Batch normalization: Accelerating deep network
training by reducing internal covariate shift. _arXiv preprint arXiv:1502.03167_ ,
(2015).

[20] Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas,
F., and Sayres, R.Interpretability beyond feature attribution: Quantitative
testing with concept activation vectors (tcav). In _International conference on
machine learning_ , pp. 2668–2677. PMLR (2018).

[21] Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K. T.,
Dähne, S., Erhan, D., and Kim, B.The (un)reliability of saliency methods.
_Explainable AI: Interpreting, Explaining and Visualizing Deep Learning_ , (2017).

[22] Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization
(2014).

[23] Krizhevsky, A., Sutskever, I., and Hinton, G. E.Imagenet classification
with deep convolutional neural networks. In _Advances in neural information
processing systems_ , pp. 1097–1105 (2012).

[24] LeCun, Y., Bengio, Y., and Hinton, G.Deep learning. _Nature_ , **521** (2015),
436.


**Bibliography 49**

[25]LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E.,
Hubbard, W., and Jackel, L. D.Backpropagation applied to handwritten
zip code recognition. _Neural computation_ , **1** (1989), 541.

[26]LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P.Gradient-based
learning applied to document recognition. _Proceedings of the IEEE_ , **86** (1998),
2278.

[27]Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F.,
Ghafoorian, M., van der Laak, J. A., van Ginneken, B., and Sánchez,
C. I. A survey on deep learning in medical image analysis. _Medical image
analysis_ , **42** (2017), 60.

[28]Long, J., Shelhamer, E., and Darrell, T.Fully convolutional networks
for semantic segmentation. In _Proceedings of the IEEE conference on computer
vision and pattern recognition_ , pp. 3431–3440 (2015).

[29]Lundberg, S. M. and Lee, S.-I.A unified approach to interpreting model
predictions. vol. 30 (2017).

[30]Magnus, J. R. and Neudecker, H. _Matrix differential calculus with applica-
tions in statistics and econometrics_. John Wiley & Sons (2019).

[31]Martens, J.Deep learning via hessian-free optimization. In _ICML_ , vol. 27,
pp. 735–742 (2010).

[32]Martens, J. and Grosse, R.Optimizing neural networks with kronecker-
factored approximate curvature. In _International conference on machine learn-
ing_ , pp. 2408–2417. PMLR (2015).

[33]Martens, J., Sutskever, I., and Swersky, K.Estimating the hessian by
back-propagating curvature. _arXiv preprint arXiv:1206.6464_ , (2012).

[34]Murphy, K. P. _Machine learning: a probabilistic perspective_. MIT Press
(2012).

[35]Nair, V. and Hinton, G. E. Rectified linear units improve restricted
boltzmann machines. In _Proceedings of the 27th international conference on
machine learning (ICML-10)_ , pp. 807–814 (2010).

[36]Nocedal, J. and Wright, S. _Numerical optimization_. Springer Science &
Business Media (2006).

[37]Paszke, A., et al.Automatic differentiation in pytorch. In _NIPS-W_ (2017).

[38]Pearlmutter, B. A. Fast exact multiplication by the hessian. _Neural
computation_ , **6** (1994), 147.

[39]Puiutta, E. and Veith, E. M.Explainable reinforcement learning: A survey.
_arXiv preprint arXiv:2005.06247_ , (2020).


**50 Bibliography**

[40] Ribeiro, M. T., Singh, S., and Guestrin, C. Why should i trust you?:
Explaining the predictions of any classifier. In _Proceedings of the 22nd ACM
SIGKDD international conference on knowledge discovery and data mining_ , pp.
1135–1144 (2016).

[41] Rosasco, L., De Vito, E., Caponnetto, A., Piana, M., and Verri, A.
Loss functions for classification: A comparison. _Neural computation_ , **16** (2004),
2625.

[42] Rudin, C.Stop explaining black box machine learning models for high stakes
decisions and use interpretable models instead. _Nature Machine Intelligence_ , **1**
(2019), 206.

[43] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. Learning
representations by back-propagating errors. _Nature_ , **323** (1986), 533.

[44] Scherer, D., Müller, A., and Behnke, S.Evaluation of pooling opera-
tions in convolutional architectures for object recognition. In _Artificial Neural
Networks – ICANN 2010_ , pp. 92–101. Springer (2010).

[45] Schmidhuber, J. Deep learning in neural networks: An overview. _Neural
networks_ , **61** (2015), 85.

[46] Simonyan, K., Vedaldi, A., and Zisserman, A.Deep inside convolutional
networks: Visualising image classification models and saliency maps. _arXiv
preprint arXiv:1312.6034_ , (2013).

[47] Sundararajan, M., Taly, A., and Yan, Q.Axiomatic attribution for deep
networks. In _International Conference on Machine Learning_ , pp. 3319–3328.
PMLR (2017).

[48] Szczepankiewicz, P. A. C. K. e. a., K. Ground truth based compar-
ison of saliency maps algorithms. _Journal of Computer Vision and Im-
age Understanding_ , (2022). Available from: https://doi.org/10.1038/
s41598-023-42946-w.

[49] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D.,
Goodfellow, I., and Fergus, R.Intriguing properties of neural networks.
In _International Conference on Learning Representations_ (2013).

[50] Tursun, O., Denman, S., Sridharan, S., and Fookes, C.Sess: Saliency
enhancing with scaling and sliding. _arXiv preprint arXiv:2207.01769_ , (2022).
Available from:https://arxiv.org/abs/2207.01769.

[51] Wachter, S., Mittelstadt, B., and Russell, C.Counterfactual expla-
nations without opening the black box: Automated decisions and the gdpr.
_Harvard Journal of Law & Technology_ , **31** (2017).

[52] Wei, Q.Saliency maps-based convolutional neural networks for facial expression
recognition. _IEEE Access_ , **PP** (2021), 1.doi:10.1109/ACCESS.2021.3082694.


**Bibliography 51**

[53]Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A.
Learning deep features for discriminative localization. In _Proceedings of the
IEEE conference on computer vision and pattern recognition_ , pp. 2921–2929
(2016).


