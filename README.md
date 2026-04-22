# Targeted Fine-Grained Frequency Poisoning Attacks on Set-Valued LDP Mechanisms

## Project Description
This project hosts the source code and resources for reproducing the experiments detailed in our paper. The core focus is to evaluate the vulnerability of various set valued data LDP mechanisms to fine-grained targeted poisoning attacks.

## Contents and Structure
The repository is organized as follows:

### dataset:
Contains the two real-world datasets utilized throughout our empirical evaluation. These datasets serve as the ground truth for assessing the effectiveness of the attacks.

### frequency:
This module is dedicated to the implementation of LDP protocols and their corresponding attacks. Specifically, we provide code for four LDP mechanisms:
- dRAPPOR
- OLH-Sampling
- PrivSet
- Wheel
For each protocol, we have implemented two types of privacy attacks:
TFIPA (Fine-Grained Targeted Input Poisoning Attack)
TFOPA (Fine-Grained Targeted Output Poisoning Attack)

### AOA:
This directory contains the comprehensive comparison framework. It includes the code for our novelly proposed attacks as well as re-implementations of the state-of-the-art attack from prior work. This setup allows for a direct and fair experimental comparison of their performance metrics.
