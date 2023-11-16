# LUPUMA: Low Complexity Unitary Principal-Singular-Vector Utilization for Model Analysis

## Introduction
This repository contains the implementation of LUPUMA, a novel algorithm for single-tone frequency estimation of one-dimensional complex signals in complex white Gaussian noise. LUPUMA stands out for its low space and time complexity, making it highly efficient for applications with limited resources and requiring rapid processing.

## Abstract
LUPUMA, based on subspace methods and unitary transformation, offers a uniform estimation variance across the entire frequency spectrum. It excels in short observations by achieving the lowest time complexity among its counterparts and asymptotically reaching the Cramér-Rao Lower Bound. The algorithm's robustness is evident in its performance, matching that of maximum likelihood estimators under specific signal-to-noise ratio conditions.

## Algorithm Overview
LUPUMA operates by employing subspace techniques combined with unitary transformations. Key highlights include:
- Uniform variance estimation over the full frequency range.
- Optimal performance in scenarios with short observation lengths.
- Ideal for situations with limited sample numbers, computational power, or memory.
- Fast processing suitable for low-latency applications.

## Usage
This repository contains the Python implementation of LUPUMA and related algorithms (e.g., PUMA, Unitary-PUMA, DFT-WLS, A&M, Parabolic estimators). A Jupyter Notebook is also provided to demonstrate how to use these algorithms for frequency estimation.

### Prerequisites
- Python 3.x
- NumPy
- SciPy

### Running the Code
Clone the repository and navigate to the folder containing the scripts. Use the Jupyter Notebook for a step-by-step guide on implementing LUPUMA and other algorithms for your specific use case.

## Citing Our Work
If you find LUPUMA or any part of this repository useful in your research or work, please consider citing our paper. This will help us in our efforts to advance the field and provide recognition for our work. Here is the citation format:

Alireza Pourafzal , Pavel Skrabanek , Michael Cheffena , Sule Yildirim ,
Thomas Roi-Taravella , Low Complexity Subspace Approach for Unbiased Frequency Estimation of a
Complex Single-tone, Digital Signal Processing (2023), doi: https://doi.org/10.1016/j.dsp.2023.104304

## Original Paper
For a detailed explanation of the algorithm and its comprehensive analysis, refer to our published paper. 
http://dx.doi.org/10.1016/j.dsp.2023.104304
