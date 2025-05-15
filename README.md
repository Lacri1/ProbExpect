# ProbExpect - Probability Expectation Calculator

**ProbExpect** is a simple web-based calculator for estimating the expected cost and number of attempts required to achieve at least one success in probabilistic scenarios with replacement — such as gacha pulls in games.

## Purpose

This tool was built to make it easier to understand and calculate expected values in situations where:

- You are drawing items with a known fixed probability
- Each attempt is independent (with replacement)
- You want to know how many trials it typically takes to succeed
- You want to estimate how much it will cost to reach a certain probability of success

## Features

- Input success probability, cost per attempt, and number of items drawn per trial
- Calculates and visualizes cumulative success probability
- Provides key statistics:
  - Attempts and cost for 20% success rate (top 20%)
  - Attempts and cost for 63.2% success rate (average case)
  - Attempts and cost for 80% success rate (top 80%)
- Optimized probability chart that scales based on input probability
