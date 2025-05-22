# ProbExpect - Probability Expectation Calculator

**ProbExpect** is a simple web-based calculator for estimating the expected cost and number of attempts required to achieve at least one success in probabilistic scenarios with replacement — such as gacha pulls in games.  
link: [https://lacri1.github.io/ProbExpect/](https://lacri1.github.io/ProbExpect/)

---

## Purpose

This tool was built to make it easier to understand and calculate expected values in situations where:

- You are drawing items with a known fixed probability
- Each attempt is independent (with replacement)
- You want to know how many trials it typically takes to succeed
- You want to estimate how much it will cost to reach a certain probability of success

---

## Features

- Input success probability, cost per attempt, and number of items drawn per trial
- Option to set a target number of successes to calculate probability of multiple wins
- Calculates and visualizes cumulative success probability over a range of attempts
- Provides key statistics for attempts and cost at:
  - 20% success probability (top 20%)
  - 63.2% success probability (average expected case)
  - 80% success probability (top 80%)
- Optimized adaptive probability chart scaling based on input probability and target
- Displays results with formatted numbers (including exponential notation for large values)
- Responsive and interactive chart with detailed tooltips
- Ceiling system logic using Monte Carlo simulation:
  - **Normal ceiling**: Guarantees a success if no success is achieved within a fixed number of attempts (the ceiling count)
  - **Mileage ceiling**: Guarantees +1 success every fixed number of attempts, regardless of previous results

---

## Usage

1. Enter the success probability per single attempt (in %).
2. Enter the cost per attempt.
3. Specify the number of items drawn per trial (batch size).
4. (Optional) Check "목표 당첨 횟수 설정" to set a target number of successes.
5. Click the **계산하기 (Calculate)** button to view the probability graph and statistics.
6. Review the graph and key stats to understand expected attempts and costs.

---

## Implementation Details

- Built with React using functional components and hooks.
- Uses memoization and callbacks to optimize performance on heavy computations.
- Prevents calculations under conditions of extreme complexity (e.g., very low probability with target wins and ceiling enabled) to avoid performance issues.
- Probability calculations consider both single and multiple win scenarios.
- Chart.js (via `react-chartjs-2`) used for rendering the probability graph with custom tooltip formatting.
- Monte Carlo simulation is used to support ceiling mechanics:
  - Normal pity guarantees a success if none occur within the ceiling count
  - Mileage-style pity adds one guaranteed win every fixed number of attempts
