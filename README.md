# Fork details
- Updated to work with Gymasium 1.0.0.
- Minor superficial changes for readability. The math is the same as the original.

# gym-inventory

The Inventory environment is a single agent domain featuring discrete state and action spaces. Currently, one task is supported:

## Simple Inventory Control with Lost Sales

This environment corresponds to the version of the inventory control with lost sales problem described in Example 1.1 in [Algorithms for Reinforcement Learning by Csaba Szepesvari (2010)](https://sites.ualberta.ca/~szepesva/RLBook.html).

## Future tasks

Future tasks will have more complex environments that take into account:

 * Demand-effecting factors such as trend, seasonality, holidays, weather, etc.
 * Business-specific factors such as lead times, penalties for carrying inventory, etc.
 * Scaling from 1 to millions of SKUs and learning across SKUs.
