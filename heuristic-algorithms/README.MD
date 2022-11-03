# Heuristic Algorithms for the N-Queen Problem 
The Heuristic Functions Used (Summary):
1. Computes and sums the number of queens attacking each queen.
2. Counts the number of queens that are not under attack.
These are the heuristic functions we use to compare states with each other.

Our two search approches:
1. Search for the solution using the hill climbing algorithm.
2. Search for the solution using the hill climbing algorithm with randomness added.
The randomess helps the algorithm to scape from the local optimum when stuck.
The randomness is added by using a random child from the children list instead of the first.
The randomness is applied when we do not see any improvement in the heuristic score.

---

The code can be run as follows:
```
Example:
> python n-queen.py --n=16 --heuristic=1 --time-out=5 --verbose
> python n-queen.py --n=8 --heuristic=1 --randomness --verbose
> python n-queen.py --n=12 --heuristic=2 --randomness --time-out=30 --verbose
```


Use --help to understand the options:
```
> python n-queen.py --help
```

For example:
```
> python n-queen.py --n=8 --heuristic=2 --randomness
```
Output:
```
Heuristic function 2 is used.
Randomness is used.

------------------------------------------------------------

Solution for 8x8 Queens Problem found!
[[0 0 0 0 0 1 0 0]
 [0 0 1 0 0 0 0 0]
 [1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 1 0]
 [0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 1]
 [0 1 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]]
Heuristic Score = 8

------------------------------------------------------------

Time taken: 1.982863187789917 seconds
```
