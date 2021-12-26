# ELSA\_CANONICAL\_PROJECTION

In the paper: [T. J. Ham et al., "ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks," 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), 2021, pp. 692-705, doi: 10.1109/ISCA52012.2021.00060](https://taejunham.github.io/data/elsa_isca21.pdf), an approximation scheme inspired by [Sign Random Projection SRP](https://en.wikipedia.org/wiki/Random_projection) for the dot product is proposed. Given two vectors _x_ and _y_ of dimension __d__, it takes 3 main steps:
1. Project _x_ and _y_ to a random orthogonal family of __k__ vectors,
2. Use the Hamming distance on the signs of the obtained coordinates to approximate the angle _a_ between _x_ and _y_,
3. Compute `|x| * |y| * cos(a)`.

Note that the first step (the projection) requires `k * d` multiplications for each vector.

In the paper, the authors end up choosing to use `k = d = 64`, but we argue that this choice makes the first step superfluous. If `k = p`, then there is no dimensionality reduction taking place, and one might as well use the canonical base to estimate the angle.

As a reminder, the canonical base of R(N) is the list of vectors: `b0 = (1 0 0 .... 0 0)`, `b1 = (0 1 0 .... 0 0)`, ..., `bn-2 = (0 0 0 ... 1 0)`, `bn-1 = (0 0 0 ... 0 1)`, i.e. the base in which the coefficient for the vectors are already written. In particular, for `x = (x0, x1, x2, ..., xn-2, xn-1)` we have `<x, bi> = xi`.

The goal of this experiment is twofold:
 - Check that the first step can indeed be skipped if `d = k`, i.e. that the estimations with a random base and the canonical base are close enough.
 - Check the quality of the final estimation.

For the experiment we are going to define four values. For any vectors _x_ and _y_:
 - The __Reference Value__ is the real result of the dot product between _x_ and _y_, i.e. `<x, y>`.
 - The __Projected Value__ is the real result of the dot product after the random project, i.e. after step 1. defined above.
 - The __Random Estimation__ is the result of the estimation scheme as described in the ELSA paper, with the random projection.
 - The __Canonical Estimation__ is the result of the proposed estimation scheme, skipping the random projection altogether.

# CODE

The provided script used to test the different estimations is written in python3 and uses __numpy__ for the computations as well as __matplotlib__ for the plots. Running the script from the terminal should plot the different graphs and print the statistic measures on the terminal. The number of experiments can be changed from the code, look at the bottom of the file. The default number of experiments is one million.

Example usage:

```
python3 script.py
```

# PLOT RESULTS

Upon execution, the provided script will plot several views of the Canonical and Random estimation methods proposed in the paper.

This first plot shows how close the estimations are when using the Canonical base and the Random base for the Hamming estimation of the angle between two random vectors. On the left we can see the histogram showing the repartition of the distance between the two estimators, while on the right we can see the progression of the error (useful to see the quartiles graphically).

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/absolute_error_1.png "Absolute Distance Between Canonical and Random Hamming")

This second plot shows the same distance, but this time as a percentage of the estimated value. This help tell how meaningful the distance between the two estimators really is.

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/relative_error_1.png "Relative Distance Between Canonical and Random Hamming")

The error in the next plots are comparing the two estimators with the reference value of the dot product in order to tell how far-off they really are, and how much of an approximation we are making when using the Hamming estimation for the angle. The third suplot shows how much (or little in this case) error is introduced by the random projection if we don't use the Hamming estimation but instead the ideal computation.

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/absolute_error_2.png "Absolute Error of Canonical and Random Estimators")

The same error can also be plotted in percentage, but the plot is quite out of scale.

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/relative_error_2.png "Relative Error of Canonical and Random Estimators")

Plotting with a smaller nummber of tries yields a more readable graph for the percentage of error of each method.

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/relative_error_2_small.png "Relative Error of Canonical and Random Estimators with Fewer Samples")

Finally, we show for 20 sample vector pairs how the Canonical and Random estimations of the dot product compare against the real value.

![](https://github.com/roadelou/ELSA_Canonical_Projection/raw/main/img/samples.png "Sample of Error of Canonical and Random Estimators")

# QUANTITATIVE RESULTS

Running the provided script with a large numbre of samples yields some statistical measures about the quality of the estimations.

## CANONICAL VS RANDOM

This section contains results comparing the estimation of the Hamming distance using the canonical base versus using a random base.

The output of the script was:

```
Absolute Errors...
Max:  21
Mean:  3.236413
Quartiles:  1 3 5

Relative Errors...
Max:  79.24528301886792
Mean:  10.235980549733435
Quartiles:  3.389830508474576 8.695652173913043 14.814814814814815
```

This means that the maximum distance between the Hamming distance in the canonical base and the one in the random base was at most 21 units, i.e. 79% of uncertainty. On average, the distance was only 3.24 i.e. 10% of uncertainty. 25% of time, the canonical and random estimations were within 3.4% of each other, 50% of the time within 8.6% of each other and finally 75% within 15% of each other.

From this data we can conclude that the canonical and random estimates yield approximately the same results on random data, and thus the random projection in the ELSA architecture is likely superfluous.

## ESTIMATION VS REAL

This section contains results comparing the estimation of dot product of two vectors with its real value. Three type of errors are shown:
 - The error of the Hamming estimation using the Canonical base (i.e. without Random Projection)
 - The error of the Hamming estimation using the Random base
 - The error of the ideal dot product once projected in the Random base (i.e. without Hamming Estimation)

The output of the script was:
```
Absolute Errors...
Max Canonical:  1588.6928521854027
Max Random:  1713.059533978723
Max Projection:  1.0231815394945443e-12
Mean Canonical:  312.47448742509806
Mean Random:  334.7670995484917
Mean Projection:  1.4511795187743813e-13
Quartiles Canonical: 137.54307896510707 281.06760077345706 453.59824058907066
Quartiles Random: 142.805655205246 295.16392062528627 485.3167355527122

Relative Errors...
Max Canonical:  28817648.93482824
Max Random:  55159166.17904965
Max Projection:  1.847539673173763e-08
Mean Canonical:  1201.989254347752
Mean Random:  1334.275877686253
Mean Projection:  5.262444479990217e-13
Quartiles Canonical: 65.33196077350453 153.58338940474073 358.9496345302443
Quartiles Random: 67.98046074693188 162.4086279458619 384.10928267117595
```

The data yields a few noteworthy findings:
 - The projection in the Random base introduces very little error on its own. Even though it seems superfluous in the ELSA architecture, it doesn't have a noticeable negative effect on precision.
 - Both the Canonical and Random Hamming estimations of the dot product are quite far off from the truth on random data, with at least 150% of error 50% of the time.

We can thus guess that the Self-Attention layers must be very redundant for the layer to yield the expected output after the Hamming approximation is used.

# CONCLUSION

From the python3 simulation, I would conclude that:
 - The Canonical and Random estimations are equivalent and that the preprocessing multiplications from the ELSA architecture can be skipped (just for the hash computation, the norms are still needed).
 - Both estimations are pretty poor and shouldn't be used in precision-sensitive applications.

### METADATA

Field | Value
--- | ---
:pencil: Contributors | roadelou
:email: Contacts | 
:date: Creation Date | 2021-12-26
:bulb: Language | Markdown Document
:page_with_curl: Repository | git@github.com:roadelou/ELSA\_Canonical\_Projection.git

### EOF
