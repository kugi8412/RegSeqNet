#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm, cauchy, uniform

# Params
np.random.seed(1)
n = 5000000

# Data
x_norm = np.random.normal(0, 1, n)
x_cauchy = np.random.standard_cauchy(n)
a, b = -np.sqrt(4), np.sqrt(4)
x_uniform = np.random.uniform(a, b, n)

# Curtoise
def classify_kurtosis(excess):
    if excess < 0:
        return "platykurtyczny (płaski, krótkie ogony)"
    elif abs(excess) < 1e-2:
        return "mezokurtyczny (normalny, umiarkowane ogony)"
    else:
        return "leptokurtyczny (spiczasty, ciężkie ogony)"

k_norm = kurtosis(x_norm, fisher=True, bias=False)
k_cauchy = kurtosis(x_cauchy, fisher=True, bias=False)
k_uniform = kurtosis(x_uniform, fisher=True, bias=False)

desc_norm = classify_kurtosis(k_norm)
desc_cauchy = classify_kurtosis(k_cauchy)
desc_uniform = classify_kurtosis(k_uniform)

# PDF
xs = np.linspace(-5, 5, 1000)
pdf_norm = norm.pdf(xs, 0, 1)
pdf_cauchy = cauchy.pdf(xs, 0, 1)
pdf_uniform = uniform.pdf(xs, loc=a, scale=b - a)

plt.figure(figsize=(10, 6))
plt.plot(xs, pdf_norm, label=f"Normalny (kurtoza = 0)", color="blue")
plt.plot(xs, pdf_cauchy, label=f"Cauchy (kurtoza = +∞)", color="red")
plt.plot(xs, pdf_uniform, label=f"Jednostajny (kurtoza = 1.8)", color="green")

plt.title("Porównanie kurtoz wybranych funkcji gęstości prawdopodobieństwa", fontsize=14, weight="bold")
plt.xlabel("x")
plt.ylabel("gęstość")
plt.legend()
plt.grid(alpha=0.2)

# Description
text = (
    f"Normalny: kurtoza={k_norm:.2f} → {desc_norm}\n"
    f"Cauchy: kurtoza≈{k_cauchy:.1f} → {desc_cauchy} (momenty formalnie nie istnieją)\n"
    f"Jednostajny: kurtoza={k_uniform:.2f} → {desc_uniform}"
)
plt.figtext(0.5, -0.05, text, ha="center", va="top", fontsize=11)

plt.tight_layout()
plt.show()
plt.savefig("kurtoza.png", dpi=300)
