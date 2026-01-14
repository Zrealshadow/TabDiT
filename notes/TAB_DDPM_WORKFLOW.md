# Tab-DDPM Training & Generation Workflow

## Model Architecture

**Input**: $(x_t, t, y)$
- $x_t$: Noisy data at timestep t
- $t$: Timestep (0-1000)
- $y$: Label (optional, for conditional generation)

**Output**: $\epsilon_\theta$
- Predicted **total noise** from $x_0$ to $x_t$

**Networks**: MLPDiffusion or ResNetDiffusion (tab_ddpm/modules.py)

---

## Training Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load batch: (x_0, y) from dataloader                    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 2. Sample random timestep: t ~ Uniform(0, 1000)            │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 3. Generate noise: ε ~ N(0, I)                             │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 4. Forward diffusion (direct jump):                        │
│    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε                         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 5. Model forward: ε_θ = model(x_t, t, y)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 6. Compute loss: L = ||ε - ε_θ||²                          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ 7. Backprop & update weights                               │
└─────────────────────────────────────────────────────────────┘
```

**Code**: `scripts/train.py`, `gaussian_multinomial_diffsuion.py:591-630`

---

## Generation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Initialize: x_T ~ N(0, I) (pure noise)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │ for t = T-1 to 0   │
                   └─────────┬──────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ 1. Predict noise: ε_θ = model(x_t, t, y)   │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ 2. Estimate x_0:                           │
        │    x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t       │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ 3. Compute posterior mean:                 │
        │    μ = coef1·x̂_0 + coef2·x_t               │
        └────────────────────┬────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ 4. Sample: x_{t-1} ~ N(μ, σ²)              │
        └────────────────────┬────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │ Repeat next step   │
                   └────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Output: x_0 (synthetic clean data)                          │
└─────────────────────────────────────────────────────────────┘
```

**Code**: `scripts/sample.py`, `gaussian_multinomial_diffsuion.py:927-964`

---

## Key Formulas

### Forward Process (Training)

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$$

where:
$$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i \quad \text{(cumulative product)}$$
$$\epsilon \sim \mathcal{N}(0, I)$$

### Loss Function

**Gaussian (numerical):**
$$\mathcal{L}_{\text{gauss}} = \mathbb{E}\left[||\epsilon - \epsilon_\theta||^2\right]$$

**Multinomial (categorical):**
$$\mathcal{L}_{\text{multi}} = \text{KL divergence}$$

**Total:**
$$\mathcal{L} = \mathcal{L}_{\text{gauss}} + \mathcal{L}_{\text{multi}}$$

### Reverse Process (Generation)

**Estimate $x_0$:**
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

**Sample next step:**
$$x_{t-1} \sim \mathcal{N}(\mu_t(x_t, \hat{x}_0), \sigma_t^2)$$

---

## Hybrid Diffusion Details

### Model Predictions by Feature Type

| Feature Type | Model Predicts | Loss Type |
|--------------|----------------|-----------|
| **Numerical** | Gaussian noise $\epsilon_\theta$ | MSE: $\|\|\epsilon - \epsilon_\theta\|\|^2$ |
| **Categorical** | Logits → probabilities | KL divergence |

### Categorical Feature Processing

**Per-feature softmax:**
- Softmax applied **independently** to each categorical feature
- Example: 3 features [3, 5, 2] classes → 10 logits total
  - Feature 1: softmax over 3 dimensions
  - Feature 2: softmax over 5 dimensions
  - Feature 3: softmax over 2 dimensions

### Multinomial Loss

$$\mathcal{L}_{\text{multi}} = \frac{1}{m} \sum_{i=1}^{K_{\text{total}}} \text{KL}_i$$

where $m$ = number of categorical features, $K_{\text{total}} = \sum K_i$

**Computation:**
1. KL divergence for each category dimension
2. Sum over all dimensions
3. Average by number of features

---

## Key Points

1. **Training**: Direct jump from $x_0$ to $x_t$ (closed form)
2. **Generation**: Step-by-step from $x_T$ to $x_0$ (iterative)
3. **Model output**: Always predicts **total noise $\epsilon$** from $x_0$ to $x_t$
4. **Hybrid diffusion**: Gaussian for numerical + Multinomial for categorical
5. **Steps**: $T = 1000$ timesteps (default)

---

## Code References

- Model: `tab_ddpm/modules.py:425-487`
- Diffusion: `tab_ddpm/gaussian_multinomial_diffsuion.py`
- Training: `scripts/train.py:34-44`
- Sampling: `scripts/sample.py:20-159`
- Loss: `gaussian_multinomial_diffsuion.py:591-630`
