# Light-Switch

**One set of weights, many models: switch parts of a sparse LLM on and off.**

> Status: research ideas and hypotheses. Nothing here has been tested yet.

## Background

In a weight-sparse LLM (for example, a model pruned with SparseGPT or Wanda), most weights are exactly 0. A binary mask $M$ marks which weights are kept:

$$W_{\text{sparse}} = M \odot W, \qquad M_{ij} \in \{0, 1\}$$

where $\odot$ is element-wise multiplication.

The zero positions still sit inside the weight matrix, but they do nothing. This project asks: **can those empty slots hold something useful?**

> "Sparse" here means *weight* sparsity: pruned weights are exactly 0. It is not Mixture-of-Experts sparsity, where all weights are nonzero and only some experts run for each token.

---

## Idea 1: Train the reverse mask to get a dense model with an on/off switch

1. Start from a sparse model $M \odot W$.
2. Take the **reverse (complement) mask** $\bar{M} = 1 - M$, which covers the positions that are currently 0.
3. **Freeze** the sparse weights and train **only** the complement positions with SFT or RL. Call the new weights $\Delta$. They live only where $\bar{M} = 1$.
4. The result is one dense weight matrix that holds two models:

| Switch | Weights used                          | Model                               |
|--------|---------------------------------------|-------------------------------------|
| OFF    | $M \odot W$                           | the original sparse model, unchanged |
| ON     | $M \odot W + \bar{M} \odot \Delta$    | a dense model                       |

Because the sparse part is frozen, switching OFF gives back exactly the original sparse model, so nothing is forgotten. Switching ON adds whatever the complement learned.

A **router** decides when the switch is on. It could decide:
- per request (easy prompt → sparse, hard prompt → dense),
- per token,
- per layer or per weight matrix (turn on only some layers).

**Hypothesis 1:** Training only the complement positions can turn a sparse model into a stronger dense model. The original sparse model stays inside the same weights, ready to use at no extra memory cost.

**Open questions**
- Should $\Delta$ start at zero, or at the original values that were pruned away?
- How much does dense mode gain over sparse mode? How does it compare to the original dense model before pruning?
- Can a learned router choose sparse or dense well enough to save compute without losing quality?
- With per-token routing, the KV cache for earlier tokens may come from the other mode. Does that break anything?

---

## Idea 2: Orthogonal sparsity, several models in one weight

Instead of one mask, find several masks $M_1, M_2, \dots, M_k$ that **do not overlap**:

$$M_i \odot M_j = 0 \quad \text{for } i \neq j$$

Each mask picks a different set of positions in the same weight matrix $W$, so each $M_i \odot W$ is its own model. Their nonzero positions never overlap, which makes the masked matrices orthogonal ($\langle M_i \odot W,\ M_j \odot W \rangle_F = 0$). Training one sub-model never touches another.

With $k$ disjoint masks, **one weight tensor stores $k$ models**:

```
W  =  M1⊙W  +  M2⊙W  +  M3⊙W  +  ...
      model 1  model 2  model 3
```

**Variants**
- **Fully disjoint masks:** $k$ independent models, such as one for code, one for math and one for chat.
- **Partly shared masks (mask only part of the sparsity):** every model shares a common core mask $C$ and adds its own private part, $M_i = C + P_i$, where the $P_i$ are disjoint from each other and from $C$. Shared knowledge lives in $C$ and specialist knowledge lives in $P_i$.
- **Combining masks:** turning on several masks at once (e.g. $M_1 + M_2$) gives a bigger model. Turning on all of them gives the full dense model. **Idea 1 is the special case $k = 2$.**

The router now chooses which mask or masks are on for each input. Recording which model owns each weight takes a small index: 1 bit per weight for 2 models, and about $\lceil \log_2 k \rceil$ bits for $k$ models.

**Hypothesis 2:** One dense-sized weight tensor can be split into orthogonal sparse masks, each trained into a useful model. That gives several models, and their combinations, for the memory cost of one.

**Open questions**
- How do we find good orthogonal masks? One option is to repeat prune → train → prune the remaining slots → train. Another is to learn the masks directly.
- How sparse can each sub-model get before it stops being useful?
- Do combined masks ($M_1 + M_2$) work without extra training, or do they need joint training?
- How should the router pick the mask for a given input?

---

## Practical note: 2:4 sparsity

NVIDIA GPUs from the Ampere generation onward can accelerate **2:4 sparsity**, where exactly 2 of every 4 consecutive weights are nonzero. **The complement of a 2:4 mask is also a 2:4 mask.** A weight matrix therefore splits exactly into two 2:4 halves, and both halves can run on sparse hardware. That makes 2:4 a natural first setting for both ideas.

Unstructured sparsity usually gives little real speedup on GPUs, even when it reaches high sparsity.

---

## First experiments

1. Take a small open LLM (about 0.5–1B parameters) and prune it to 50% with Wanda or SparseGPT, using either a 2:4 or an unstructured pattern.
2. **Idea 1:** freeze the kept weights and run SFT only on the complement. Compare the original dense model, the sparse model, and sparse + complement (dense mode).
3. **Idea 2:** split the weights into two disjoint 2:4 masks and train each on a different domain. Evaluate each mask alone and both together.
4. Add a simple per-request router and measure quality against compute.

Minimal sketch of the Idea 1 switch for one weight matrix:

```python
# W:    frozen sparse weight (zeros where mask == False)
# mask: bool tensor, True where the sparse model keeps a weight
delta = torch.nn.Parameter(torch.zeros_like(W))  # trainable; only complement slots are used

def effective_weight(switch_on: bool):
    if not switch_on:
        return W                        # sparse model (switch OFF)
    return W + (~mask) * delta          # dense model  (switch ON)
```

---

## Related work

Earlier work close to these ideas:

- **Lottery Ticket Hypothesis** (Frankle & Carbin, 2019): sparse subnetworks inside dense networks can train well on their own.
- **SparseGPT** (Frantar & Alistarh, 2023) and **Wanda** (Sun et al., 2023): one-shot pruning methods for LLMs.
- **PackNet** (Mallya & Lazebnik, 2018): prunes a network, then trains the freed weights on a new task, packing several tasks into one network. This is very close to Ideas 1 and 2.
- **Piggyback** (Mallya et al., 2018) and **Supermasks in Superposition** (Wortsman et al., 2020): learn task-specific masks over one fixed set of weights.
- **Slimmable Networks** (Yu et al., 2019) and **MatFormer** (Devvrit et al., 2023): a single model that can run at several sizes.

Light-Switch differs in three ways. It targets LLMs, post-trains the free slots with SFT/RL, and uses a router that switches sub-models on and off at inference time.
