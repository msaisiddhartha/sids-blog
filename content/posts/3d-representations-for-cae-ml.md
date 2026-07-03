---
title: "How Machine Learning Sees Geometry: 3D Representations for Engineering Design"
date: 2026-07-03T00:00:00-05:00
draft: false
tags: ["Machine Learning", "CAE", "Geometric Deep Learning", "Design Optimization"]
categories: ["Tech"]
math: true
ShowToc: true
TocOpen: true
---

Before a neural network can predict a part's stress, generate a lighter version of it, or decide where to put material, somebody has to hand it the geometry. And a network only eats one thing: tensors — dense, ordered grids of numbers. A 3D shape is none of that. It's a continuous piece of space with a curved skin, no natural coordinate frame, and no "first" point. So between the CAD model and the network sits a translation step, and someone has to choose how to do it.

That choice usually gets treated as plumbing. It isn't. The representation you pick quietly decides *which tasks are even possible*. A format that's perfect for reading stress off a fixed part can be useless for inventing new ones; one that's ideal for topology optimization can be hopeless as editable CAD. Pick wrong and you're not slightly slower — you're solving a different problem than the one you meant to.

This post walks the five main ways to feed 3D geometry to a model, and for each one asks the two questions that actually matter for engineering ML:

1. **How well does a network learn from it?** — is it invariant to the things that shouldn't matter, does it scale, can you take gradients through the shape?
2. **What is it actually used for?** — surrogate modeling, generative design, topology optimization, shape recognition, reconstruction.

I like to line the five up as a progression, because reading it left to right is itself the story — each step hands the network less raw data and more *structure*:

$$
\underbrace{\textbf{grids}}_{\text{voxels}} \;\rightarrow\; \underbrace{\textbf{sets}}_{\text{point clouds}} \;\rightarrow\; \underbrace{\textbf{graphs}}_{\text{meshes}} \;\rightarrow\; \underbrace{\textbf{fields}}_{\text{implicit / SDF}} \;\rightarrow\; \underbrace{\textbf{programs}}_{\text{CAD B-rep}}
$$

To keep it concrete, we'll carry one shape along the whole way — a jet-engine bracket, the kind of small structural part every method in this post has been tried on — and look at it five different ways:

![The same bracket as voxels, points, mesh, SDF, and B-rep](/sids-blog/images/cae_spine_overview.png)

Same physical part in every panel. Five completely different things to hand a network, and five different sets of tasks they're good at. Let's take them one at a time.

## What a Model Asks of a Representation

A little scaffolding first, so we can be precise about "how well does a network learn from it."

Call the true geometry $\mathcal{G} \subset \mathbb{R}^3$ — the region filled with material, wrapped in a surface $\partial\mathcal{G}$. It has infinitely many points, no ordering, no fixed size; you can't type it into a tensor. So every representation in this post is really one operation — a map that turns the shape into something a network can hold:

$$
R : \mathcal{G} \;\longmapsto\; \mathbf{X} \in \mathcal{X}.
$$

Different $R$ send the shape into different homes $\mathcal{X}$ — a voxel cube, a point set, a graph, a function. Once it's there, a model

$$
f_\theta : \mathcal{X} \;\longrightarrow\; \mathcal{Y}
$$

can map it to whatever we care about: a scalar like peak stress, a field over the surface, a class label, or even another shape. The task lives in the choice of $\mathcal{Y}$ and how we train $f_\theta$.

Whether learning goes well comes down to a handful of properties of $R$, and they're worth naming once so we can grade each representation against them.

- **Invariance.** A shape doesn't change if you rotate it or renumber the samples you took from it, so the model's answer shouldn't either: $f_\theta(R(\pi\cdot\mathcal{G})) = f_\theta(R(\mathcal{G}))$ for a relabeling $\pi$. If the representation lacks this, the network wastes capacity learning to ignore bookkeeping.
- **Scaling.** How fast do memory and compute grow as you ask for more detail? This is what caps resolution in practice.
- **Shape-differentiability.** Can gradients flow back into the geometry, $\partial f_\theta / \partial \mathbf{X}$? This is the difference between *evaluating* shapes and *optimizing* them.
- **Fidelity.** Does the representation preserve the sharp edges, thin webs, and exact surfaces an engineer actually cares about?
- **Data efficiency.** How much labeled data does a model need before it generalizes?

And the tasks those properties enable, which we'll keep returning to:

- **Surrogate modeling** — predict a simulation result (stress, drag, a flow field) from geometry, in milliseconds instead of hours.
- **Generative design** — produce new shapes, ideally novel and manufacturable ones.
- **Topology optimization** — decide where material should and shouldn't go, subject to physics.
- **Recognition & retrieval** — classify a part, segment it, spot a machining feature, find similar designs.
- **Reconstruction** — turn a scan or an image into a usable surface or CAD model.

It's worth seeing why these tasks are hard enough to throw a neural network at. Take the most common one, surrogate modeling: predict a part's peak stress from its geometry. You might hope that's a tidy function of something simple like mass — but it isn't. Plot the two for a family of brackets and the relationship is a stubborn cloud, which is precisely why we learn it instead of writing a formula:

![Peak stress versus geometry is a messy target for a surrogate](/sids-blog/images/cae_surrogate_target.png)

With that scorecard in hand, the five representations more or less sort themselves.

## Grids — Voxels

The first idea is the one that needs no new ideas: if an image is a grid of pixels, a shape is a grid of *voxels*. Drop a lattice over the part and, cell by cell, ask a yes-or-no question — is there material here?

$$
R_{\text{vox}}(\mathcal{G}) = \mathbf{V} \in \{0,1\}^{\,n\times n\times n}, \qquad
\mathbf{V}_{ijk} = \mathbb{1}\!\left[\, c_{ijk} \cap \mathcal{G} \neq \varnothing \,\right].
$$

The reason this is attractive is that *every* tool from image deep learning transfers one dimension up: 3D convolutions, pooling, translation equivariance, the works. That makes voxels the natural first home for three tasks at once. For **surrogate modeling**, a 3D-CNN reads a stress or drag estimate straight off the cube. For **generative design**, 3D-GANs [12] learn a latent space of voxelized shapes and sample new ones from it. And most naturally of all, voxels are the native language of **topology optimization**: the classic SIMP method already represents a design as a grid of material densities $\rho_{ijk}\in[0,1]$, so learning-based accelerators slot right in — CNNs that predict a converged layout in one shot instead of hundreds of solver iterations [13].

![3D-GAN voxel generator](/sids-blog/images/ref_3dgan.png)
*Generating shapes on a grid: a latent vector $\mathbf{z}$ is expanded by a stack of 3D deconvolutions into an occupancy cube — here, a chair — and sampling a new $\mathbf{z}$ yields a new shape. (Image source: Wu et al., 2016 [12])*

The wall every voxel method hits is arithmetic. Memory and compute scale like

$$
\mathcal{O}(n^3),
$$

so each doubling of resolution costs eight times as much — and most of what you pay for is empty space, because a real part is thin and branching while the grid insists on storing the void around it.

![Coarse vs fine voxelization of the bracket](/sids-blog/images/cae_voxels.png)

Refining the grid actually makes the occupied fraction *smaller*, and even a fairly fine grid smears the fillet radii where parts really fail — so voxels are simultaneously expensive and blurry. Octree and sparse convolutions (OctNet [2] and its descendants) skip the empty cells and push the ceiling higher, but they're patching a deeper mismatch: a grid spends its resolution evenly over a volume when almost everything interesting about a shape lives on its 2D skin. Which is a strong hint about what to try next.

## Sets — Point Clouds

If the information is on the surface, represent the surface and nothing else. Scatter points across the skin, keep their coordinates, throw the rest away:

$$
R_{\text{set}}(\mathcal{G}) = \mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_n\}, \qquad \mathbf{x}_i \in \mathbb{R}^3 .
$$

![Sparse vs dense point sampling of the bracket](/sids-blog/images/cae_pointcloud.png)

The subtlety that makes point clouds interesting is hiding in the braces: a set has *no order*. Sample the same part twice and you get the same points in a different sequence, and the model had better not care. PointNet [3] solved this with an almost embarrassingly simple idea — build the network out of an operation that can't tell the order changed:

$$
f_\theta(\{\mathbf{x}_1,\dots,\mathbf{x}_n\}) \;=\; \gamma\!\left( \underset{i=1,\dots,n}{\square}\; h(\mathbf{x}_i) \right).
$$

Every point goes through the same small network $h$; a permutation-blind aggregator — canonically a channel-wise $\color{teal}{\max}$ — crushes the results into one summary no matter what order they came in; and $\gamma$ turns that into an answer. That single $\color{teal}{\max}$ buys permutation invariance *for free from the architecture*, and it stops the model caring whether you fed it 3,000 points or 3,050.

![PointNet architecture](/sids-blog/images/ref_pointnet_arch.png)
*The PointNet architecture in one picture: every point runs through the same shared MLP, then a single symmetric $\max$-pool collapses them into one global feature that's blind to point order — everything downstream builds on that. (Image source: Qi et al., 2017 [3])*

That combination — invariant, and linear in the number of points rather than cubic in resolution — makes point clouds a workhorse for tasks that start from *measured* geometry. In **recognition and retrieval**, PointNet-style classifiers and segmenters are a default for labeling parts and picking out regions. In **reconstruction**, points are the raw output of a scanner, so learning to turn them into surfaces is the whole game. And for **surrogate modeling** on a fixed family of parts, a PointNet regressor is often the quickest baseline that works.

The bill comes due on connectivity. By keeping only *where* the points are, we've discarded *which* points are neighbors — and neighborhoods are where the physics lives, in the thin web or the fillet that concentrates stress. PointNet++ [4] recovers some locality by pooling over local clusters, but it only sharpens the obvious question: if the neighbors matter this much, why did we throw them away?

## Graphs — Meshes

We never had to. The mesh an engineer already built for FEA *is* a graph — vertices are nodes, element edges are edges — so the most CAE-native move is to feed the network the discretization you already own:

$$
R_{\text{graph}}(\mathcal{G}) = G = (\mathcal{V}, \mathcal{E}), \qquad
\mathcal{V} = \{\mathbf{x}_i\},\;\; \mathcal{E} = \{(i,j) : \text{$i,j$ share an element edge}\}.
$$

A graph neural network works this graph like gossip works a room. Each node carries a feature vector $h_i$ and updates itself by listening to its neighbors — one round of message passing:

$$
h_i' \;=\; \phi\!\left(\, h_i,\;\; \underset{j \in N(i)}{\color{teal}{\bigoplus}}\; \psi(h_i, h_j) \,\right).
$$

Neighbors send messages, the permutation-blind $\color{teal}{\bigoplus}$ pools them, and $\phi$ folds them back in; stack $k$ rounds and information travels $k$ hops, all without leaving the surface.

![A vertex, its 1-ring neighborhood, and one message-passing step](/sids-blog/images/cae_graph.png)

Because locality and grain are baked into the edges, meshes are the premier representation for the one task that most needs them: **surrogate modeling and learned simulation** of physics. MeshGraphNets [5] famously learn to *run the simulation itself* by passing messages on the mesh, and a whole family of GNN surrogates read stress or drag directly off the discretized part — respecting exactly the anisotropy the physics has (a long web behaves differently along its length than across it).

![MeshGraphNets learned simulator](/sids-blog/images/ref_meshgraphnets.png)
*MeshGraphNets in action: the simulation mesh becomes a graph, and an encode–process–decode network learns a single time-step update by message passing, then rolls it forward. The same machinery underlies GNN stress and drag surrogates. (Image source: Pfaff et al., 2021 [5])*

The catch is that the representation is *one particular mesh*. Remesh the part a little finer and it becomes a different graph, so a model can quietly start keying off meshing choices instead of geometry, and predictions wobble when the mesh does. Generation is awkward for the same reason — producing a valid, well-conditioned mesh from scratch is much harder than reading one — which is why mesh methods dominate prediction but rarely generation. And you still needed a mesh in the first place, the most human-intensive step in the whole pipeline. Which raises a genuinely radical thought: what if we didn't discretize at all?

## Fields — Implicit and Signed-Distance Representations

Instead of storing samples *of* the shape, store a function that *is* the shape — a rule that answers, for any point in space, how it relates to the part. Two flavors dominate: an **occupancy field** answering the blunt question, and a **signed-distance field (SDF)** answering a richer one:

$$
o(\mathbf{x}) = \mathbb{1}\!\left[\mathbf{x} \in \mathcal{G}\right],
\qquad\qquad
s(\mathbf{x}) = \pm\, \min_{\mathbf{p}\in\partial\mathcal{G}} \lVert \mathbf{x} - \mathbf{p}\rVert .
$$

The surface is now stored nowhere — it's *implied*, the shoreline where $s = 0$ between inside ($s<0$) and out ($s>0$). Slice through the part and you can see the field: cold in the metal, warm in the air, the outline emerging where they meet.

![Signed-distance field on a slice through the bracket](/sids-blog/images/cae_sdf.png)

![The signed-distance concept from DeepSDF](/sids-blog/images/ref_deepsdf.png)
*The same idea, drawn on a rabbit: every point in space stores its signed distance to the surface — negative inside, positive outside (a, b) — and the shape itself is just the zero level set, rendered in (c). (Image source: Park et al., 2019 [6])*

The neural version, DeepSDF [6] (and Occupancy Networks [7]), learns that function with an MLP and adds one inspired twist — a per-shape **latent code** $\mathbf{z}$, so a single network is a whole *family* of shapes:

$$
f_\theta(\mathbf{x}, \mathbf{z}) \approx s_{\mathbf{z}}(\mathbf{x}).
$$

Training makes predicted distance match the truth, usually with an extra term enforcing the one law a true distance field obeys — take a step, and your distance changes by exactly that step, i.e. the gradient has unit norm everywhere (the Eikonal condition [8]):

$$
\mathcal{L}(\theta) = \sum_{\mathbf{x}} \big| f_\theta(\mathbf{x},\mathbf{z}) - s(\mathbf{x}) \big|
\;+\; \lambda\, \mathbb{E}_{\mathbf{x}} \big( \lVert \color{teal}{\nabla_{\mathbf{x}} f_\theta(\mathbf{x},\mathbf{z})} \rVert - 1 \big)^2 .
$$

This is where the task list really opens up, because two properties combine. First, the field is resolution-free — query it anywhere, extract a surface as fine as you like with no cubic tensor. Second, that latent code is a smooth, *differentiable* dial for the shape. Together they make fields the engine of modern **generative design**: learn a distribution over $\mathbf{z}$ and you can sample new shapes, which is exactly what SDF- and occupancy-based generators (and, lately, diffusion models over these fields) do. They're just as central to **reconstruction** — recovering a watertight surface from a sparse scan is naturally posed as fitting an occupancy or distance field.

And they change the character of **shape and topology optimization**. Freeze the weights, treat $\mathbf{z}$ as unknown, and "make this lighter without breaking the stress limit" becomes plain gradient descent,

$$
\mathbf{z}^\star = \arg\min_{\mathbf{z}} \; \big[\, \text{mass}(\mathbf{z}) + \beta\,\max(0,\; \sigma_{\max}(\mathbf{z}) - \sigma_{\text{allow}}) \,\big],
$$

the same frozen-model-optimize-the-input trick from my [earlier post](/sids-blog/posts/parametric-optimization-neural-networks/), only now the input is a whole 3D shape. Even classical **topology optimization** has moved this way: methods like TOuNN [15] replace the voxel density grid with a neural field, so the design is mesh-independent and optimized by backprop, and level-set formulations are implicit fields by definition — holes appear and merge freely just by letting $s$ change sign, with none of the topology headaches a fixed mesh suffers.

![Sweeping the level set of a bracket's signed-distance field](/sids-blog/images/cae_topology_sweep.gif)
*The same idea, live on SimJEB bracket #148. Every frame is the zero set of **one** signed-distance field; sliding the level $s$ from a small positive offset (material added, holes shrink) down through zero into mild erosion (holes grow) opens and merges the lightening and bolt holes — the surface staying watertight the whole way, no remeshing. A fixed mesh would have to re-tessellate, and change its element count and connectivity, at every one of these steps.*

Nothing's free. The surface being implicit means that to actually mesh-and-solve an optimized part you have to extract the $s=0$ level set and remesh it — the discretization we escaped greets us again at the door — and a poorly fit field shows up as a lumpy skin, which for a stress calculation is simply the wrong part.

## Programs — CAD Boundary Representations

Notice that every representation so far is a *discretization*. But no engineer designed the part as a point cloud — they drew it in CAD, as a **boundary representation**: exact trimmed surfaces (planes, cylinders, NURBS patches) stitched by a map of which face meets which. If you want the network to speak the language of the tools that will cut the metal, this is it.

![B-rep as a face-adjacency graph](/sids-blog/images/cae_brep.png)

A B-rep is two-storied — exact geometry per face, plus the topology wiring them together:

$$
R_{\text{brep}}(\mathcal{G}) = \big(\, \{S_f\}_{f\in\mathcal{F}},\; T \,\big),
$$

with each $S_f : [0,1]^2 \to \mathbb{R}^3$ a parametric surface patch and $T$ the face-edge adjacency graph. Two tasks fit it especially well. In **recognition** — part classification, and machining-feature detection for manufacturing — models like UV-Net [9] and BRepNet [14] learn directly on the faces and topology, which is where the manufacturing-relevant information actually lives. And in **generative design**, DeepCAD [10] models the *construction program* itself — the sequence of sketches and extrudes a designer would type — so the output isn't a mesh but editable CAD you can open, tweak, and send to a machine.

![CAD models generated by DeepCAD](/sids-blog/images/ref_deepcad.png)
*Shapes invented by DeepCAD, each emitted as a CAD construction sequence rather than a mesh — so every one is an editable, parametric solid a designer could open and modify. (Image source: Wu et al., 2021 [10])*

That's the real prize of programs: exact geometry, and outputs that are genuinely *parts*, closing the loop back to the engineer's toolchain. The reason they aren't simply the winner is data. A B-rep is a heterogeneous, variable-length tangle of trimmed surfaces, harder for a network to digest than a grid or point cloud, and clean labeled CAD is scarce — orders of magnitude scarcer than the images a vision model trains on. So CAD-native learning leans hard on augmentation and synthetic data, and tasks like B-rep surrogate modeling are still comparatively immature.

Step back, and the progression — grids, sets, graphs, fields, programs — is really one idea unfolding: give the network less undifferentiated tensor and more of the *structure* an engineer already knows is there, from a dense cube that doesn't know where the metal is, to a program that remembers the part was built from sketches and extrudes. The tasks track that structure — the blunt early representations were enough to *recognize* and *predict*, but it took differentiable fields and CAD-native programs to start *creating* and *optimizing*. And that's the live frontier, because the most valuable thing you can do with a shape isn't to label it — it's to change it for the better, and that takes a representation the optimizer can actually move.

---

## References

[1] Wu et al. ["3D ShapeNets: A Deep Representation for Volumetric Shapes."](https://arxiv.org/abs/1406.5670) CVPR 2015.

[2] Riegler et al. ["OctNet: Learning Deep 3D Representations at High Resolutions."](https://arxiv.org/abs/1611.05009) CVPR 2017.

[3] Qi et al. ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation."](https://arxiv.org/abs/1612.00593) CVPR 2017.

[4] Qi et al. ["PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space."](https://arxiv.org/abs/1706.02413) NeurIPS 2017.

[5] Pfaff et al. ["Learning Mesh-Based Simulation with Graph Networks."](https://arxiv.org/abs/2010.03409) ICLR 2021.

[6] Park et al. ["DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation."](https://arxiv.org/abs/1901.05103) CVPR 2019.

[7] Mescheder et al. ["Occupancy Networks: Learning 3D Reconstruction in Function Space."](https://arxiv.org/abs/1812.03828) CVPR 2019.

[8] Gropp et al. ["Implicit Geometric Regularization for Learning Shapes."](https://arxiv.org/abs/2002.10099) ICML 2020.

[9] Jayaraman et al. ["UV-Net: Learning from Boundary Representations."](https://arxiv.org/abs/2006.10211) CVPR 2021.

[10] Wu et al. ["DeepCAD: A Deep Generative Network for Computer-Aided Design Models."](https://arxiv.org/abs/2105.09492) ICCV 2021.

[11] Whalen et al. ["SimJEB: Simulated Jet Engine Bracket Dataset."](https://simjeb.github.io/) Computer Graphics Forum 2021. *(source of the bracket used for the figures)*

[12] Wu et al. ["Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN)."](https://arxiv.org/abs/1610.07584) NeurIPS 2016.

[13] Sosnovik & Oseledets. ["Neural Networks for Topology Optimization."](https://arxiv.org/abs/1709.09578) 2017.

[14] Lambourne et al. ["BRepNet: A Topological Message Passing System for Solid Models."](https://arxiv.org/abs/2104.00706) CVPR 2021.

[15] Chandrasekhar & Suresh. ["TOuNN: Topology Optimization using Neural Networks."](https://doi.org/10.1007/s00158-020-02748-4) Structural and Multidisciplinary Optimization, 2021.
