# Alzheimer's model

This is a Python implementation of a solver to a diffusion-reaction partial
differential equation that intends to model the propagation of an infection
throughout the brain. The infection is modeled as a point source.

This project is part of the
[Mathematics in Medicine Program](https://www.houstonmethodist.org/math-in-medicine/)
at the [Houston Methodist Research Institute](https://www.houstonmethodist.org/research/).

## Equation

The model has the form: <br>
<img src='./math/latex_equation.png' alt='PDE' height='30px' width='200px' /> <br>
where &rho;(x,t) denotes the density of the infection in the tissue.
In 1D, for a domain of length L, the analytical solution for the given equation is:<br>
<img src='./math/latex_solution.png' alt='Solution to PDE' />

## Videos

[<img src='./solution/1D/solution_0.png' height='320px' width='240px'/>](https://jrr3.github.io/Alzheimer/)


## Usage

To generate the simulation execute the command:

```bash
python3 driver.py
```

To create a movie of the simulation uncomment the line

```python
obj.create_movie()
```

