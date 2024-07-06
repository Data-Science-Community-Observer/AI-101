

### ***Day 2: Tensors in PyTorch***

- Introduction to Tensors.

- Tensor operations: creation, manipulation, and basic
  operations.

- Resources: [PyTorch Tensors](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

# 

#### Introduction to Tensors

Tensors are the fundamental building blocks of PyTorch. They are similar to NumPy arrays but with additional features that support the operations required for deep learning. Tensors can run on GPUs, which accelerates computations, making them suitable for high-performance computing tasks.

##### Key Features of Tensors:

- **Multi-dimensional arrays** similar to NumPy arrays.
- **Support for GPU computation** to leverage parallelism.
- **Autograd functionality** for automatic differentiation.

#### Tensor Operations: Creation, Manipulation, and Basic Operations

##### Creating Tensors

Tensors can be created in various ways, such as directly from data, from NumPy arrays, or by using built-in functions.

**Creating a tensor from data:**

```python
import torch

# Creating a tensor from a list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("Tensor from data:\n", x_data)
```

**Creating a tensor from a NumPy array:**

```python
import numpy as np

# Creating a NumPy array
np_array = np.array(data)
# Creating a tensor from a NumPy array
x_np = torch.from_numpy(np_array)
print("Tensor from NumPy array:\n", x_np)
```

**Creating a tensor using built-in functions:**

```python
# Creating a tensor of ones
x_ones = torch.ones(2, 3)
print("Tensor of ones:\n", x_ones)

# Creating a tensor of random values
x_rand = torch.rand(2, 3)
print("Tensor of random values:\n", x_rand)
```

##### Manipulating Tensors

Tensors can be manipulated using various operations such as indexing, slicing, reshaping, and joining.

**Indexing and slicing:**

```python
# Creating a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original tensor:\n", x)

# Indexing
print("First row:\n", x[0])
print("First column:\n", x[:, 0])
print("Last column:\n", x[:, -1])

# Slicing
print("Second row, second and third columns:\n", x[1, 1:3])
```

**Reshaping tensors:**

```python
# Reshaping a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original shape:", x.shape)

# Reshape to a 3x2 tensor
x_reshaped = x.view(3, 2)
print("Reshaped tensor:\n", x_reshaped)
```

**Joining tensors:**

```python
# Joining tensors along a dimension
x1 = torch.tensor([[1, 2], [3, 4]])
x2 = torch.tensor([[5, 6], [7, 8]])

# Concatenate along the first dimension (rows)
x_cat = torch.cat([x1, x2], dim=0)
print("Concatenated tensor along rows:\n", x_cat)

# Stack along the second dimension (columns)
x_stack = torch.stack([x1, x2], dim=1)
print("Stacked tensor along columns:\n", x_stack)
```

##### Basic Tensor Operations

Tensors support various arithmetic and linear algebra operations. Here are some examples:

**Arithmetic operations:**

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = x + y
print("Addition:\n", z)

# Subtraction
z = x - y
print("Subtraction:\n", z)

# Multiplication
z = x * y
print("Multiplication:\n", z)

# Division
z = x / y
print("Division:\n", z)
```

**Matrix multiplication:**

```python
# Matrix multiplication
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# Using the matmul function
z = torch.matmul(x, y)
print("Matrix multiplication result:\n", z)
```

**Element-wise operations:**

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# Element-wise multiplication
z = x * y
print("Element-wise multiplication result:\n", z)

# Element-wise power
z = x ** 2
print("Element-wise power result:\n", z)
```

**Aggregating operations:**

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Sum
sum_x = torch.sum(x)
print("Sum of all elements:\n", sum_x)

# Mean
mean_x = torch.mean(x.float())
print("Mean of all elements:\n", mean_x)

# Maximum
max_x = torch.max(x)
print("Maximum element:\n", max_x)

# Minimum
min_x = torch.min(x)
print("Minimum element:\n", min_x)
```

---

### Resources

- [PyTorch Tensors Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

These examples and explanations provide a comprehensive understanding of tensor operations in PyTorch, setting a solid foundation for more advanced topics.
