# micrograd.zig
A tiny scalar-valued autograd engine and a neural net library in Zig. It is inspired by
the micrograd project from Andrej Karpathy.

* Youtube video: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=YMjTVeoWDI8VKSZf).
* micrgrad on [GitHub](https://github.com/karpathy/micrograd).

I wrote it in Zig for education purpose, and for fun of course. :)

### Examples
Examples can be found in the `examples` folder.

* expression.zig: a backpropagation on an expression.
* tiny_dataset: a neural network on a tiny dataset.

To run them:
```shell
% zig build run -Dexample=expression
% zig build run -Dexample=tiny_dataset
```

### Run tests
```shell
% zig build test --summary all
```

### License
MIT
