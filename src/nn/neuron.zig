const std = @import("std");
const Prng = std.Random.DefaultPrng;
const Allocator = std.mem.Allocator;

const Value = @import("../value.zig").Value;

/// Neuron represents a single neuron in a neural network.
pub const Neuron = struct {
    w: []Value(f32),
    b: Value(f32),
    non_linear: bool = true,
    arena_allocator: std.heap.ArenaAllocator,

    /// Creates a neuron given the input values.
    pub fn init(allocator: Allocator, num_inputs: usize, non_linear: bool) Allocator.Error!Neuron {
        var arena = std.heap.ArenaAllocator.init(allocator);
        const w = try arena.allocator().alloc(Value(f32), num_inputs);
        var prng = Prng.init(@intCast(std.time.milliTimestamp()));
        for (w) |*weight| {
            weight.* = Value(f32).init(prng.random().float(f32) * 2.0 - 1.0); // [-1.0, 1.0)
        }
        return Neuron{
            .w = w,
            .b = Value(f32).init(0.0),
            .non_linear = non_linear,
            .arena_allocator = arena,
        };
    }

    /// Deinitializes the neuron and frees its resources.
    pub fn deinit(self: Neuron) void {
        self.arena_allocator.deinit();
    }

    pub fn forward(self: *Neuron, x: []*Value(f32)) Allocator.Error!Value(f32) {
        const allocator = self.arena_allocator.allocator();
        // products = [w1 * x1, w2 * x2, ..., wn * xn]
        const products = try allocator.alloc(Value(f32), self.w.len);
        for (products, 0..) |*product, i| {
            product.* = self.w[i].mul(x[i]);
        }
        // sum = w1 * x1 + w2 * x2 + ... + wn * xn
        var sum = &products[0];
        for (products[1..]) |*p| {
            const add = try allocator.create(Value(f32));
            add.* = sum.add(p);
            sum = add;
        }
        // z = sum + b
        var z = try allocator.create(Value(f32));
        z.* = sum.add(&self.b);
        return if (self.non_linear) z.relu() else z.*;
    }

    const testing = @import("std").testing;

    test init {
        const allocator = std.testing.allocator;
        for (0..5) |_| {
            const neuron = try Neuron.init(allocator, 3, true);
            defer neuron.deinit();

            for (neuron.w) |weight| {
                try testing.expect(weight.data >= -1.0 and weight.data < 1.0);
            }
            try testing.expectEqual(3, neuron.w.len);
            try testing.expectEqual(0.0, neuron.b.data);
            try testing.expectEqual(true, neuron.non_linear);
        }
    }

    test forward {
        const allocator = std.testing.allocator;
        var neuron = try Neuron.init(allocator, 3, true);
        defer neuron.deinit();

        // w is randomly generated. Reset them to fixed value for easy testing.
        var w = [_]Value(f32){
            Value(f32).init(-1.0),
            Value(f32).init(-0.25),
            Value(f32).init(0.75),
        };
        neuron.w = w[0..];

        var x1 = Value(f32).init(1.0);
        var x2 = Value(f32).init(2.0);
        var x3 = Value(f32).init(3.0);
        var x = [_]*Value(f32){
            &x1,
            &x2,
            &x3,
        };
        const output1 = try neuron.forward(x[0..]);
        try testing.expectEqual(0.75, output1.data);

        x3 = Value(f32).init(-1.0);
        const output2 = try neuron.forward(x[0..]);
        try testing.expectEqual(0.0, output2.data); // since relu is applied
    }

    test "foward without non_linear" {
        const allocator = std.testing.allocator;
        var neuron = try Neuron.init(allocator, 2, false);
        defer neuron.deinit();

        // w is randomly generated. Reset them to fixed value for easy testing.
        var w = [_]Value(f32){
            Value(f32).init(-1.0),
            Value(f32).init(-0.25),
        };
        neuron.w = w[0..];

        var x1 = Value(f32).init(1.0);
        var x2 = Value(f32).init(2.0);
        var x = [_]*Value(f32){
            &x1,
            &x2,
        };
        const output = try neuron.forward(x[0..]);
        try testing.expectEqual(-1.5, output.data);
    }
};
