const std = @import("std");
const Op = @import("operator.zig").Op;

/// A value stores a scalar data and its gradient.
pub fn Value(comptime T: type) type {
    return struct {
        /// The scalar data.
        data: T,
        /// The operation that produced this value, if any.
        op: ?Op(T) = null,
        /// The gradient of this value, used for backpropagation.
        grad: f32 = 0.0,

        /// Creates a value with the given type and value.
        pub fn init(value: T) Value(T) {
            return switch (T) {
                f32 => .{
                    .data = value,
                },
                i32 => .{
                    .data = value,
                },
                else => @compileError("Unsupported data type: " ++ @typeName(T)),
            };
        }

        /// Adds this value with another value of the same type.
        pub fn add(self: *Value(T), other: *Value(T)) Value(T) {
            const op = Op(T){
                .add = .{
                    .left = self,
                    .right = other,
                },
            };
            return op.add.apply();
        }

        /// Subtracts another value from this value of the same type.
        pub fn sub(self: *Value(T), other: *Value(T)) Value(T) {
            const op = Op(T){
                .sub = .{
                    .left = self,
                    .right = other,
                },
            };
            return op.sub.apply();
        }

        /// Multiplies this value with another value of the same type.
        pub fn mul(self: *Value(T), other: *Value(T)) Value(T) {
            const op = Op(T){
                .mul = .{
                    .left = self,
                    .right = other,
                },
            };
            return op.mul.apply();
        }

        /// Divides this value by another value of the same type.
        pub fn div(self: *Value(T), other: *Value(T)) Value(T) {
            const op = Op(T){
                .div = .{
                    .left = self,
                    .right = other,
                },
            };
            return op.div.apply();
        }

        /// Applies the ReLU activation function to this value.
        pub fn relu(self: *Value(T)) Value(T) {
            const op = Op(T){
                .relu = .{
                    .value = self,
                },
            };
            return op.relu.apply();
        }

        /// Raises this value to the power of the given exponent.
        pub fn pow(self: *Value(T), exponent: T) Value(T) {
            const op = Op(T){
                .pow = .{
                    .base = self,
                    .exponent = exponent,
                },
            };
            return op.pow.apply();
        }

        /// Performs backpropagation to compute gradients.
        pub fn backward(self: *Value(T)) std.mem.Allocator.Error!void {
            var gpa = std.heap.GeneralPurposeAllocator(.{}){};
            defer _ = gpa.deinit();
            const allocator = gpa.allocator();

            // run topological sort to get the order of backpropagation
            var visited = std.AutoHashMap(*Value(T), bool).init(allocator);
            defer visited.deinit();
            var stack = std.ArrayList(*Value(T)).init(allocator);
            defer stack.deinit();
            try self.dfs(&visited, &stack);

            self.grad = 1.0;
            var len = stack.items.len;
            while (len > 0) : (len -= 1) {
                const current = stack.items[len - 1];
                if (current.op) |op| {
                    op.backward(current);
                }
            }
        }

        fn dfs(self: *Value(T), visited: *std.AutoHashMap(*Value(T), bool), stack: *std.ArrayList(*Value(T))) std.mem.Allocator.Error!void {
            if (visited.get(self) orelse false) {
                return;
            }
            try visited.put(self, true);
            if (self.op) |op| {
                switch (op) {
                    .add => |add_op| {
                        try add_op.left.dfs(visited, stack);
                        try add_op.right.dfs(visited, stack);
                    },
                    .sub => |sub_op| {
                        try sub_op.left.dfs(visited, stack);
                        try sub_op.right.dfs(visited, stack);
                    },
                    .mul => |mul_op| {
                        try mul_op.left.dfs(visited, stack);
                        try mul_op.right.dfs(visited, stack);
                    },
                    .div => |div_op| {
                        try div_op.left.dfs(visited, stack);
                        try div_op.right.dfs(visited, stack);
                    },
                    .relu => |relu_op| {
                        try relu_op.value.dfs(visited, stack);
                    },
                    .pow => |pow_op| {
                        try pow_op.base.dfs(visited, stack);
                    },
                }
                try stack.append(self);
            }
        }
    };
}

const testing = @import("std").testing;

test Value {
    const value_f32 = Value(f32).init(10.0);
    const value_i32 = Value(i32).init(10);

    try testing.expectEqual(10.0, value_f32.data);
    try testing.expectEqual(null, value_f32.op);
    try testing.expectEqual(0.0, value_f32.grad);
    try testing.expectEqual(10, value_i32.data);
    try testing.expectEqual(null, value_i32.op);
    try testing.expectEqual(0.0, value_i32.grad);
}

test "value add" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const result_f32 = value_f32_1.add(&value_f32_2);
    try testing.expectEqual(15.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value sub" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const result_f32 = value_f32_1.sub(&value_f32_2);
    try testing.expectEqual(5.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value mul" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const result_f32 = value_f32_1.mul(&value_f32_2);
    try testing.expectEqual(50.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value div" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const result_f32 = value_f32_1.div(&value_f32_2);
    try testing.expectEqual(2.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value relu" {
    var value_f32 = Value(f32).init(-10.0);
    const result_f32 = value_f32.relu();
    try testing.expectEqual(0.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value pow" {
    var value_f32 = Value(f32).init(2.0);
    const result_f32 = value_f32.pow(3.0);
    try testing.expectEqual(8.0, result_f32.data);
    try testing.expect(result_f32.op != null);
}

test "value backpropagation" {
    // This example is taken from Andrej Karpathy's building micrograd video:
    // https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s
    // Expression: l = (a * b + c) * f
    const Vf32 = Value(f32);
    var a = Vf32.init(2.0);
    var b = Vf32.init(-3.0);
    var e = a.mul(&b);
    var c = Vf32.init(10.0);
    var d = e.add(&c);
    var f = Vf32.init(-2.0);
    var l = d.mul(&f);
    try testing.expectEqual(-8.0, l.data);

    try l.backward();
    try testing.expectEqual(1.0, l.grad);
    try testing.expectEqual(4.0, f.grad);
    try testing.expectEqual(-2.0, d.grad);
    try testing.expectEqual(-2.0, e.grad);
    try testing.expectEqual(-2.0, c.grad);
    try testing.expectEqual(-4.0, b.grad);
    try testing.expectEqual(6.0, a.grad);
}
