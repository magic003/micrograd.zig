const std = @import("std");
const Value = @import("value.zig").Value;

/// The operator union defines various operations that can be performed
/// on `Value` types.
pub fn Op(comptime T: type) type {
    return union(enum) {
        add: Add,
        sub: Sub,
        mul: Mul,
        div: Div,
        relu: Relu,
        pow: Pow,

        pub fn backward(self: *const Op(T), out: *const Value(T)) void {
            switch (self.*) {
                .add => |op| op.backward(out),
                .sub => |op| op.backward(out),
                .mul => |op| op.backward(out),
                .div => |op| op.backward(out),
                .relu => |op| op.backward(out),
                .pow => |op| op.backward(out),
            }
        }

        fn toFloat(value: T) f32 {
            return switch (T) {
                f32 => value,
                i32 => @floatFromInt(value),
                else => @compileError("Unsupported data type: " ++ @typeName(T)),
            };
        }

        const Add = struct {
            left: *Value(T),
            right: *Value(T),

            pub fn apply(self: Add) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = self.left.data + self.right.data,
                        .op = .{ .add = self },
                    },
                    i32 => .{
                        .data = self.left.data + self.right.data,
                        .op = .{ .add = self },
                    },
                    else => unreachable,
                };
            }

            pub fn backward(self: Add, out: *const Value(T)) void {
                self.left.grad += out.grad;
                self.right.grad += out.grad;
            }
        };

        const Sub = struct {
            left: *Value(T),
            right: *Value(T),

            pub fn apply(self: Sub) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = self.left.data - self.right.data,
                        .op = .{ .sub = self },
                    },
                    i32 => .{
                        .data = self.left.data - self.right.data,
                        .op = .{ .sub = self },
                    },
                    else => unreachable,
                };
            }

            pub fn backward(self: Sub, out: *const Value(T)) void {
                self.left.grad += out.grad;
                self.right.grad -= out.grad;
            }
        };

        const Mul = struct {
            left: *Value(T),
            right: *Value(T),

            pub fn apply(self: Mul) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = self.left.data * self.right.data,
                        .op = .{ .mul = self },
                    },
                    i32 => .{
                        .data = self.left.data * self.right.data,
                        .op = .{ .mul = self },
                    },
                    else => unreachable,
                };
            }

            pub fn backward(self: Mul, out: *const Value(T)) void {
                self.left.grad += self.right.data * out.grad;
                self.right.grad += self.left.data * out.grad;
            }
        };

        const Div = struct {
            left: *Value(T),
            right: *Value(T),

            pub fn apply(self: Div) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = self.left.data / self.right.data,
                        .op = .{ .div = self },
                    },
                    i32 => .{
                        .data = @divTrunc(self.left.data, self.right.data),
                        .op = .{ .div = self },
                    },
                    else => unreachable,
                };
            }

            pub fn backward(self: Div, out: *const Value(T)) void {
                const left: f32 = toFloat(self.left.data);
                const right: f32 = toFloat(self.right.data);
                self.left.grad += out.grad / right;
                self.right.grad += -1.0 * (left / (right * right)) * out.grad;
            }
        };

        const Relu = struct {
            value: *Value(T),

            pub fn apply(self: Relu) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = if (self.value.data < 0.0) 0.0 else self.value.data,
                        .op = .{ .relu = self },
                    },
                    i32 => .{
                        .data = if (self.value.data < 0) 0 else self.value.data,
                        .op = .{ .relu = self },
                    },
                    else => unreachable,
                };
            }

            pub fn backward(self: Relu, out: *const Value(T)) void {
                if (self.value.data > 0) {
                    self.value.grad += out.grad;
                }
            }
        };

        const Pow = struct {
            base: *Value(T),
            exponent: T,

            pub fn apply(self: Pow) Value(T) {
                return switch (T) {
                    f32 => .{
                        .data = std.math.pow(f32, self.base.data, self.exponent),
                        .op = .{ .pow = self },
                    },
                    i32 => .{
                        .data = std.math.powi(i32, self.base.data, self.exponent) catch unreachable,
                        .op = .{ .pow = self },
                    },
                    else => @compileError("Unsupported data type: " ++ @typeName(T)),
                };
            }

            pub fn backward(self: Pow, out: *const Value(T)) void {
                const base = toFloat(self.base.data);
                const exponent = toFloat(self.exponent);
                self.base.grad += exponent * std.math.pow(f32, base, exponent - 1) * out.grad;
            }
        };
    };
}

const testing = @import("std").testing;

test "add apply" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op1 = Op(f32).Add{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    const result_f32 = op1.apply();
    try testing.expectEqual(15.0, result_f32.data);
    try testing.expectEqual(op1, result_f32.op.?.add);

    var value_i32_1 = Value(i32).init(10);
    var value_i32_2 = Value(i32).init(5);
    const op2 = Op(i32).Add{
        .left = &value_i32_1,
        .right = &value_i32_2,
    };
    const result_i32 = op2.apply();
    try testing.expectEqual(15, result_i32.data);
    try testing.expectEqual(op2, result_i32.op.?.add);
}

test "add backward" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op = Op(f32).Add{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    var out = op.apply();
    out.grad = 3.0;
    op.backward(&out);
    try testing.expectEqual(3.0, value_f32_1.grad);
    try testing.expectEqual(3.0, value_f32_2.grad);
}

test "sub apply" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op1 = Op(f32).Sub{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    const result_f32 = op1.apply();
    try testing.expectEqual(5.0, result_f32.data);
    try testing.expectEqual(op1, result_f32.op.?.sub);

    var value_i32_1 = Value(i32).init(10);
    var value_i32_2 = Value(i32).init(5);
    const op2 = Op(i32).Sub{
        .left = &value_i32_1,
        .right = &value_i32_2,
    };
    const result_i32 = op2.apply();
    try testing.expectEqual(5, result_i32.data);
    try testing.expectEqual(op2, result_i32.op.?.sub);
}

test "sub backward" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op = Op(f32).Sub{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    var out = op.apply();
    out.grad = 3.0;
    op.backward(&out);
    try testing.expectEqual(3.0, value_f32_1.grad);
    try testing.expectEqual(-3.0, value_f32_2.grad);
}

test "mul apply" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op1 = Op(f32).Mul{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    const result_f32 = op1.apply();
    try testing.expectEqual(50.0, result_f32.data);
    try testing.expectEqual(op1, result_f32.op.?.mul);

    var value_i32_1 = Value(i32).init(10);
    var value_i32_2 = Value(i32).init(5);
    const op2 = Op(i32).Mul{
        .left = &value_i32_1,
        .right = &value_i32_2,
    };
    const result_i32 = op2.apply();
    try testing.expectEqual(50, result_i32.data);
    try testing.expectEqual(op2, result_i32.op.?.mul);
}

test "mul backward" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op = Op(f32).Mul{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    var out = op.apply();
    out.grad = 3.0;
    op.backward(&out);
    try testing.expectEqual(15.0, value_f32_1.grad);
    try testing.expectEqual(30.0, value_f32_2.grad);
}

test "div apply" {
    var value_f32_1 = Value(f32).init(10.0);
    var value_f32_2 = Value(f32).init(5.0);
    const op1 = Op(f32).Div{
        .left = &value_f32_1,
        .right = &value_f32_2,
    };
    const result_f32 = op1.apply();
    try testing.expectEqual(2.0, result_f32.data);
    try testing.expectEqual(op1, result_f32.op.?.div);

    var value_i32_1 = Value(i32).init(10);
    var value_i32_2 = Value(i32).init(5);
    const op2 = Op(i32).Div{
        .left = &value_i32_1,
        .right = &value_i32_2,
    };
    const result_i32 = op2.apply();
    try testing.expectEqual(2, result_i32.data);
    try testing.expectEqual(op2, result_i32.op.?.div);
}

test "div backward" {
    var value_i32_1 = Value(i32).init(10);
    var value_i32_2 = Value(i32).init(5);
    const op = Op(i32).Div{
        .left = &value_i32_1,
        .right = &value_i32_2,
    };
    var out = op.apply();
    out.grad = 3.0;
    op.backward(&out);
    try testing.expectEqual(0.6, value_i32_1.grad);
    try testing.expectEqual(-1.2, value_i32_2.grad);
}

test "relu apply" {
    var value_f32 = Value(f32).init(-10.0);
    const op1 = Op(f32).Relu{ .value = &value_f32 };
    const result_f32 = op1.apply();
    try testing.expectEqual(0.0, result_f32.data);
    try testing.expectEqual(op1, result_f32.op.?.relu);

    var value_i32 = Value(i32).init(10);
    const op2 = Op(i32).Relu{ .value = &value_i32 };
    const result_i32 = op2.apply();
    try testing.expectEqual(10, result_i32.data);
    try testing.expectEqual(op2, result_i32.op.?.relu);
}

test "relu backward" {
    var value_f32 = Value(f32).init(-10.0);
    const op_f32 = Op(f32).Relu{ .value = &value_f32 };
    var out_f32 = op_f32.apply();
    out_f32.grad = 3.0;
    op_f32.backward(&out_f32);
    try testing.expectEqual(0.0, value_f32.grad);

    var value_i32 = Value(i32).init(10);
    const op_i32 = Op(i32).Relu{ .value = &value_i32 };
    var out_i32 = op_i32.apply();
    out_i32.grad = 3.0;
    op_i32.backward(&out_i32);
    try testing.expectEqual(3.0, value_i32.grad);
}

test "pow apply" {
    var value_f32 = Value(f32).init(2.0);
    const op_f32 = Op(f32).Pow{ .base = &value_f32, .exponent = 3.0 };
    const result_f32 = op_f32.apply();
    try testing.expectEqual(8.0, result_f32.data);
    try testing.expectEqual(op_f32, result_f32.op.?.pow);

    var value_i32 = Value(i32).init(2);
    const op_i32 = Op(i32).Pow{ .base = &value_i32, .exponent = 3 };
    const result_i32 = op_i32.apply();
    try testing.expectEqual(8, result_i32.data);
    try testing.expectEqual(op_i32, result_i32.op.?.pow);
}

test "pow backward" {
    var value_f32 = Value(f32).init(2.0);
    const op_f32 = Op(f32).Pow{ .base = &value_f32, .exponent = 3.0 };
    var out_f32 = op_f32.apply();
    out_f32.grad = 2.0;
    op_f32.backward(&out_f32);
    try testing.expectEqual(24.0, value_f32.grad); // 3 * 2^2 * 2.0

    var value_i32 = Value(i32).init(2);
    const op_i32 = Op(i32).Pow{ .base = &value_i32, .exponent = 3 };
    var out_i32 = op_i32.apply();
    out_i32.grad = 2;
    op_i32.backward(&out_i32);
    try testing.expectEqual(24, value_i32.grad); // 3 * 2^2 * 2
}
