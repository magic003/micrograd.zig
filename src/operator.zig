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
