const Value = @import("value.zig").Value;
const Data = @import("data.zig").Data;

pub fn Op(comptime T: type) type {
    return union(enum) {
        add: Add,
        // sub: Sub,
        // mul: Mul,
        // div: Div,
        // relu: Relu,

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

            pub fn backward(self: Add, out: Value(T)) void {
                self.left.grad += out.grad;
                self.right.grad += out.grad;
            }
        };

        // const Sub = struct {
        //     left: *const Value,
        //     right: *const Value,

        //     pub fn apply(self: Sub) Value {
        //         return switch (self.left.T) {
        //             f32 => .{
        //                 .data = Data{ .f32 = self.left.data.f32 - self.right.data.f32 },
        //                 .T = f32,
        //                 .op = .{ .sub = self },
        //             },
        //             i32 => .{
        //                 .data = Data{ .i32 = self.left.data.i32 - self.right.data.i32 },
        //                 .T = i32,
        //                 .op = .{ .sub = self },
        //             },
        //             else => unreachable,
        //         };
        //     }

        //     const testing = @import("std").testing;

        //     test apply {
        //         const value_f32_1 = Value.init(f32, 10.0);
        //         const value_f32_2 = Value.init(f32, 5.0);
        //         const op1 = Sub{
        //             .left = &value_f32_1,
        //             .right = &value_f32_2,
        //         };
        //         const result_f32 = op1.apply();
        //         try testing.expectEqual(5.0, result_f32.data.f32);
        //         try testing.expectEqual(f32, result_f32.T);
        //         try testing.expectEqual(op1, result_f32.op.?.sub);

        //         const value_i32_1 = Value.init(i32, 10);
        //         const value_i32_2 = Value.init(i32, 5);
        //         const op2 = Sub{
        //             .left = &value_i32_1,
        //             .right = &value_i32_2,
        //         };
        //         const result_i32 = op2.apply();
        //         try testing.expectEqual(5, result_i32.data.i32);
        //         try testing.expectEqual(i32, result_i32.T);
        //         try testing.expectEqual(op2, result_i32.op.?.sub);
        //     }
        // };

        // const Mul = struct {
        //     left: *const Value,
        //     right: *const Value,

        //     pub fn apply(self: Mul) Value {
        //         return switch (self.left.T) {
        //             f32 => .{
        //                 .data = Data{ .f32 = self.left.data.f32 * self.right.data.f32 },
        //                 .T = f32,
        //                 .op = .{ .mul = self },
        //             },
        //             i32 => .{
        //                 .data = Data{ .i32 = self.left.data.i32 * self.right.data.i32 },
        //                 .T = i32,
        //                 .op = .{ .mul = self },
        //             },
        //             else => unreachable,
        //         };
        //     }

        //     const testing = @import("std").testing;

        //     test apply {
        //         const value_f32_1 = Value.init(f32, 10.0);
        //         const value_f32_2 = Value.init(f32, 5.0);
        //         const op1 = Mul{
        //             .left = &value_f32_1,
        //             .right = &value_f32_2,
        //         };
        //         const result_f32 = op1.apply();
        //         try testing.expectEqual(50.0, result_f32.data.f32);
        //         try testing.expectEqual(f32, result_f32.T);
        //         try testing.expectEqual(op1, result_f32.op.?.mul);

        //         const value_i32_1 = Value.init(i32, 10);
        //         const value_i32_2 = Value.init(i32, 5);
        //         const op2 = Mul{
        //             .left = &value_i32_1,
        //             .right = &value_i32_2,
        //         };
        //         const result_i32 = op2.apply();
        //         try testing.expectEqual(50, result_i32.data.i32);
        //         try testing.expectEqual(i32, result_i32.T);
        //         try testing.expectEqual(op2, result_i32.op.?.mul);
        //     }
        // };

        // const Div = struct {
        //     left: *const Value,
        //     right: *const Value,

        //     pub fn apply(self: Div) Value {
        //         return switch (self.left.T) {
        //             f32 => .{
        //                 .data = Data{ .f32 = self.left.data.f32 / self.right.data.f32 },
        //                 .T = f32,
        //                 .op = .{ .div = self },
        //             },
        //             i32 => .{
        //                 .data = Data{ .i32 = self.left.data.i32 / self.right.data.i32 },
        //                 .T = i32,
        //                 .op = .{ .div = self },
        //             },
        //             else => unreachable,
        //         };
        //     }

        //     const testing = @import("std").testing;

        //     test apply {
        //         const value_f32_1 = Value.init(f32, 10.0);
        //         const value_f32_2 = Value.init(f32, 5.0);
        //         const op1 = Div{
        //             .left = &value_f32_1,
        //             .right = &value_f32_2,
        //         };
        //         const result_f32 = op1.apply();
        //         try testing.expectEqual(2.0, result_f32.data.f32);
        //         try testing.expectEqual(f32, result_f32.T);
        //         try testing.expectEqual(op1, result_f32.op.?.div);

        //         const value_i32_1 = Value.init(i32, 10);
        //         const value_i32_2 = Value.init(i32, 5);
        //         const op2 = Div{
        //             .left = &value_i32_1,
        //             .right = &value_i32_2,
        //         };
        //         const result_i32 = op2.apply();
        //         try testing.expectEqual(2, result_i32.data.i32);
        //         try testing.expectEqual(i32, result_i32.T);
        //         try testing.expectEqual(op2, result_i32.op.?.div);
        //     }
        // };

        //     const Relu = struct {
        //         value: *const Value,

        //         pub fn apply(self: Relu) Value {
        //             return switch (self.value.T) {
        //                 f32 => .{
        //                     .data = Data{ .f32 = if (self.value.data.f32 < 0.0) 0.0 else self.value.data.f32 },
        //                     .T = f32,
        //                     .op = .{ .relu = self },
        //                 },
        //                 i32 => .{
        //                     .data = Data{ .i32 = if (self.value.data.i32 < 0) 0 else self.value.data.i32 },
        //                     .T = i32,
        //                     .op = .{ .relu = self },
        //                 },
        //                 else => unreachable,
        //             };
        //         }

        //         const testing = @import("std").testing;

        //         test apply {
        //             const value_f32 = Value.init(f32, -10.0);
        //             const op1 = Relu{ .value = &value_f32 };
        //             const result_f32 = op1.apply();
        //             try testing.expectEqual(0.0, result_f32.data.f32);
        //             try testing.expectEqual(f32, result_f32.T);
        //             try testing.expectEqual(op1, result_f32.op.?.relu);

        //             const value_i32 = Value.init(i32, 10);
        //             const op2 = Relu{ .value = &value_i32 };
        //             const result_i32 = op2.apply();
        //             try testing.expectEqual(10, result_i32.data.i32);
        //             try testing.expectEqual(i32, result_i32.T);
        //             try testing.expectEqual(op2, result_i32.op.?.relu);
        //         }
        //     };
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
    op.backward(out);
    try testing.expectEqual(3.0, value_f32_1.grad);
    try testing.expectEqual(3.0, value_f32_2.grad);
}
