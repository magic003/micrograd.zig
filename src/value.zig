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
