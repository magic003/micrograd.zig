const Data = @import("data.zig").Data;
const Op = @import("operator.zig").Op;

/// A value stores a scalar data and its gradient.
pub const Value = @This();

/// The type of the value.
T: type,
/// The scalar data.
data: Data,
/// The operation that produced this value, if any.
op: ?Op = null,

/// Creates a value with the given type and value.
pub fn init(comptime T: type, value: T) Value {
    return switch (T) {
        f32 => .{
            .data = Data{ .f32 = value },
            .T = T,
        },
        i32 => .{
            .data = Data{ .i32 = value },
            .T = T,
        },
        else => @compileError("Unsupported data type: " ++ @typeName(T)),
    };
}

/// Adds this value with another value of the same type.
pub fn add(self: Value, other: Value) Value {
    if (self.T != other.T) {
        @compileError("Cannot add values of different types: " ++ @typeName(self.T) ++ " and " ++ @typeName(other.T));
    }

    const op = Op{
        .add = .{
            .left = &self,
            .right = &other,
        },
    };
    return op.add.apply();
}

/// Subtracts another value from this value of the same type.
pub fn sub(self: Value, other: Value) Value {
    if (self.T != other.T) {
        @compileError("Cannot subtract values of different types: " ++ @typeName(self.T) ++ " and " ++ @typeName(other.T));
    }

    const op = Op{
        .sub = .{
            .left = &self,
            .right = &other,
        },
    };
    return op.sub.apply();
}

/// Multiplies this value with another value of the same type.
pub fn mul(self: Value, other: Value) Value {
    if (self.T != other.T) {
        @compileError("Cannot multiply values of different types: " ++ @typeName(self.T) ++ " and " ++ @typeName(other.T));
    }

    const op = Op{
        .mul = .{
            .left = &self,
            .right = &other,
        },
    };
    return op.mul.apply();
}

/// Divides this value by another value of the same type.
pub fn div(self: Value, other: Value) Value {
    if (self.T != other.T) {
        @compileError("Cannot divide values of different types: " ++ @typeName(self.T) ++ " and " ++ @typeName(other.T));
    }

    const op = Op{
        .div = .{
            .left = &self,
            .right = &other,
        },
    };
    return op.div.apply();
}

/// Applies the ReLU activation function to this value.
pub fn relu(self: Value) Value {
    const op = Op{
        .relu = .{
            .value = &self,
        },
    };
    return op.relu.apply();
}

const testing = @import("std").testing;

test init {
    const value_f32 = Value.init(f32, 10.0);
    const value_i32 = Value.init(i32, 10);

    try testing.expectEqual(10.0, value_f32.data.f32);
    try testing.expectEqual(f32, value_f32.T);
    try testing.expectEqual(null, value_f32.op);
    try testing.expectEqual(10, value_i32.data.i32);
    try testing.expectEqual(i32, value_i32.T);
    try testing.expectEqual(null, value_i32.op);
}

test add {
    const value_f32_1 = Value.init(f32, 10.0);
    const value_f32_2 = Value.init(f32, 5.0);
    const result_f32 = value_f32_1.add(value_f32_2);
    try testing.expectEqual(15.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);
    try testing.expect(result_f32.op != null);
}

test sub {
    const value_f32_1 = Value.init(f32, 10.0);
    const value_f32_2 = Value.init(f32, 5.0);
    const result_f32 = value_f32_1.sub(value_f32_2);
    try testing.expectEqual(5.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);
    try testing.expect(result_f32.op != null);
}

test mul {
    const value_f32_1 = Value.init(f32, 10.0);
    const value_f32_2 = Value.init(f32, 5.0);
    const result_f32 = value_f32_1.mul(value_f32_2);
    try testing.expectEqual(50.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);
    try testing.expect(result_f32.op != null);
}

test div {
    const value_f32_1 = Value.init(f32, 10.0);
    const value_f32_2 = Value.init(f32, 5.0);
    const result_f32 = value_f32_1.div(value_f32_2);
    try testing.expectEqual(2.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);
    try testing.expect(result_f32.op != null);
}

test relu {
    const value_f32 = Value.init(f32, -10.0);
    const result_f32 = value_f32.relu();
    try testing.expectEqual(0.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);
    try testing.expect(result_f32.op != null);
}
