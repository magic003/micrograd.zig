/// The data saved in a value.
pub const Data = union(enum) {
    f32: f32,
    i32: i32,

    /// Creates a data with the given type and value.
    pub fn init(comptime T: type, value: T) Data {
        return switch (T) {
            f32 => Data{ .f32 = value },
            i32 => Data{ .i32 = value },
            else => @compileError("Unsupported data type: " ++ @typeName(T)),
        };
    }
};
