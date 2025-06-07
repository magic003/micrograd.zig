const std = @import("std");
const mg = @import("micrograd");

/// Example copied from the README of [micrograd](https://github.com/karpathy/micrograd).
///
/// In Python:
/// ```python
/// a = Value(-4.0)
/// b = Value(2.0)
/// c = a + b
/// d = a * b + b**3
/// c += c + 1
/// c += 1 + c + (-a)
/// d += d * 2 + (b + a).relu()
/// d += 3 * d + (b - a).relu()
/// e = c - d
/// f = e**2
/// g = f / 2.0
/// g += 10.0 / f
/// print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
/// g.backward()
/// print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
/// print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
/// ```
pub fn main() !void {
    const Vf32 = mg.Value(f32);
    var a = Vf32.init(-4.0);
    var b = Vf32.init(2.0);
    // c = a + b
    var c = blk: {
        var a_add_b = a.add(&b);
        break :blk &a_add_b;
    };
    // c += c + 1
    {
        var one = Vf32.init(1.0);
        var c_add_one = c.add(&one);
        var result = c.add(&c_add_one);
        c = &result;
    }
    // c += 1 + c - a
    {
        var one = Vf32.init(1.0);
        var one_add_c = one.add(c);
        var sub_a = one_add_c.sub(&a);
        var result = c.add(&sub_a);
        c = &result;
    }
    // d = a * b + b**3
    var d: *Vf32 = blk: {
        var a_mul_b = a.mul(&b);
        var b_pow_3 = b.pow(3.0);
        var result = a_mul_b.add(&b_pow_3);
        break :blk &result;
    };
    // d += d * 2 + (b + a).relu()
    {
        var two = Vf32.init(2.0);
        var d_mul_two = d.mul(&two);
        var b_add_a = b.add(&a);
        var relu_b_add_a = b_add_a.relu();
        var d_add = d_mul_two.add(&relu_b_add_a);
        var result = d.add(&d_add);
        d = &result;
    }
    // d += 3 * d + (b - a).relu()
    {
        var three = Vf32.init(3.0);
        var d_mul_three = d.mul(&three);
        var b_sub_a = b.sub(&a);
        var relu_b_sub_a = b_sub_a.relu();
        var d_add = d_mul_three.add(&relu_b_sub_a);
        var result = d.add(&d_add);
        d = &result;
    }
    // e = c - d
    var e = c.sub(d);
    // f = e**2
    var f = e.pow(2.0);
    // g = f / 2.0
    var g: *Vf32 = blk: {
        var two = Vf32.init(2.0);
        var result = f.div(&two);
        break :blk &result;
    };
    // g += 10.0 / f
    {
        var ten = Vf32.init(10.0);
        var ten_div_f = ten.div(&f);
        var result = g.add(&ten_div_f);
        g = &result;
    }
    // outcome of the forward pass
    std.debug.print("g: {d:.4}\n", .{g.data});
    try g.backward();
    std.debug.print("a.grad: {d:.4}\n", .{a.grad});
    std.debug.print("b.grad: {d:.4}\n", .{b.grad});
}
