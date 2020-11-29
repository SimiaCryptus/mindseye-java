package com.simiacryptus.math;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;

import java.util.function.Function;
import java.util.stream.IntStream;

public interface SpacialTransform extends Function<Point, Point> {

    public default SpacialTransform then(SpacialTransform left) {
        return x -> left.apply(SpacialTransform.this.apply(x));
    }

    public static SpacialTransform mobiusTranslate(double... pt) {
        Complex offset = Complex.valueOf(pt[0], pt[1]);
        return x -> {
            Complex point = Complex.valueOf(x.x, x.y);
            Complex result = point.add(offset).divide(point.multiply(offset.conjugate()).add(Complex.ONE));
            return new Point(
                    result.getReal(),
                    result.getImaginary()
            );
        };
    }

    public static SpacialTransform scale(double scaleX, double scaleY) {
        return pt -> new Point(pt.x * scaleX, pt.y * scaleY);
    }


    public static SpacialTransform rotate(double theta) {
        double cos = FastMath.cos(theta);
        double sin = FastMath.sin(theta);
        return x -> new Point(
                x.x * cos - x.y * sin,
                x.y * cos + x.x * sin
        );
    }
}
