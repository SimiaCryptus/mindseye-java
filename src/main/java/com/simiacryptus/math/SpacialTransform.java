package com.simiacryptus.math;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;

import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * This is the SpacialTransform interface.
 *
 * @docgenVersion 9
 */
public interface SpacialTransform extends Function<Point, Point> {

  /**
   * Returns a new SpacialTransform that first applies this SpacialTransform to its input, and then applies the given SpacialTransform to the result.
   *
   * @param left the SpacialTransform to apply after this SpacialTransform
   * @return a new SpacialTransform that first applies this SpacialTransform and then applies the given SpacialTransform
   * @docgenVersion 9
   */
  public default SpacialTransform then(SpacialTransform left) {
    return x -> left.apply(SpacialTransform.this.apply(x));
  }

  /**
   * Returns a Mobius transformation that translates a point by a given vector.
   *
   * @param pt the point to be translated
   * @return the Mobius transformation that translates the point
   * @docgenVersion 9
   */
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

  /**
   * Returns a spacial transform that scales points by the given x and y factors.
   *
   * @param scaleX the x scaling factor
   * @param scaleY the y scaling factor
   * @return a spacial transform that scales points by the given x and y factors
   * @docgenVersion 9
   */
  public static SpacialTransform scale(double scaleX, double scaleY) {
    return pt -> new Point(pt.x * scaleX, pt.y * scaleY);
  }


  /**
   * Returns a SpacialTransform that rotates points by theta radians.
   *
   * @param theta the angle to rotate by, in radians
   * @return a SpacialTransform that performs the rotation
   * @docgenVersion 9
   */
  public static SpacialTransform rotate(double theta) {
    double cos = FastMath.cos(theta);
    double sin = FastMath.sin(theta);
    return x -> new Point(
        x.x * cos - x.y * sin,
        x.y * cos + x.x * sin
    );
  }
}
