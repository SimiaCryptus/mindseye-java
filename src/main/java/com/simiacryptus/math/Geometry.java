package com.simiacryptus.math;

import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.Random;

/**
 * The Geometry class contains a static final Random field.
 *
 * @docgenVersion 9
 */
public class Geometry {

  static final Random random = new Random();

  /**
   * Returns the root mean square of the given array.
   *
   * @param v the array to take the root mean square of
   * @return the root mean square of the array
   * @docgenVersion 9
   */
  static double rms(double[] v) {
    return FastMath.sqrt(sumSq(v));
  }

  /**
   * Calculates the root mean square of two values.
   *
   * @param x the first value
   * @param y the second value
   * @return the root mean square of the two values
   * @docgenVersion 9
   */
  static double rms(double x, double y) {
    return FastMath.sqrt(sumSq(x, y));
  }

  /**
   * Returns the sum of the squares of the given values.
   *
   * @param values the values to sum
   * @return the sum of the squares of the given values
   * @docgenVersion 9
   */
  public static double sumSq(double... values) {
    double s = 0;
    for (double x : values) s += x * x;
    return s;
  }

  /**
   * Calculates the sum of the squares of two numbers.
   *
   * @param x the first number
   * @param y the second number
   * @return the sum of the squares of the two numbers
   * @docgenVersion 9
   */
  public static double sumSq(double x, double y) {
    double s = 0;
    s += x * x;
    s += y * y;
    return s;
  }

  /**
   * Calculates the distance between two points.
   *
   * @param a the first point
   * @param b the second point
   * @return the distance between the two points
   * @docgenVersion 9
   */
  public static double dist(Point a, Point b) {
    return rms(a.x - b.x, a.y - b.y);
  }
}
