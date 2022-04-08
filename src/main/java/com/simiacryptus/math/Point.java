package com.simiacryptus.math;

/**
 * This class represents a point in two-dimensional space.
 *
 * @param x the x-coordinate of the point
 * @param y the y-coordinate of the point
 * @docgenVersion 9
 */
public class Point {
  public final double x;
  public final double y;

  public Point(double x, double y) {
    if (!Double.isFinite(x)) {
      throw new IllegalArgumentException();
    }
    if (!Double.isFinite(y)) {
      throw new IllegalArgumentException();
    }
    this.x = x;
    this.y = y;
  }

  /**
   * Returns the root mean square of this vector.
   *
   * @return the root mean square of this vector
   * @docgenVersion 9
   */
  public double rms() {
    return Geometry.rms(this.x, this.y);
  }

  /**
   * Returns the theta value.
   *
   * @return the theta value
   * @docgenVersion 9
   */
  public double theta() {
    return Math.sin(scale(1.0 / rms()).x);
  }

  /**
   * Returns the sum of the squares of the x and y fields.
   *
   * @return the sum of the squares of the x and y fields
   * @docgenVersion 9
   */
  public double sumSq() {
    return Geometry.sumSq(this.x, this.y);
  }

  /**
   * Adds the given point to this point and returns the result as a new point.
   *
   * @param r the point to add
   * @return the result of the addition
   * @docgenVersion 9
   */
  public Point add(Point r) {
    return new Point(x + r.x, y + r.y);
  }

  /**
   * Calculates the distance between this point and another point.
   *
   * @param r The other point.
   * @return The distance between the two points.
   * @docgenVersion 9
   */
  public double dist(Point r) {
    return r.scale(-1).add(this).rms();
  }

  /**
   * Scales the point by a factor.
   *
   * @param f the factor to scale by
   * @return the scaled point
   * @docgenVersion 9
   */
  public Point scale(double f) {
    return new Point(x * f, y * f);
  }

  /**
   * Rotates the point around the origin by the given angle in radians.
   *
   * @param f The angle to rotate by in radians
   * @return The new point after rotation
   * @docgenVersion 9
   */
  public Point rotate(double f) {
    return new Point(
        x * Math.cos(f) - y * Math.sin(f),
        x * Math.sin(f) + y * Math.cos(f));
  }
}
