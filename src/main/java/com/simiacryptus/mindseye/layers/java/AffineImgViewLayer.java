/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.math.Point;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.wrappers.RefArrays;
import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * This class represents an AffineImgViewLayer.
 *
 * @author Some Author
 * @version 1.0
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class AffineImgViewLayer extends ImgViewLayerBase {

  private int offsetX;
  private int offsetY;
  private int xMax = Integer.MAX_VALUE;
  private int xMin = 0;
  private int yMax = Integer.MAX_VALUE;
  private int yMin = 0;
  private int rMax = Integer.MAX_VALUE;
  private int rMin = 0;
  private int rotationCenterX;
  private int rotationCenterY;
  private double rotationRadians;

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public AffineImgViewLayer(final int sizeX, final int sizeY) {
    this(sizeX, sizeY, false);
  }

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   * @param wrap  the wrap
   */
  public AffineImgViewLayer(final int sizeX, final int sizeY, boolean wrap) {
    this(sizeX, sizeY, 0, 0, wrap);
  }

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX   the size x
   * @param sizeY   the size y
   * @param offsetX the offset x
   * @param offsetY the offset y
   */
  public AffineImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY) {
    this(sizeX, sizeY, offsetX, offsetY, false);
  }

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX   the size x
   * @param sizeY   the size y
   * @param offsetX the offset x
   * @param offsetY the offset y
   * @param wrap    the wrap
   */
  public AffineImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY, final boolean wrap) {
    super();
    this.setSizeX(sizeX);
    this.setSizeY(sizeY);
    this.setOffsetX(offsetX);
    this.setOffsetY(offsetY);
    this.setWrap(wrap);
  }

  /**
   * Instantiates a new Img view layer.
   *
   * @param json the json
   */
  protected AffineImgViewLayer(@Nonnull final JsonObject json) {
    super(json);
    setOffsetX(json.getAsJsonPrimitive("offsetX").getAsInt());
    setOffsetY(json.getAsJsonPrimitive("offsetY").getAsInt());
    setxMax(json.getAsJsonPrimitive("xMax").getAsInt());
    setxMin(json.getAsJsonPrimitive("xMin").getAsInt());
    setyMax(json.getAsJsonPrimitive("yMax").getAsInt());
    setyMin(json.getAsJsonPrimitive("yMin").getAsInt());
    setrMax(json.getAsJsonPrimitive("xMax").getAsInt());
    setrMin(json.getAsJsonPrimitive("xMin").getAsInt());
    setRotationCenterX(json.getAsJsonPrimitive("rotationCenterX").getAsInt());
    setRotationCenterY(json.getAsJsonPrimitive("rotationCenterY").getAsInt());
    setRotationRadians(json.getAsJsonPrimitive("rotationRadians").getAsDouble());
    //channelSelector
  }

  /**
   * Returns the view dimensions of the input dimensions.
   *
   * @param inputDims the input dimensions
   * @return the view dimensions
   * @docgenVersion 9
   */
  @Nonnull
  public int[] getViewDimensions(@Nonnull int @NotNull [] inputDims) {
    int[] destinationDimensions = new int[]{getSizeX(), getSizeY(), inputDims[2]};
    int[] offset = new int[]{getOffsetX(), getOffsetY(), 0};
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim, i -> isWrap() ? destinationDimensions[i]
        : Math.min(inputDims[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0));
    if (null != channelSelector)
      viewDim[2] = channelSelector.length;
    return viewDim;
  }

  /**
   * Returns the offsetX.
   *
   * @return the offsetX.
   * @docgenVersion 9
   */
  public int getOffsetX() {
    return offsetX;
  }

  /**
   * Sets the x offset.
   *
   * @param offsetX the x offset
   * @docgenVersion 9
   */
  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  /**
   * Returns the offset on the y-axis.
   *
   * @return the offset on the y-axis
   * @docgenVersion 9
   */
  public int getOffsetY() {
    return offsetY;
  }

  /**
   * Sets the offset on the y-axis.
   *
   * @param offsetY the offset on the y-axis
   * @docgenVersion 9
   */
  public void setOffsetY(int offsetY) {
    this.offsetY = offsetY;
  }

  /**
   * Returns the x-coordinate of the center of rotation.
   *
   * @return the x-coordinate of the center of rotation
   * @docgenVersion 9
   */
  public int getRotationCenterX() {
    return rotationCenterX;
  }

  /**
   * Sets the x-coordinate of the center of rotation.
   *
   * @param rotationCenterX the x-coordinate of the center of rotation
   * @docgenVersion 9
   */
  public void setRotationCenterX(int rotationCenterX) {
    this.rotationCenterX = rotationCenterX;
  }

  /**
   * Returns the rotation center's y-coordinate.
   *
   * @return the rotation center's y-coordinate
   * @docgenVersion 9
   */
  public int getRotationCenterY() {
    return rotationCenterY;
  }

  /**
   * Sets the rotation center along the y-axis.
   *
   * @param rotationCenterY the new rotation center along the y-axis
   * @docgenVersion 9
   */
  public void setRotationCenterY(int rotationCenterY) {
    this.rotationCenterY = rotationCenterY;
  }

  /**
   * Returns the rotation in radians.
   *
   * @return the rotation in radians
   * @docgenVersion 9
   */
  public double getRotationRadians() {
    return rotationRadians;
  }

  /**
   * Sets the rotation in radians.
   *
   * @param rotationRadians the rotation in radians
   * @docgenVersion 9
   */
  public void setRotationRadians(double rotationRadians) {
    this.rotationRadians = rotationRadians;
  }

  /**
   * Creates an AffineImgViewLayer from a JSON object.
   *
   * @param json The JSON object to create the AffineImgViewLayer from.
   * @param rs   A map of character sequences to byte arrays.
   * @return The newly created AffineImgViewLayer.
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static AffineImgViewLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AffineImgViewLayer(json);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("offsetX", getOffsetX());
    json.addProperty("offsetY", getOffsetY());
    json.addProperty("xMax", getxMax());
    json.addProperty("xMin", getxMin());
    json.addProperty("yMax", getyMax());
    json.addProperty("yMin", getyMin());
    json.addProperty("rMax", getrMax());
    json.addProperty("rMin", getrMin());
    json.addProperty("rotationCenterX", getRotationCenterX());
    json.addProperty("rotationCenterY", getRotationCenterY());
    json.addProperty("rotationRadians", getRotationRadians());
    if (null != getChannelSelector()) {
      JsonArray _channelPermutationFilter = new JsonArray();
      for (int i : getChannelSelector()) {
        _channelPermutationFilter.add(i);
      }
      json.add("channelSelector", _channelPermutationFilter);
    }
    return json;
  }

  /**
   * Coordinate mapping double [ ].
   *
   * @param xy the xy
   * @return the double [ ]
   */
  @Override
  protected Point coordinateMapping(@Nonnull Point xy) {
    double x = xy.x;
    double y = xy.y;
    if (x < xMin || x >= xMax) return xy;
    if (y < yMin || y >= yMax) return xy;
    double dist = Math.sqrt(x * x + y * y);
    if (dist >= rMin && dist < rMax) {
      x += offsetX;
      y += offsetY;
      x -= rotationCenterX;
      y -= rotationCenterY;
      double sin = FastMath.sin(rotationRadians);
      double cos = FastMath.cos(rotationRadians);
      double x2 = x;
      double y2 = y;
      x = cos * x2 - sin * y2;
      y = cos * y2 + sin * x2;
      x += rotationCenterX;
      y += rotationCenterY;
    }
    return new Point(x, y);
  }

  /**
   * Returns the maximum x value of the graph.
   *
   * @return the maximum x value of the graph
   * @docgenVersion 9
   */
  public int getxMax() {
    return xMax;
  }

  /**
   * Sets the maximum x value
   *
   * @param xMax the maximum x value
   * @docgenVersion 9
   */
  public void setxMax(int xMax) {
    this.xMax = xMax;
  }

  /**
   * Returns the minimum x value of the rectangle.
   *
   * @return the minimum x value of the rectangle
   * @docgenVersion 9
   */
  public int getxMin() {
    return xMin;
  }

  /**
   * Sets the minimum x value.
   *
   * @param xMin the minimum x value
   * @docgenVersion 9
   */
  public void setxMin(int xMin) {
    this.xMin = xMin;
  }

  /**
   * Returns the maximum y value of the graph.
   *
   * @return the maximum y value of the graph
   * @docgenVersion 9
   */
  public int getyMax() {
    return yMax;
  }

  /**
   * Sets the maximum y value
   *
   * @param yMax the maximum y value
   * @docgenVersion 9
   */
  public void setyMax(int yMax) {
    this.yMax = yMax;
  }

  /**
   * Returns the minimum y value of the graph.
   *
   * @return the minimum y value of the graph
   * @docgenVersion 9
   */
  public int getyMin() {
    return yMin;
  }

  /**
   * Sets the minimum value of y.
   *
   * @param yMin the minimum value of y
   * @docgenVersion 9
   */
  public void setyMin(int yMin) {
    this.yMin = yMin;
  }

  /**
   * Returns the maximum radius of the shape.
   *
   * @return the maximum radius of the shape
   * @docgenVersion 9
   */
  public int getrMax() {
    return rMax;
  }

  /**
   * Sets the maximum radius of the shape.
   *
   * @param rMax the maximum radius of the shape
   * @docgenVersion 9
   */
  public void setrMax(int rMax) {
    this.rMax = rMax;
  }

  /**
   * Returns the minimum value of the range.
   *
   * @return the minimum value of the range
   * @docgenVersion 9
   */
  public int getrMin() {
    return rMin;
  }

  /**
   * Sets the minimum value for the range.
   *
   * @param rMin the minimum value for the range
   * @docgenVersion 9
   */
  public void setrMin(int rMin) {
    this.rMin = rMin;
  }

  @Override
  public void _free() {
    super._free();
  }
}
