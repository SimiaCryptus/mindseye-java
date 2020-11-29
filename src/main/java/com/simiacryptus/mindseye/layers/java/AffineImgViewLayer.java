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
 * The type Img view layer.
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
   * Gets offset x.
   *
   * @return the offset x
   */
  public int getOffsetX() {
    return offsetX;
  }

  /**
   * Sets offset x.
   *
   * @param offsetX the offset x
   */
  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  /**
   * Gets offset y.
   *
   * @return the offset y
   */
  public int getOffsetY() {
    return offsetY;
  }

  /**
   * Sets offset y.
   *
   * @param offsetY the offset y
   */
  public void setOffsetY(int offsetY) {
    this.offsetY = offsetY;
  }

  /**
   * Gets rotation center x.
   *
   * @return the rotation center x
   */
  public int getRotationCenterX() {
    return rotationCenterX;
  }

  /**
   * Sets rotation center x.
   *
   * @param rotationCenterX the rotation center x
   */
  public void setRotationCenterX(int rotationCenterX) {
    this.rotationCenterX = rotationCenterX;
  }

  /**
   * Gets rotation center y.
   *
   * @return the rotation center y
   */
  public int getRotationCenterY() {
    return rotationCenterY;
  }

  /**
   * Sets rotation center y.
   *
   * @param rotationCenterY the rotation center y
   */
  public void setRotationCenterY(int rotationCenterY) {
    this.rotationCenterY = rotationCenterY;
  }

  /**
   * Gets rotation radians.
   *
   * @return the rotation radians
   */
  public double getRotationRadians() {
    return rotationRadians;
  }

  /**
   * Sets rotation radians.
   *
   * @param rotationRadians the rotation radians
   */
  public void setRotationRadians(double rotationRadians) {
    this.rotationRadians = rotationRadians;
  }

  /**
   * From json img view layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img view layer
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
    if(x < xMin || x >= xMax) return xy;
    if(y < yMin || y >= yMax) return xy;
    double dist = Math.sqrt(x*x+y*y);
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
    return new Point(x,y);
  }

  public int getxMax() {
    return xMax;
  }

  public AffineImgViewLayer setxMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public int getxMin() {
    return xMin;
  }

  public AffineImgViewLayer setxMin(int xMin) {
    this.xMin = xMin;
    return this;
  }

  public int getyMax() {
    return yMax;
  }

  public AffineImgViewLayer setyMax(int yMax) {
    this.yMax = yMax;
    return this;
  }

  public int getyMin() {
    return yMin;
  }

  public AffineImgViewLayer setyMin(int yMin) {
    this.yMin = yMin;
    return this;
  }

  public int getrMax() {
    return rMax;
  }

  public AffineImgViewLayer setrMax(int rMax) {
    this.rMax = rMax;
    return this;
  }

  public int getrMin() {
    return rMin;
  }

  public AffineImgViewLayer setrMin(int rMin) {
    this.rMin = rMin;
    return this;
  }
}
