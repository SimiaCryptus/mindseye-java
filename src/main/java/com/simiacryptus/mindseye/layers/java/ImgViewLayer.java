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
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.apache.commons.math3.util.FastMath;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

/**
 * The type Img view layer.
 */
@SuppressWarnings("serial")
public class ImgViewLayer extends LayerBase {

  private double negativeBias = 255;
  private boolean wrap;
  private int sizeX;
  private int sizeY;
  private int offsetX;
  private int offsetY;
  private int rotationCenterX;
  private int rotationCenterY;
  private int[] channelSelector;
  private double rotationRadians;

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgViewLayer(final int sizeX, final int sizeY) {
    this(sizeX, sizeY, false);
  }

  /**
   * Instantiates a new Img view layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   * @param wrap  the wrap
   */
  public ImgViewLayer(final int sizeX, final int sizeY, boolean wrap) {
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
  public ImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY) {
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
  public ImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY, final boolean wrap) {
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
  protected ImgViewLayer(@Nonnull final JsonObject json) {
    super(json);
    setSizeX(json.getAsJsonPrimitive("sizeX").getAsInt());
    setSizeY(json.getAsJsonPrimitive("sizeY").getAsInt());
    setOffsetX(json.getAsJsonPrimitive("offsetX").getAsInt());
    setOffsetY(json.getAsJsonPrimitive("offsetY").getAsInt());
    setNegativeBias(json.getAsJsonPrimitive("negativeBias").getAsDouble());
    setRotationCenterX(json.getAsJsonPrimitive("rotationCenterX").getAsInt());
    setRotationCenterY(json.getAsJsonPrimitive("rotationCenterY").getAsInt());
    setRotationRadians(json.getAsJsonPrimitive("rotationRadians").getAsDouble());
    JsonArray _channelPermutationFilter = json.getAsJsonArray("channelPermutationFilter");
    if (null != _channelPermutationFilter) {
      int[] channelSelector1 = new int[_channelPermutationFilter.size()];
      setChannelSelector(channelSelector1);
      for (int i = 0; i < getChannelSelector().length; i++) {
        getChannelSelector()[i] = _channelPermutationFilter.get(i).getAsInt();
      }
    }
    //channelSelector
    JsonPrimitive toroidal = json.getAsJsonPrimitive("wrap");
    this.setWrap(null != toroidal && toroidal.getAsBoolean());
  }

  /**
   * Get channel selector int [ ].
   *
   * @return the int [ ]
   */
  public int[] getChannelSelector() {
    return channelSelector;
  }

  /**
   * Sets channel selector.
   *
   * @param channelSelector the channel selector
   */
  public void setChannelSelector(int[] channelSelector) {
    this.channelSelector = channelSelector;
  }

  /**
   * Gets negative bias.
   *
   * @return the negative bias
   */
  public double getNegativeBias() {
    return negativeBias;
  }

  /**
   * Sets negative bias.
   *
   * @param negativeBias the negative bias
   */
  public void setNegativeBias(double negativeBias) {
    this.negativeBias = negativeBias;
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
   * Gets size x.
   *
   * @return the size x
   */
  public int getSizeX() {
    return sizeX;
  }

  /**
   * Sets size x.
   *
   * @param sizeX the size x
   */
  public void setSizeX(int sizeX) {
    this.sizeX = sizeX;
  }

  /**
   * Gets size y.
   *
   * @return the size y
   */
  public int getSizeY() {
    return sizeY;
  }

  /**
   * Sets size y.
   *
   * @param sizeY the size y
   */
  public void setSizeY(int sizeY) {
    this.sizeY = sizeY;
  }

  /**
   * Is wrap boolean.
   *
   * @return the boolean
   */
  public boolean isWrap() {
    return wrap;
  }

  /**
   * Sets wrap.
   *
   * @param wrap the wrap
   */
  public void setWrap(boolean wrap) {
    this.wrap = wrap;
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
  public static ImgViewLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgViewLayer(json);
  }

  private static void set(@Nonnull Tensor tensor, int width, int height, int x, int y, int channel, boolean wrap,
                          double value) {
    try {
      assert channel >= 0 : channel;
      if (wrap) {
        while (x < 0)
          x += width;
        x %= width;
        while (y < 0)
          y += height;
        y %= height;
      }
      if (x < 0) {
        return;
      } else if (x >= width) {
        return;
      }
      if (y < 0) {
        return;
      } else if (y >= height) {
        return;
      }
      tensor.set(x, y, channel, value);
    } finally {
      tensor.freeRef();
    }
  }

  private static double get(@Nonnull @RefIgnore Tensor tensor, int width, int height, int x, int y, int channel, boolean wrap) {
    assert channel >= 0 : channel;
    if (wrap) {
      while (x < 0)
        x += width;
      x %= width;
      while (y < 0)
        y += height;
      y %= height;
    }
    if (x < 0) {
      return 0.0;
    } else if (x >= width) {
      return 0.0;
    }
    if (y < 0) {
      return 0.0;
    } else if (y >= height) {
      return 0.0;
    }
    return tensor.get(x, y, channel);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull final int[] dimOut = getViewDimensions(inputDims, new int[]{getSizeX(), getSizeY(), inputDims[2]},
        new int[]{getOffsetX(), getOffsetY(), 0});
    if (null != channelSelector)
      dimOut[2] = channelSelector.length;
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(addRef(), inputDims, input.getAccumulator(), input.isAlive());
    input.freeRef();
    TensorArray data = fwd(batch, dimOut);
    return new Result(data, accumulator, alive);
  }

  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @param offset                the offset
   * @return the int [ ]
   */
  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    RefArrays.parallelSetAll(viewDim, i -> isWrap() ? destinationDimensions[i]
        : Math.min(sourceDimensions[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", getSizeX());
    json.addProperty("sizeY", getSizeY());
    json.addProperty("offsetX", getOffsetX());
    json.addProperty("offsetY", getOffsetY());
    json.addProperty("negativeBias", getNegativeBias());
    json.addProperty("rotationCenterX", getRotationCenterX());
    json.addProperty("rotationCenterY", getRotationCenterY());
    json.addProperty("rotationRadians", getRotationRadians());
    json.addProperty("wrap", isWrap());
    if (null != getChannelSelector()) {
      JsonArray _channelPermutationFilter = new JsonArray();
      for (int i : getChannelSelector()) {
        _channelPermutationFilter.add(i);
      }
      json.add("channelSelector", _channelPermutationFilter);
    }
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgViewLayer addRef() {
    return (ImgViewLayer) super.addRef();
  }

  /**
   * Fwd.
   *
   * @param inputData  the input data
   * @param outputData the output data
   */
  protected void fwd(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    int[] inputDims = inputData.getDimensions();
    @Nonnull final int[] inDim = inputDims;
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : RefArrays.toString(inDim) + "; " + RefArrays.toString(outDim);
    outputData.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
      int[] coords = c.getCoords();
      double[] xy = coordinateMapping(coords[0], coords[1]);
      int x = (int) Math.round(xy[0]);
      int y = (int) Math.round(xy[1]);
      int channel;
      if (null != channelSelector)
        channel = channelSelector[coords[2]];
      else
        channel = coords[2] + 1;
      if (0 < channel) {
        outputData.set(c, get(inputData, inputDims[0], inputDims[1], x, y, channel - 1, wrap));
      } else {
        final double value = getNegativeBias() -
            get(inputData, inputDims[0], inputDims[1], x, y, -channel - 1, wrap);
        outputData.set(c, value);
      }
    }, inputData, outputData));
  }

  /**
   * Bck.
   *
   * @param outputDelta the output delta
   * @param inputDelta  the input delta
   */
  protected void bck(@Nonnull final Tensor outputDelta, @Nonnull final Tensor inputDelta) {
    int[] outDeltaDims = outputDelta.getDimensions();
    @Nonnull final int[] inputDeltaDims = inputDelta.getDimensions();
    assert 3 == outDeltaDims.length;
    assert 3 == inputDeltaDims.length;
    assert outDeltaDims[2] == inputDeltaDims[2] : RefArrays.toString(outDeltaDims) + "; "
        + RefArrays.toString(inputDeltaDims);
    outputDelta.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
      int[] outCoord = c.getCoords();
      double[] inCoords = coordinateMapping(outCoord[0], outCoord[1]);
      int x = (int) Math.round(inCoords[0]);
      int y = (int) Math.round(inCoords[1]);
      int channel;
      if (null != channelSelector)
        channel = channelSelector[outCoord[2]];
      else
        channel = outCoord[2] + 1;
      if (0 < channel) {
        set(inputDelta.addRef(), inputDeltaDims[0], inputDeltaDims[1], x, y, channel - 1,
            wrap, outputDelta.get(c));
      } else {
        set(inputDelta.addRef(), inputDeltaDims[0], inputDeltaDims[1], x, y, -channel - 1,
            wrap, -outputDelta.get(c));
      }
    }, inputDelta, outputDelta));
  }

  /**
   * Coordinate mapping double [ ].
   *
   * @param xy the xy
   * @return the double [ ]
   */
  @Nonnull
  protected double[] coordinateMapping(@Nonnull double... xy) {
    xy[0] += offsetX;
    xy[1] += offsetY;
    xy[0] -= rotationCenterX;
    xy[1] -= rotationCenterY;
    double x1 = xy[0];
    double y1 = xy[1];
    double sin = FastMath.sin(rotationRadians);
    double cos = FastMath.cos(rotationRadians);
    xy[0] = cos * x1 - sin * y1;
    xy[1] = sin * x1 + cos * y1;
    xy[0] += rotationCenterX;
    xy[1] += rotationCenterY;
    return xy;
  }

  @NotNull
  private TensorArray fwd(TensorList batch, int[] dimOut) {
    return new TensorArray(RefIntStream.range(0, batch.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nonnull final Tensor outputData = new Tensor(dimOut);
          fwd(batch.get(dataIndex), outputData.addRef());
          return outputData;
        }, batch)).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private ImgViewLayer imgViewLayer;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param imgViewLayer the img view layer
     * @param inputDims    the input dims
     * @param accumulator  the accumulator
     * @param alive        the alive
     */
    public Accumulator(ImgViewLayer imgViewLayer, int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.inputDims = inputDims;
      this.imgViewLayer = imgViewLayer;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable final Tensor err = error.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(inputDims);
              imgViewLayer.bck(err.addRef(),
                  passback.addRef());
              err.freeRef();
              return passback;
            }, error.addRef())).toArray(Tensor[]::new));
        this.accumulator.accept(buffer == null ? null : buffer.addRef(), tensorArray);
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      imgViewLayer.freeRef();
    }
  }
}
