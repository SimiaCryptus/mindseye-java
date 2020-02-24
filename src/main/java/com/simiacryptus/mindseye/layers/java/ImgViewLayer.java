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

  public ImgViewLayer(final int sizeX, final int sizeY) {
    this(sizeX, sizeY, false);
  }

  public ImgViewLayer(final int sizeX, final int sizeY, boolean wrap) {
    this(sizeX, sizeY, 0, 0, wrap);
  }

  public ImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY) {
    this(sizeX, sizeY, offsetX, offsetY, false);
  }

  public ImgViewLayer(final int sizeX, final int sizeY, final int offsetX, final int offsetY, final boolean wrap) {
    super();
    this.setSizeX(sizeX);
    this.setSizeY(sizeY);
    this.setOffsetX(offsetX);
    this.setOffsetY(offsetY);
    this.setWrap(wrap);
  }

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

  public int[] getChannelSelector() {
    return channelSelector;
  }

  public void setChannelSelector(int[] channelSelector) {
    this.channelSelector = channelSelector;
  }

  public double getNegativeBias() {
    return negativeBias;
  }

  public void setNegativeBias(double negativeBias) {
    this.negativeBias = negativeBias;
  }

  public int getOffsetX() {
    return offsetX;
  }

  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }

  public int getOffsetY() {
    return offsetY;
  }

  public void setOffsetY(int offsetY) {
    this.offsetY = offsetY;
  }

  public int getRotationCenterX() {
    return rotationCenterX;
  }

  public void setRotationCenterX(int rotationCenterX) {
    this.rotationCenterX = rotationCenterX;
  }

  public int getRotationCenterY() {
    return rotationCenterY;
  }

  public void setRotationCenterY(int rotationCenterY) {
    this.rotationCenterY = rotationCenterY;
  }

  public double getRotationRadians() {
    return rotationRadians;
  }

  public void setRotationRadians(double rotationRadians) {
    this.rotationRadians = rotationRadians;
  }

  public int getSizeX() {
    return sizeX;
  }

  public void setSizeX(int sizeX) {
    this.sizeX = sizeX;
  }

  public int getSizeY() {
    return sizeY;
  }

  public void setSizeY(int sizeY) {
    this.sizeY = sizeY;
  }

  public boolean isWrap() {
    return wrap;
  }

  public void setWrap(boolean wrap) {
    this.wrap = wrap;
  }

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

  private static double get(@Nonnull Tensor tensor, int width, int height, int x, int y, int channel, boolean wrap) {
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
    } finally {
      tensor.freeRef();
    }
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
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @NotNull
  private TensorArray fwd(TensorList batch, int[] dimOut) {
    return new TensorArray(RefIntStream.range(0, batch.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor outputData = new Tensor(dimOut);
              Tensor inputData = batch.get(dataIndex);
              fwd(inputData.addRef(), outputData.addRef());
              inputData.freeRef();
              return outputData;
            }, batch)).toArray(Tensor[]::new));
  }

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

  @Nonnull
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
        outputData.set(c, get(inputData.addRef(), inputDims[0], inputDims[1], x, y, channel - 1, wrap));
      } else {
        final double value = getNegativeBias() - get(inputData.addRef(),
            inputDims[0], inputDims[1], x, y, -channel - 1, wrap);
        outputData.set(c, value);
      }
    }, inputData, outputData));
  }

  @Nonnull
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

  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private ImgViewLayer imgViewLayer;
    private Result.Accumulator accumulator;
    private boolean alive;

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
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        try {
          this.accumulator.accept(buffer1, tensorArray);
        } finally {
          this.accumulator.freeRef();
        }
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
