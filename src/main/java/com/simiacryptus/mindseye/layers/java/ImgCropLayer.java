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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntFunction;

/**
 * This class represents an image crop layer.
 *
 * @param sizeX The width of the crop layer.
 * @param sizeY The height of the crop layer.
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class ImgCropLayer extends LayerBase {

  private final int sizeX;
  private final int sizeY;

  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgCropLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  /**
   * Instantiates a new Img crop layer.
   *
   * @param json the json
   */
  protected ImgCropLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }

  /**
   * Copies the data from the input tensor to the output tensor.
   *
   * @param inputData  the input tensor
   * @param outputData the output tensor
   * @docgenVersion 9
   */
  public static void copy(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : RefArrays.toString(inDim) + "; " + RefArrays.toString(outDim);
    double fx = (inDim[0] - outDim[0]) / 2.0;
    double fy = (inDim[1] - outDim[1]) / 2.0;
    final int paddingX = (int) (fx < 0 ? Math.ceil(fx) : Math.floor(fx));
    final int paddingY = (int) (fy < 0 ? Math.ceil(fy) : Math.floor(fy));
    outputData.coordStream(true).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
      int x = c.getCoords()[0] + paddingX;
      int y = c.getCoords()[1] + paddingY;
      int z = c.getCoords()[2];
      int width = inputData.getDimensions()[0];
      int height = inputData.getDimensions()[1];
      double value;
      if (x < 0) {
        value = 0.0;
      } else if (x >= width) {
        value = 0.0;
      } else if (y < 0) {
        value = 0.0;
      } else if (y >= height) {
        value = 0.0;
      } else {
        value = inputData.get(x, y, z);
      }
      outputData.set(c, value);
    }, outputData, inputData));
  }

  /**
   * @param json the JSON object to create the layer from
   * @param rs   the resources to use for the layer
   * @return the new layer
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgCropLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgCropLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(inputDims, input.getAccumulator(), alive);
    input.freeRef();
    TensorArray data = fwd(batch, inputDims[2]);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgCropLayer addRef() {
    return (ImgCropLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList batch, int inputDim) {
    return new TensorArray(RefIntStream.range(0, batch.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nonnull final Tensor outputData = new Tensor(sizeX, sizeY, inputDim);
          Tensor inputData = batch.get(dataIndex);
          ImgCropLayer.copy(inputData.addRef(),
              outputData.addRef());
          inputData.freeRef();
          return outputData;
        }, batch)).toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to track the results of an operation.
   *
   * @param inputDims   The dimensions of the input data.
   * @param accumulator The accumulator to use.
   * @param alive       A boolean value indicating whether the operation is still alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputDims   the input dims
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.inputDims = inputDims;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable final Tensor err = error.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(inputDims);
              copy(err.addRef(), passback.addRef());
              err.freeRef();
              return passback;
            }, error.addRef())).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
