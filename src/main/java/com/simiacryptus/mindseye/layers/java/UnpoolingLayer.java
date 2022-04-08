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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * This class represents an unpooling layer.
 *
 * @param sizeX The size in the x dimension.
 * @param sizeY The size in the y dimension.
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class UnpoolingLayer extends LayerBase {

  private final int sizeX;
  private final int sizeY;

  /**
   * Instantiates a new Unpooling layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public UnpoolingLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  /**
   * Instantiates a new Unpooling layer.
   *
   * @param json the json
   */
  protected UnpoolingLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }

  /**
   * Copies and condenses the input data into the output data.
   *
   * @param inputData  the input data to copy and condense
   * @param outputData the output data to copy the input data into
   * @return the output data
   * @docgenVersion 9
   */
  @Nonnull
  public static Tensor copyCondense(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    assert inDim[2] == outDim[2];
    assert 0 == inDim[0] % outDim[0];
    assert 0 == inDim[1] % outDim[1];
    final int kernelSizeX = inDim[0] / outDim[0];
    final int kernelSizeY = inDim[0] / outDim[0];
    int index = 0;
    @Nullable final double[] outputDataData = outputData.getData();
    for (int z = 0; z < inDim[2]; z++) {
      for (int y = 0; y < inDim[1]; y += kernelSizeY) {
        for (int x = 0; x < inDim[0]; x += kernelSizeX) {
          int xx = kernelSizeX / 2;
          int yy = kernelSizeY / 2;
          final double value = inputData.get(x + xx, y + yy, z);
          //          final double value = IntStream.range(0, kernelSizeX).mapToDouble(i -> i).flatMap(xx -> {
          //            return IntStream.range(0, kernelSizeY).mapToDouble(yy -> {
          //              return inputData.get((int) (finalX + xx), finalY + yy, finalZ);
          //            });
          //          }).sum();
          outputData.set(x / kernelSizeX, y / kernelSizeY, z, value);
        }
      }
    }
    inputData.freeRef();
    return outputData;
  }

  /**
   * Copies the input data to the output data, expanding the output data if necessary.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the output data
   * @throws NullPointerException if either inputData or outputData is null
   * @docgenVersion 9
   */
  @Nonnull
  public static Tensor copyExpand(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] == outDim[2];
    assert 0 == outDim[0] % inDim[0];
    assert 0 == outDim[1] % inDim[1];
    final int kernelSizeX = outDim[0] / inDim[0];
    final int kernelSizeY = outDim[0] / inDim[0];
    for (int z = 0; z < outDim[2]; z++) {
      for (int y = 0; y < outDim[1]; y += kernelSizeY) {
        for (int x = 0; x < outDim[0]; x += kernelSizeX) {
          final double value = inputData.get(x / kernelSizeX, y / kernelSizeY, z);
          int xx = kernelSizeX / 2;
          int yy = kernelSizeY / 2;
          outputData.set(x + xx, y + yy, z, value);
          //          for (int xx = 0; xx < kernelSizeX; xx++) {
          //            for (int yy = 0; yy < kernelSizeY; yy++) {
          //              outputData.set(x + xx, y + yy, z, value);
          //            }
          //          }
        }
      }
    }
    inputData.freeRef();
    return outputData;
  }

  /**
   * Creates a new {@link UnpoolingLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link UnpoolingLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static UnpoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new UnpoolingLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    TensorArray data = fwd(batch, inputDims);
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(inputDims, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
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
  UnpoolingLayer addRef() {
    return (UnpoolingLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList batch, int[] inputDims) {
    Tensor outputDims = new Tensor(inputDims[0] * sizeX, inputDims[1] * sizeY, inputDims[2]);
    TensorArray data = new TensorArray(RefIntStream.range(0, batch.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor inputData = batch.get(dataIndex);
          Tensor temp_58_0002 = UnpoolingLayer.copyExpand(inputData.addRef(),
              outputDims.copy());
          inputData.freeRef();
          return temp_58_0002;
        }, outputDims.addRef(), batch.addRef()))
        .toArray(Tensor[]::new));
    outputDims.freeRef();
    batch.freeRef();
    return data;
  }

  /**
   * The Accumulator class is used to track the input dimensions and the accumulator result.
   * It also has a boolean value to track whether it is alive or not.
   *
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
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              return UnpoolingLayer.copyCondense(err, passback);
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
