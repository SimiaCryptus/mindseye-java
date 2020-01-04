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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class UnpoolingLayer extends LayerBase {

  private final int sizeX;
  private final int sizeY;

  public UnpoolingLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  protected UnpoolingLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }

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
    return outputData;
  }

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
    return outputData;
  }

  @SuppressWarnings("unused")
  public static UnpoolingLayer fromJson(@Nonnull final JsonObject json,
                                        com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new UnpoolingLayer(json);
  }

  public static @SuppressWarnings("unused")
  UnpoolingLayer[] addRefs(UnpoolingLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(UnpoolingLayer::addRef)
        .toArray((x) -> new UnpoolingLayer[x]);
  }

  public static @SuppressWarnings("unused")
  UnpoolingLayer[][] addRefs(UnpoolingLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(UnpoolingLayer::addRefs)
        .toArray((x) -> new UnpoolingLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    Tensor outputDims;
    outputDims = new Tensor(inputDims[0] * sizeX, inputDims[1] * sizeY, inputDims[2]);
    TensorArray data = new TensorArray(
        com.simiacryptus.ref.wrappers.RefIntStream.range(0, batch.length()).parallel().mapToObj(dataIndex -> {
          Tensor inputData = batch.get(dataIndex);
          return UnpoolingLayer.copyExpand(inputData, outputDims.copy());
        }).toArray(i -> new Tensor[i]));
    return new Result(data, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      if (input.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, error.length()).parallel().mapToObj(dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              return UnpoolingLayer.copyCondense(err, passback);
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return new com.simiacryptus.ref.wrappers.RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  UnpoolingLayer addRef() {
    return (UnpoolingLayer) super.addRef();
  }

}
