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
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.IntArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.function.Function;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class MaxDropoutNoiseLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxDropoutNoiseLayer.class);
  @Nullable
  private final int[] kernelSize;
  private final Function<IntArray, com.simiacryptus.ref.wrappers.RefList<com.simiacryptus.ref.wrappers.RefList<Coordinate>>> getCellMap_cached = Util
      .cache(this::getCellMap);

  public MaxDropoutNoiseLayer() {
    this(2, 2);
  }

  public MaxDropoutNoiseLayer(@org.jetbrains.annotations.Nullable final int... dims) {
    super();
    kernelSize = dims;
  }

  protected MaxDropoutNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    kernelSize = JsonUtil.getIntArray(json.getAsJsonArray("kernelSize"));
  }

  @SuppressWarnings("unused")
  public static MaxDropoutNoiseLayer fromJson(@Nonnull final JsonObject json,
                                              com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new MaxDropoutNoiseLayer(json);
  }

  public static @SuppressWarnings("unused")
  MaxDropoutNoiseLayer[] addRefs(MaxDropoutNoiseLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayer::addRef)
        .toArray((x) -> new MaxDropoutNoiseLayer[x]);
  }

  public static @SuppressWarnings("unused")
  MaxDropoutNoiseLayer[][] addRefs(MaxDropoutNoiseLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayer::addRefs)
        .toArray((x) -> new MaxDropoutNoiseLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0];
    final TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    final Tensor[] mask = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nullable final Tensor input = data0.get(dataIndex);
      @Nullable final Tensor output = input.map(x -> 0);
      final com.simiacryptus.ref.wrappers.RefList<com.simiacryptus.ref.wrappers.RefList<Coordinate>> cells = getCellMap_cached
          .apply(new IntArray(output.getDimensions()));
      cells.forEach(cell -> {
        output.set(
            cell.stream().max(com.simiacryptus.ref.wrappers.RefComparator.comparingDouble(c -> input.get(c))).get(), 1);
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(
        new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(dataIndex -> {
          Tensor inputData = data0.get(dataIndex);
          @Nullable final double[] input = inputData.getData();
          @Nullable final double[] maskT = mask[dataIndex].getData();
          @Nonnull final Tensor output = new Tensor(inputData.getDimensions());
          @Nullable final double[] outputData = output.getData();
          for (int i = 0; i < outputData.length; i++) {
            outputData[i] = input[i] * maskT[i];
          }
          return output;
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (in0.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
              Tensor deltaTensor = delta.get(dataIndex);
              @Nullable final double[] deltaData = deltaTensor.getData();
              @Nonnull final int[] dims = data0.getDimensions();
              @Nullable final double[] maskData = mask[dataIndex].getData();
              @Nonnull final Tensor passback = new Tensor(dims);
              for (int i = 0; i < passback.length(); i++) {
                passback.set(i, maskData[i] * deltaData[i]);
              }
              return passback;
            }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return in0.isAlive() || !isFrozen();
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
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MaxDropoutNoiseLayer addRef() {
    return (MaxDropoutNoiseLayer) super.addRef();
  }

  private com.simiacryptus.ref.wrappers.RefList<com.simiacryptus.ref.wrappers.RefList<Coordinate>> getCellMap(
      @Nonnull final IntArray dims) {
    Tensor tensor = new Tensor(dims.data);
    return new com.simiacryptus.ref.wrappers.RefArrayList<>(tensor.coordStream(true)
        .collect(com.simiacryptus.ref.wrappers.RefCollectors.groupingBy((@Nonnull final Coordinate c) -> {
          int cellId = 0;
          int max = 0;
          for (int dim = 0; dim < dims.size(); dim++) {
            final int pos = c.getCoords()[dim] / kernelSize[dim];
            cellId = cellId * max + pos;
            max = dims.get(dim) / kernelSize[dim];
          }
          return cellId;
        })).values());
  }

}
