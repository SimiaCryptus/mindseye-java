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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class MaxImageBandLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);

  public MaxImageBandLayer() {
    super();
  }

  protected MaxImageBandLayer(@Nonnull final JsonObject id, final int... kernelDims) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static MaxImageBandLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new MaxImageBandLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    assert 1 == inObj.length;
    final TensorList inputData = inObj[0].getData();
    inputData.length();
    @Nonnull
    final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    final Coordinate[][] maxCoords = inputData.stream().map(data -> {
      return com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDims[2]).mapToObj(band -> {
        return data.coordStream(true).filter(e -> e.getCoords()[2] == band)
            .max(com.simiacryptus.ref.wrappers.RefComparator.comparing(c -> data.get(c))).get();
      }).toArray(i -> new Coordinate[i]);
    }).toArray(i -> new Coordinate[i][]);

    return new Result(
        new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputData.length()).mapToObj(dataIndex -> {
          Tensor tensor = inputData.get(dataIndex);
          final com.simiacryptus.ref.wrappers.RefDoubleStream doubleStream = com.simiacryptus.ref.wrappers.RefIntStream
              .range(0, inputDims[2]).mapToDouble(band -> {
                final int[] maxCoord = maxCoords[dataIndex][band].getCoords();
                return tensor.get(maxCoord[0], maxCoord[1], band);
              });
          return new Tensor(1, 1, inputDims[2]).set(Tensor.getDoubles(doubleStream, inputDims[2]));
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (inObj[0].isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nonnull
                  final Tensor passback = new Tensor(inputData.getDimensions());
                  com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDims[2]).forEach(b -> {
                    final int[] maxCoord = maxCoords[dataIndex][b].getCoords();
                    passback.set(new int[] { maxCoord[0], maxCoord[1], b }, deltaTensor.get(0, 0, b));
                  });
                  return passback;
                }).toArray(i -> new Tensor[i]));
            inObj[0].accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public static @com.simiacryptus.ref.lang.RefAware class CalcRegionsParameter {
    public final int[] inputDims;
    public final int[] kernelDims;

    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }

    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      @Nonnull
      final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
      if (!com.simiacryptus.ref.wrappers.RefArrays.equals(inputDims, other.inputDims)) {
        return false;
      }
      return com.simiacryptus.ref.wrappers.RefArrays.equals(kernelDims, other.kernelDims);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + com.simiacryptus.ref.wrappers.RefArrays.hashCode(inputDims);
      result = prime * result + com.simiacryptus.ref.wrappers.RefArrays.hashCode(kernelDims);
      return result;
    }

  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") MaxImageBandLayer addRef() {
    return (MaxImageBandLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") MaxImageBandLayer[] addRefs(MaxImageBandLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxImageBandLayer::addRef)
        .toArray((x) -> new MaxImageBandLayer[x]);
  }

  public static @SuppressWarnings("unused") MaxImageBandLayer[][] addRefs(MaxImageBandLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MaxImageBandLayer::addRefs)
        .toArray((x) -> new MaxImageBandLayer[x][]);
  }
}
